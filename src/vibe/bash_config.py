from pathlib import Path
import os

# --------------------------------------------------
# Resolve repository root automatically
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
# src/vibe/bash_config.py → parents[2] = repo root

YAML_DIR = REPO_ROOT / "yamls"
SRC_DIR  = REPO_ROOT / "src"
BASH_DIR = REPO_ROOT / "bash"
LOG_DIR  = BASH_DIR / "bash_output"

LOG_DIR.mkdir(parents=True, exist_ok=True)



simulation_template = '''#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --chdir={workdir}
{extra_directives}
# --- Micromamba activation ---
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate {env_name}

# --- Debug/diagnostic env for tracking down native (C-level) crashes ---
#   PYTHONFAULTHANDLER=1  -> dump a Python + C stack trace on SIGSEGV / SIGABRT
#   PYTHONUNBUFFERED=1    -> never lose the last print before a crash
#   MALLOC_CHECK_=3       -> glibc heap checker: SIGABRT with location on double
#                            free / heap corruption (rather than a silent segfault)
#   HDF5_USE_FILE_LOCKING=FALSE -> prevents h5py from stalling / crashing on shared
#                                   filesystems (BeeGFS/Lustre) when the DABAM .h5
#                                   files are opened concurrently.
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export MALLOC_CHECK_=3
export HDF5_USE_FILE_LOCKING=FALSE

# `-X faulthandler` and `-X dev` complement the env-var above (extra dev-mode
# warnings + guaranteed faulthandler regardless of env stripping); `-u` is
# unbuffered stdout (redundant with PYTHONUNBUFFERED but harmless).
python -X faulthandler -X dev -u {script_path} {script_kwargs}
'''


# One SLURM script for a whole parameter scan: a bash-array of YAMLs run as a
# job ARRAY. The "%N" in --array throttles concurrency to N tasks at a time.
array_template = '''#!/bin/bash
#SBATCH --job-name={jobname}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --chdir={workdir}
#SBATCH --array={array_spec}

# --- Micromamba activation ---
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate {env_name}

# --- Debug/diagnostic env (see write_bash's template for the rationale) ---
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export MALLOC_CHECK_=3
export HDF5_USE_FILE_LOCKING=FALSE

# ================================================================
#  Parameter scan as a SLURM job ARRAY (bash-array of YAMLs).
#  One array task == one YAML; the "%{throttle}" in --array above
#  caps how many run at once so the cluster stays free for others.
#  Edit the list / the "%N" to run only a few or change the load.
# ================================================================
YAMLS=(
{yaml_lines}
)

YAML="${{YAMLS[$SLURM_ARRAY_TASK_ID]}}"
echo "[array] task ${{SLURM_ARRAY_TASK_ID}} of ${{#YAMLS[@]}}  ->  ${{YAML}}"

python -X faulthandler -X dev -u {script_path} --yaml "${{YAML}}" {res_arg}
'''



DEFAULT_SBATCH_PARAMS = {
    'partition': 'hij-gpu',
    'n_cpus': 24,
    'time': '24:00:00',
    'mem': '100GB',
    'env_name': 'VIBE_env',
    'script_path': str(SRC_DIR / "vibe" / "VIBE.py"),
    'workdir': str(REPO_ROOT),
}


def estimate_mem_gb(yaml_path, coeff_bytes_px=3100, floor_gb=64, round_to=10):
    """Safe --mem (GB) for one yaml, from its grid size N.

    VIBE (now pyfftw complex64) peaks at ~2360 bytes/px * N**2 (measured at
    N=8000). The default 3100 bytes/px keeps ~30% headroom. Returns None if N
    can't be determined. NOTE: a request above a node's RAM (~1031 GB here) will
    be rejected by SLURM -- the caller should skip sims whose estimate exceeds it.
    """
    import math
    import yaml as _yaml
    yp = Path(yaml_path)
    if not yp.is_absolute():
        yp = YAML_DIR / yp
    if yp.suffix != ".yaml":
        yp = yp.with_suffix(".yaml")
    if not yp.exists():
        return None
    sim = (_yaml.safe_load(yp.read_text()) or {}).get("simulation", {})
    res, ps = sim.get("Resolution"), sim.get("propsize")
    if isinstance(res, (int, float)) and res > 10:
        N = int(round(res))                       # Resolution given directly as N grid cells
    elif isinstance(res, (int, float)) and 0 < res <= 10 and ps:
        N = int(round(ps * 1e6 / res))            # Resolution as um/pixel -> N = propsize / res
    elif sim.get("N"):
        N = int(sim["N"])
    else:
        return None
    gb = coeff_bytes_px * N * N / 1e9
    return int(math.ceil(max(gb, floor_gb) / round_to) * round_to)


def estimate_resources(yaml_path, cpus=4, fallback_mem_gb=200, **kw):
    """Right-sized SLURM resources for ONE yaml -> {'mem': 'NNNG', 'n_cpus': cpus}.

    VIBE runs single-threaded (measured ~1.0 core), so a few CPUs are plenty; memory
    is the real driver and is estimated from the grid size (see estimate_mem_gb).
    """
    gb = estimate_mem_gb(yaml_path, **kw)
    return {"mem": f"{gb if gb is not None else fallback_mem_gb}G", "n_cpus": int(cpus)}


def write_bash(path, resolution, upd_params={}, bash_name=None,
               jobname=None, dependency=None, log_stem=None, comment=None):

    bash_params = DEFAULT_SBATCH_PARAMS.copy()
    if upd_params:
        bash_params.update(upd_params)

    if 'yaml' not in bash_params:
        raise ValueError("You must specify 'yaml' in upd_params.")

    yaml_arg = Path(bash_params['yaml'])
    if not yaml_arg.is_absolute():
        yaml_arg = YAML_DIR / yaml_arg

    # resolution: None -> use the Resolution written in the YAML; a number overrides it
    # (>10 = N grid cells, <=10 = um/pixel).
    res_arg = "" if resolution is None else f"--resolution {resolution}"
    bash_params['script_kwargs'] = f'--yaml {yaml_arg} {res_arg}'.strip()
    # SLURM --job-name (may be a shared "lane" name for singleton throttling); logs and
    # the .slurm file still key off the yaml stem so they stay per-simulation.
    bash_params['jobname'] = jobname or yaml_arg.stem
    stem = log_stem or yaml_arg.stem
    bash_params['log_out'] = str(LOG_DIR / f"{stem}.log")
    bash_params['log_err'] = str(LOG_DIR / f"{stem}.err")
    # singleton throttling forces a shared --job-name (the lane), so stamp the real sim
    # in --comment to keep jobs identifiable in `squeue -o "...%k..."`.
    _extra = []
    if dependency:
        _extra.append(f"#SBATCH --dependency={dependency}")
    if comment:
        _extra.append(f"#SBATCH --comment={comment}")
    bash_params['extra_directives'] = "\n".join(_extra)

    bash_script = simulation_template.format(**bash_params)

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    path.mkdir(parents=True, exist_ok=True)


    bash_path = path / (bash_name or "job.slurm")
    with open(bash_path, 'w') as f:
        f.write(bash_script)

    return str(bash_path)


def write_bash_array(path, yamls, resolution=None, throttle=3,
                     upd_params={}, bash_name=None, jobname=None):
    """Write ONE SLURM array script that runs `yamls` as a job array, throttled
    to `throttle` concurrent tasks (``--array=0-(N-1)%throttle``). Submit it with
    a single ``sbatch``. Pass a short `yamls` list to run only a few simulations.

    This replaces the "one sbatch per yaml" loop: the whole scan lives in a single
    inspectable bash-array script, and at most `throttle` tasks ever run at once
    so other cluster users keep their share. (`submitit` automates the same idea
    from pure Python.)
    """
    if not yamls:
        raise ValueError("`yamls` is empty -- nothing to run.")

    bash_params = DEFAULT_SBATCH_PARAMS.copy()
    if upd_params:
        bash_params.update(upd_params)
    bash_params.pop('yaml', None)      # not used for arrays

    yaml_paths = []
    for y in yamls:
        yp = Path(y)
        if not yp.is_absolute():
            yp = YAML_DIR / yp
        if yp.suffix != ".yaml":
            yp = yp.with_suffix(".yaml")
        yaml_paths.append(str(yp))

    n = len(yaml_paths)
    throttle = max(1, int(throttle))
    stems = [Path(p).stem for p in yaml_paths]
    if jobname is None:
        jobname = f"{stems[0]}_{stems[-1]}" if n > 1 else stems[0]

    # mem='auto' -> size the (single, shared) array --mem to the batch's largest yaml.
    if str(bash_params.get('mem', '')).strip().lower() == 'auto':
        gbs = [g for g in (estimate_mem_gb(p) for p in yaml_paths) if g]
        bash_params['mem'] = f"{max(gbs)}G" if gbs else DEFAULT_SBATCH_PARAMS['mem']

    bash_params['jobname']    = jobname
    bash_params['throttle']   = throttle
    bash_params['array_spec'] = f"0-{n - 1}%{throttle}"
    bash_params['log_out']    = str(LOG_DIR / f"{jobname}_%A_%a.log")
    bash_params['log_err']    = str(LOG_DIR / f"{jobname}_%A_%a.err")
    bash_params['res_arg']    = "" if resolution is None else f"--resolution {resolution}"
    bash_params['yaml_lines'] = "\n".join(f'    "{p}"' for p in yaml_paths)

    bash_script = array_template.format(**bash_params)

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)

    bash_path = path / (bash_name or f"{jobname}_array.slurm")
    with open(bash_path, 'w') as f:
        f.write(bash_script)

    return str(bash_path)
