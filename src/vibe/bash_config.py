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

# --- Micromamba activation ---
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate {env_name}

python {script_path} {script_kwargs}
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



def write_bash(path, N, upd_params={}, bash_name=None):

    bash_params = DEFAULT_SBATCH_PARAMS.copy()
    if upd_params:
        bash_params.update(upd_params)

    if 'yaml' not in bash_params:
        raise ValueError("You must specify 'yaml' in upd_params.")

    yaml_arg = Path(bash_params['yaml'])
    if not yaml_arg.is_absolute():
        yaml_arg = YAML_DIR / yaml_arg

    bash_params['script_kwargs'] = f'-N {N} --yaml {yaml_arg}'
    bash_params['jobname'] = yaml_arg.stem

    bash_params['log_out'] = str(LOG_DIR / f"{bash_params['jobname']}.log")
    bash_params['log_err'] = str(LOG_DIR / f"{bash_params['jobname']}.err")

    bash_script = simulation_template.format(**bash_params)

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    path.mkdir(parents=True, exist_ok=True)


    bash_path = path / (bash_name or "job.slurm")
    with open(bash_path, 'w') as f:
        f.write(bash_script)

    return str(bash_path)
