#!/bin/bash
#SBATCH --job-name=LP_2416
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=128
#SBATCH --time=7-00:00:00
#SBATCH --mem=1000GB
#SBATCH --output=/home/yu79deg/VIBE/bash/bash_output/LP_2416.log
#SBATCH --error=/home/yu79deg/VIBE/bash/bash_output/LP_2416.err
#SBATCH --chdir=/home/yu79deg/VIBE

# --- Micromamba activation ---
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate VIBE_env

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
python -X faulthandler -X dev -u /home/yu79deg/VIBE/src/vibe/VIBE.py --yaml /home/yu79deg/VIBE/yamls/LP_2416.yaml
