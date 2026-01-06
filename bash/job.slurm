#!/bin/bash
#SBATCH --job-name=LP_986
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=600GB
#SBATCH --output=/home/yu79deg/VIBE/bash/bash_output/LP_986.log
#SBATCH --error=/home/yu79deg/VIBE/bash/bash_output/LP_986.err
#SBATCH --chdir=/home/yu79deg/VIBE

# --- Micromamba activation ---
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate VIBE_env

python /home/yu79deg/VIBE/src/vibe/VIBE.py -N 1000 --yaml /home/yu79deg/VIBE/yamls/LP_986.yaml
