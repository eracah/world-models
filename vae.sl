#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH -o batch_results/"slurm-%j.out"
#SBATCH -e batch_results/"slurm-%j.out"
xvfb-run -s "-screen 0 1400x900x24" python vae.py $@
