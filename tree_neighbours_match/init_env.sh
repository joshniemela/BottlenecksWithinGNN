#!/bin/bash
#SBATCH --partition=modi_short
#SBATCH --cpus-per-task=4


source $CONDA_DIR/etc/profile.d/conda.sh
modi-load-environments
conda create --name pyg_wandb_env python=3.9 
conda activate pyg_wandb_env
conda install -y pytorch cpuonly -c pytorch
conda install -y pyg -c pyg
pip install wandb


