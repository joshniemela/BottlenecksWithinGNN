#!/bin/bash
#SBATCH --partition=modi_short
#SBATCH --cpus-per-task=4

source $CONDA_DIR/etc/profile.d/conda.sh
modi-load-environments
#conda init
conda activate pyg_wandb_env


# Set the WANDB_API_KEY environment variable
export WANDB_API_KEY=24ed862a06966b5d61ad21b3bcc9713a0452da9b

wandb agent mustafahekmat/TreeBottleneckReproduction/0c7tvqaa
