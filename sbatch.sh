#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J manten-training

# Cluster Settings
#SBATCH -n 1       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 07:00:00 # 1-04:00:00 ## # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

# Define the paths for storing output and error files
#SBATCH --output=/hkfs/work/workspace/scratch/uqtlv-code_n_mamba/code/manten/outputs/%x_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/uqtlv-code_n_mamba/code/manten/outputs/%x_%j.err

# shellcheck disable=SC1091
source /home/hk-project-sustainebot/uqtlv/.mambainit
micromamba activate /hkfs/work/workspace/scratch/uqtlv-code_n_mamba/micromamba/envs/manten

accelerate launch --main_process_port 29872 --config_file /home/hk-project-sustainebot/uqtlv/.cache/huggingface/accelerate/default_config.yaml /hkfs/work/workspace/scratch/uqtlv-code_n_mamba/code/manten/manten/scripts/train.py experiment=tdda_c2d3d__mskill
