#!/usr/bin/env bash

#SBATCH --account=jinm14
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=4
#SBATCH --time=10:00:00

source "$HOME"/.bashrc
echo 'Starting script'

echo "Enabling environment..."
source "$HOME"/modules.sh
conda deactivate
conda deactivate
conda activate dl

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

srun python scripts/train.py \
  experiment=brain_unet_3d
