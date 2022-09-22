#!/usr/bin/env bash

#SBATCH --account=hai_fzj_inm1_bda
#SBATCH --output=logs/slurm/out_%j.job
#SBATCH --error=logs/slurm/err_%j.job
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00

source $HOME/.bashrc
echo 'Starting script'

echo "Enabling environment..."
source environment/activate.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_DEBUG=INFO

srun python scripts/train.py \
  name=testmodel \
  trainer=ddp \
  datamodule.num_workers=24 \

