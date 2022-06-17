#!/bin/bash

#SBATCH --job-name=proto
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=1600  
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00 
#SBATCH --output=./slurm_logs/reppro-%A_%a.txt

date
echo ${USER}
eval "$(conda shell.bash hook)"
ml load cuda/11.0
conda activate xmodal

srun python train_proto.py --mode a2i --cuda 1 --vis_m 18 --aud_m aud_pre --sharing False --reptile True --meta-iterations 200 
