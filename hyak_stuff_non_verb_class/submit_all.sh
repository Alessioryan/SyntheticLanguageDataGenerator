#!/bin/bash
#SBATCH --job-name=train_3_all_sizes
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tosolini@uw.edu

#SBATCH --account=zlab
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

#SBATCH --export=all
#SBATCH --output=sh_output
#SBATCH --error=sh_error

# Your programs to run.
srun --label python3 main_all_test.py "${@}"
