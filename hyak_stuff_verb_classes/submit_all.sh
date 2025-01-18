#!/bin/bash
#SBATCH --job-name=train_3_all_sizes
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --export=all
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --mail-type=ALL

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load necessary modules
module load miniconda
conda activate synthetic_data_llm

# Your programs to run.
python3 -u main_all_test.py