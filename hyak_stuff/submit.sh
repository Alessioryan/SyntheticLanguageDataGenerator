#!/bin/bash
#SBATCH --job-name=million_two
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
#SBATCH --output=sh_output_mil_two
#SBATCH --error=sh_error_mil_two


# Modules to use
conda activate /gscratch/zlab/tosolini/SyntheticallyTrainedLLM_Hyak/mynlpenv

# Your programs to run.
python3 main_multitest.py