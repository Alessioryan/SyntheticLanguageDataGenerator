#!/bin/bash
#SBATCH --job-name=one-regular-paradigm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tosolini@uw.edu

#SBATCH --account=UNSURE
#SBATCH --partition=TBD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=UNSURE
#SBATCH --mem=20G
#SBATCH --gpus=gpu-l40:1
#SBATCH --time=24:00:00

#SBATCH --chdir=UNSURE
#SBATCH --export=all
#SBATCH --output=TBD
#SBATCH --error=TBD

# Modules to use (optional).
module load python/3.8.2
module load anaconda
module load transformers

# Your programs to run.
python3
