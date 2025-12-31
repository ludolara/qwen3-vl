#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/job_error.log
#SBATCH --error=logs/job_output.log
#SBATCH --nodes=1
#SBATCH --mem=32G 
#SBATCH --cpus-per-gpu=3
#SBATCH --gres=gpu:a100l:4
#SBATCH --partition=short-unkillable
#SBATCH --time=03:00:00

source .venv/bin/activate
python src/train.py
