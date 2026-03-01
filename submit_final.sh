#!/bin/bash
#SBATCH --partition=a40                 # Request an A40 GPU partition
#SBATCH --gres=gpu:1                    # Request exactly 1 GPU
#SBATCH --cpus-per-task=4               # Request 4 CPU cores for data loading
#SBATCH --mem=32G                       # Request 32GB of RAM
#SBATCH --time=48:00:00                 # Maximum run time (HH:MM:SS) - Note: Max is 3 days
#SBATCH --output=slurm_console_%j.out   # Captures standard output (the %j inserts the Job ID)
#SBATCH --error=slurm_error_%j.err      # Captures standard errors/crashes

echo "Job starting on $(hostname)"
echo "Loading virtual environment..."

source /vol/bitbucket/jtr23/venvs/modern_torch_venv/bin/activate

echo "Environment activated. Starting Python script..."

python train_kfold_optuna.py --tapt --save_prefix tapt

echo "Job completed."