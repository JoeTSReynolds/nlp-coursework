#!/bin/bash
#SBATCH --partition=a100                # Request an A100 GPU (change to a40 if a100 queue is too long)
#SBATCH --gres=gpu:1                    # 1 GPU
#SBATCH --cpus-per-task=8               # 8 CPU cores for fast tokenization/data loading
#SBATCH --mem=64G                       # 64GB of RAM
#SBATCH --time=48:00:00                 # Maximum run time
#SBATCH --output=slurm_tapt_%j.out      
#SBATCH --error=slurm_tapt_%j.err

echo "Starting TAPT Job on $(hostname)"
source /vol/bitbucket/jtr23/venvs/modern_torch_venv/bin/activate

echo "Environment activated. Checking GPU availability..."
nvidia-smi

echo "Starting Task-Adaptive Pretraining..."
python run_tapt.py

echo "TAPT Job completed successfully."