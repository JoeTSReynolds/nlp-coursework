#!/bin/bash
#SBATCH --partition=t4                  
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=4               
#SBATCH --mem=32G                       
#SBATCH --time=48:00:00                 
#SBATCH --output=slurm_baseline_%j.out  
#SBATCH --error=slurm_baseline_%j.err

echo "Starting Baseline Training on $(hostname)"
source /vol/bitbucket/jtr23/venvs/modern_torch_venv/bin/activate

python train_kfold_optuna.py --save_prefix baseline

echo "Baseline Job completed successfully."