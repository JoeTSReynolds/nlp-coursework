#!/bin/bash
#SBATCH --partition=t4                  
#SBATCH --gres=gpu:1                    
#SBATCH --cpus-per-task=4               
#SBATCH --mem=32G                       
#SBATCH --time=48:00:00                 
#SBATCH --output=slurm_roberta_%j.out  
#SBATCH --error=slurm_roberta_%j.err

echo "Starting RoBERTa Baseline Training on $(hostname)"
source /vol/bitbucket/jtr23/venvs/modern_torch_venv/bin/activate

python train_kfold_optuna.py --model_override roberta-large --save_prefix roberta

echo "RoBERTa Job completed successfully."