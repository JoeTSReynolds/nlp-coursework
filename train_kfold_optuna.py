import argparse
import os
import random
import gc
import torch
import torch.nn as nn
import pandas as pd
import optuna
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
from pathlib import Path
from dotenv import load_dotenv
from cluster_utils import RUN_ON_CLUSTER, setup_logger, smart_vram_guard

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PCL 5-Fold Training Pipeline")
parser.add_argument("--tapt", action="store_true", help="Use the local TAPT model instead of the base HuggingFace model")
parser.add_argument("--save_prefix", type=str, default="deberta", help="Prefix for the saved model files (e.g., 'baseline' or 'tapt')")
parser.add_argument("--model_override", type=str, default=None, help="Override the default DeBERTa model")
args = parser.parse_args()

log_path = os.path.join(BASE_PATH, f"{args.save_prefix}_optuna_kfold_run.log")
logger = setup_logger(log_path)
logger.info("=== STARTING NEW TRAINING PIPELINE ===")

if args.model_override:
    MODEL_NAME = args.model_override
    logger.info(f"Loading OVERRIDE model: {MODEL_NAME}")
elif args.tapt:
    MODEL_NAME = os.path.join(BASE_PATH, "deberta-v3-tapt")
    logger.info(f"Loading local TAPT model from: {MODEL_NAME}")
else:
    MODEL_NAME = "microsoft/deberta-v3-large"
    logger.info(f"Loading Hugging Face model: {MODEL_NAME}")

MAX_LEN = 192
SEED = 42
N_FOLDS = 5

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

class PCLDataset(Dataset):
    def __init__(self, texts, keywords, labels):
        self.texts, self.keywords, self.labels = texts, keywords, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, kw = str(self.texts[idx]), str(self.keywords[idx])
        text = text.replace(TOKENIZER.mask_token, "")

        prompt_q = f"Question: Is the author being patronizing or condescending towards {kw}? Answer: {TOKENIZER.mask_token}"

        enc = TOKENIZER(
            text, prompt_q, add_special_tokens=True, max_length=MAX_LEN,
            padding='max_length', truncation='only_first', return_tensors='pt'
        )

        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

def get_dataloaders(fold, bs):
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"train_fold_{fold}.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, f"val_fold_{fold}.csv"))

    train_ds = PCLDataset(train_df['text'].tolist(), train_df['keyword'].tolist(), train_df['binary_label'].tolist())
    val_ds = PCLDataset(val_df['text'].tolist(), val_df['keyword'].tolist(), val_df['binary_label'].tolist())

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    
    num_class_0 = len(train_df[train_df['binary_label'] == 0])
    num_class_1 = len(train_df[train_df['binary_label'] == 1])
    
    return train_loader, val_loader, num_class_0, num_class_1

def train_eval_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, epochs, trial=None):
    c0_id = TOKENIZER(" No", add_special_tokens=False)['input_ids'][0]
    c1_id = TOKENIZER(" Yes", add_special_tokens=False)['input_ids'][0]
    
    best_f1 = -1.0
    best_weights = None
    patience_counter = 0
    PATIENCE_LIMIT = 2
    EARLY_STOPPER_GRACE = 3

    for epoch in range(epochs):
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, disable=RUN_ON_CLUSTER)
        optimizer.zero_grad()

        for step, batch in enumerate(train_pbar):
            ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(ids, attention_mask=mask)
                seq_logits = outputs.logits
                mask_indices = (ids == TOKENIZER.mask_token_id).long().argmax(dim=-1)
                verb_logits = seq_logits[torch.arange(seq_logits.size(0), device=DEVICE), mask_indices][:, [c0_id, c1_id]]
                loss = criterion(verb_logits, lbls)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        model.eval()
        p, t = [], []
        with torch.no_grad():
            for b in val_loader:
                ids, mask, lbls = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE), b['labels'].to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(ids, attention_mask=mask)
                    seq_logits = outputs.logits
                    mask_indices = (ids == TOKENIZER.mask_token_id).long().argmax(dim=-1)
                    verb_logits = seq_logits[torch.arange(seq_logits.size(0), device=DEVICE), mask_indices][:, [c0_id, c1_id]]
                
                p.extend(torch.argmax(verb_logits, dim=1).cpu().numpy())
                t.extend(lbls.cpu().numpy())

        _, _, f1, _ = precision_recall_fscore_support(t, p, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            # Save weights in RAM to avoid constant disk writing
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            if epoch >= EARLY_STOPPER_GRACE:  # Start early stopping checks after a grace period
                patience_counter += 1

        if trial:
            trial.report(f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        if trial is None and patience_counter >= PATIENCE_LIMIT:
            logger.info(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE_LIMIT} epochs.")
            break   

    return best_f1, best_weights

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.40, 0.60)
    lr = trial.suggest_float("lr", 3e-6, 9e-6, log=True)
    wd = trial.suggest_float("weight_decay", 0.02, 0.06)
    epochs = 10
    bs = 8
    
    logger.info(f"\n[Trial {trial.number}] Testing params: lr={lr:.2e}, alpha={alpha:.2f}, wd={wd:.3f}")
    
    train_loader, val_loader, num_class_0, num_class_1 = get_dataloaders(fold=0, bs=bs)

    raw_w0 = (num_class_0 + num_class_1) / (2 * num_class_0 + 1e-8)
    scaled_w0 = 1.0 + alpha * (raw_w0 - 1.0)
    raw_w1 = (num_class_0 + num_class_1) / (2 * num_class_1 + 1e-8)
    scaled_w1 = 1.0 + alpha * (raw_w1 - 1.0)
    class_weights = torch.tensor([scaled_w0, scaled_w1], dtype=torch.float).to(DEVICE)

    logger.info(f"Class distribution in Fold 0: Class 0 = {num_class_0}, Class 1 = {num_class_1}, Ratio = {num_class_1 / (num_class_0 + 1e-8):.2f} -> Scaled Weights: Class 0 = {scaled_w0:.2f}, Class 1 = {scaled_w1:.2f}")

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).float().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    total_steps = (len(train_loader) * epochs) // 2
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')

    best_f1, _ = train_eval_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, epochs, trial)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_f1

if __name__ == "__main__":
    logger.info("=========================================")
    logger.info(" PHASE 1: OPTUNA TUNING (FOLD 0 ONLY)    ")
    logger.info("=========================================")
    
    # Prune after 3 epochs (n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=10) # 10 tightly focused trials
    
    best_params = study.best_params
    logger.info(f"\n[WINNING PARAMS] Found best hyperparameters: {best_params}")

    logger.info("\n=========================================")
    logger.info(" PHASE 2: 5-FOLD ENSEMBLE TRAINING       ")
    logger.info("=========================================")
    
    bs = 8
    epochs = 10

    for fold in range(N_FOLDS):
        logger.info(f"\n---> Training Fold {fold}/{N_FOLDS-1}")
        
        train_loader, val_loader, num_class_0, num_class_1 = get_dataloaders(fold=fold, bs=bs)

        raw_w0 = (num_class_0 + num_class_1) / (2 * num_class_0 + 1e-8)
        scaled_w0 = 1.0 + best_params["alpha"] * (raw_w0 - 1.0)
        raw_w1 = (num_class_0 + num_class_1) / (2 * num_class_1 + 1e-8)
        scaled_w1 = 1.0 + best_params["alpha"] * (raw_w1 - 1.0)
        class_weights = torch.tensor([scaled_w0, scaled_w1], dtype=torch.float).to(DEVICE)
        logger.info(f"Fold {fold} class distribution: Class 0 = {num_class_0}, Class 1 = {num_class_1}, Ratio = {num_class_1 / (num_class_0 + 1e-8):.2f} -> Scaled Weights: Class 0 = {scaled_w0:.2f}, Class 1 = {scaled_w1:.2f}")

        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).float().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        
        total_steps = (len(train_loader) * epochs) // 2
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scaler = torch.amp.GradScaler('cuda')

        best_f1, best_weights = train_eval_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, epochs)
        
        # Save the best model strictly to disk
        save_path = os.path.join(BASE_PATH, f"{args.save_prefix}_deberta_fold_{fold}.pt")
        torch.save(best_weights, save_path)
        logger.info(f"[Fold {fold} Complete] Best F1: {best_f1:.4f} -> Saved to {save_path}")
        
        del model
        del best_weights
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\n[SUCCESS] Pipeline complete! Your 5 TAPT-DeBERTa models are ready for evaluation.")