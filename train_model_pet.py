import os
import random
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import optuna
import numpy as np
from data_utils import load_training_and_validation
import sys
from pathlib import Path
from dotenv import load_dotenv
import time
import subprocess
from datetime import datetime

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
STUDY_DB_PATH = os.path.join(BASE_PATH, "pcl_optimization_pet_final.db")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best_pcl_model_pet_final.pt")
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
CHECKPOINT_PATH = os.path.join(BASE_PATH, f"checkpoint_job_{JOB_ID}_pet_final.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "roberta-large"
MAX_LEN = 192 # 95th percentile of length was 118, should be fine
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
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
       
        # make sure the text doesn't contain the special <mask> token (probably won't but just in case) as may confuse model
        text = text.replace(TOKENIZER.mask_token, "")

        # PET cloze prompt
        prompt_q = f"Question: Is the author being patronizing or condescending towards {kw}? Answer: {TOKENIZER.mask_token}"
        
        enc = TOKENIZER(
            text,
            prompt_q,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt'
        )
        
        binary_label = self.labels[idx]

        return {'input_ids': enc['input_ids'].flatten(),
                'attention_mask': enc['attention_mask'].flatten(),
                'labels': torch.tensor(binary_label, dtype=torch.long)}

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001, grace_epochs=3):
        self.patience = patience
        self.min_delta = min_delta
        self.grace_epochs = grace_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            if epoch >= self.grace_epochs:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0

# waits until there is space on gpu before running training
def wait_for_free_vram(min_gb_required=7.5, check_interval=60):
    print(f"[{JOB_ID}] VRAM Guard: Checking for {min_gb_required}GB of free space...")
    
    while True:
        try:
            res = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.free', 
                '--format=csv,nounits,noheader'
            ])
            free_mem_mb = int(res.decode().strip().split('\n')[0])
            free_gb = free_mem_mb / 1024
            
            if free_gb >= min_gb_required:
                print(f"[{JOB_ID}] Success: {free_gb:.2f}GB free. Starting training now.")
                break
            else:
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"[{current_time}] Only {free_gb:.2f}GB free. Waiting {check_interval}s...", end='\r')
                time.sleep(check_interval)
        except Exception as e:
            print(f"Error checking VRAM: {e}. Retrying in 10s...")
            time.sleep(10)

# prevents partial writes
def atomic_save(state, path):
    tmp_path = path + f".tmp_{JOB_ID}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def create_objective(train_ds, val_ds, num_class_0, num_class_1, study):
    # vocab IDs for the two target words (what the model will predict)
    c0_id = TOKENIZER(" No", add_special_tokens=False)['input_ids'][0]
    c1_id = TOKENIZER(" Yes", add_special_tokens=False)['input_ids'][0]
    
    def objective(trial):
        trial.set_user_attr("job_id", JOB_ID)

        alpha = trial.suggest_float("alpha", 0.3, 0.7)
        lr = trial.suggest_float("lr", 1e-6, 1.5e-5, log=True)
        wd = trial.suggest_float("weight_decay", 0.01, 0.05)
        
        # 2 lots of batch 4 with gradient accumulation to simulate batch 8 whilst saving VRAM
        bs = 4
        accum_steps = 2

        # dampened class weighting
        raw_w0 = (num_class_0 + num_class_1) / (2 * num_class_0 + 1e-8)
        scaled_w0 = 1.0 + alpha * (raw_w0 - 1.0)
        raw_w1 = (num_class_0 + num_class_1) / (2 * num_class_1 + 1e-8)
        scaled_w1 = 1.0 + alpha * (raw_w1 - 1.0)
        class_weights = torch.tensor([scaled_w0, scaled_w1], dtype=torch.float).to(DEVICE)
        print(f"\n[Job {JOB_ID}] Trial {trial.number} Params: alpha={alpha:.4f}, lr={lr:.2e}, weight_decay={wd:.4f}, scaled_w0={scaled_w0:.4f}, scaled_w1={scaled_w1:.4f}")    

        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).float().to(DEVICE)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = 10
        total_steps = (len(train_loader) * epochs) // accum_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopper = EarlyStopping(patience=2)

        scaler = torch.cuda.amp.GradScaler()  # for mixed precision training, for VRAM and speed benefits

        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            if checkpoint.get('trial_number') == trial.number:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                early_stopper.best_score = checkpoint.get('best_trial_f1', None) # resume early stopping tracking
                print(f"\n[Job {JOB_ID}] Resuming Trial {trial.number} from Epoch {start_epoch}")
            del checkpoint

        trial_best_f1 = -1.0
        if start_epoch > 0 and early_stopper.best_score is not None:
            trial_best_f1 = early_stopper.best_score
        for epoch in range(start_epoch, epochs):
            model.train() 
            train_pbar = tqdm(train_loader, desc=f"Job {JOB_ID} T{trial.number} E{epoch}", leave=False)
            
            optimizer.zero_grad() 
            
            for step, batch in enumerate(train_pbar):
                ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                
                # mixed precision forward pass
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(ids, attention_mask=mask)
                    seq_logits = outputs.logits
                    
                    mask_indices = (ids == TOKENIZER.mask_token_id).long().argmax(dim=-1)
                    mask_logits = seq_logits[torch.arange(seq_logits.size(0), device=DEVICE), mask_indices]
                    verb_logits = mask_logits[:, [c0_id, c1_id]]
                    
                    loss = criterion(verb_logits, lbls)
                    # divide loss by accum_steps so the total sum represents a single batch of 8
                    loss = loss / accum_steps 
                
                # backward pass with gradient scaling for mixed precision
                scaler.scale(loss).backward()
                
                # step the optimiser every accum_steps or at the end of the dataloader
                if ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader)):
                    scale_before = scaler.get_scale()

                    scaler.step(optimizer)
                    scaler.update()
                    
                    # only scale if optimiser actually updated the weights (sometimes doesnt for nan)
                    if scale_before <= scaler.get_scale():
                        scheduler.step()
                        
                    optimizer.zero_grad()

                train_pbar.set_postfix({'loss': f"{(loss.item() * accum_steps):.4f}"})

            # evaluation
            model.eval()
            p, t = [], []
            val_loss = 0
            with torch.no_grad():
                for b in val_loader:
                    ids, mask, lbls = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE), b['labels'].to(DEVICE)
                    
                    outputs = model(ids, attention_mask=mask)
                    seq_logits = outputs.logits
                    
                    mask_indices = (ids == TOKENIZER.mask_token_id).long().argmax(dim=-1)
                    mask_logits = seq_logits[torch.arange(seq_logits.size(0), device=DEVICE), mask_indices]
                    verb_logits = mask_logits[:, [c0_id, c1_id]]
                    
                    loss = criterion(verb_logits, lbls)
                    val_loss += loss.item()

                    p.extend(torch.argmax(verb_logits, dim=1).cpu().numpy())
                    t.extend(lbls.cpu().numpy())
            
            precision, recall, f1, _ = precision_recall_fscore_support(t, p, average='binary', zero_division=0)
            acc = accuracy_score(t, p)
            avg_val_loss = val_loss / len(val_loader)

            print(f"\n[{JOB_ID}] Trial {trial.number} Epoch {epoch} Report:")
            print(f"  -> Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
            print(f"  -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

            try:
                best_so_far = study.best_value
            except ValueError:
                best_so_far = -1.0

            if f1 > trial_best_f1:
                trial_best_f1 = f1
                
                # only save if best of trial and best in study
                if f1 > best_so_far:
                    atomic_save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"[Job {JOB_ID}] *** New Global Best F1: {f1:.4f} (Saved to {MODEL_SAVE_PATH}) ***")
            
            early_stopper(f1, epoch)

            # save checkpoint
            atomic_save({
                'trial_number': trial.number,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'f1': f1,
                'best_trial_f1': early_stopper.best_score
            }, CHECKPOINT_PATH)
            print(f"[Job {JOB_ID}] Saved checkpoint epoch {epoch} F1: {f1:.4f} (Saved to {CHECKPOINT_PATH})")

            trial.report(f1, epoch)
            
            # only prune if defo not best score
            if early_stopper.best_score <= best_so_far:
                if trial.should_prune():
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                    raise optuna.exceptions.TrialPruned()
            
            if early_stopper.early_stop:
                print(f"[{JOB_ID}] Early stopping triggered at epoch {epoch}")
                break

        del model
        torch.cuda.empty_cache()
        gc.collect()

        if early_stopper.best_score is not None:
            print(f"[{JOB_ID}] Trial {trial.number} Best F1: {early_stopper.best_score:.4f} at epoch {epoch - early_stopper.counter}")
            return early_stopper.best_score

        return f1
    
    return objective

if __name__ == "__main__":
    wait_for_free_vram(min_gb_required=6)

    train_df, val_df = load_training_and_validation() # uses augmented datasets

    num_class_0 = len(train_df[train_df['binary_label'] == 0])
    num_class_1 = len(train_df[train_df['binary_label'] == 1])
    print(f"Training Class distribution: Class 0 = {num_class_0}, Class 1 = {num_class_1}, Ratio = 1:{num_class_0 / (num_class_1 + 1e-8):.2f}")

    train_ds = PCLDataset(train_df['text'].tolist(), train_df['keyword'].tolist(), train_df['binary_label'].tolist())
    val_ds = PCLDataset(val_df['text'].tolist(), val_df['keyword'].tolist(), val_df['binary_label'].tolist())

    study = optuna.create_study(
        study_name="pcl_pet_binary",
        storage=f"sqlite:///{os.path.abspath(STUDY_DB_PATH)}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    # This is for if a job was interrupted then restarted
    existing_trials = [
        t for t in study.trials 
        if (t.state.name == "FAIL" or t.state.name == "RUNNING") 
        and t.user_attrs.get("job_id") == JOB_ID
    ]
    
    target_params = None
    if existing_trials:
        last_trial = sorted(existing_trials, key=lambda x: x.number)[-1]
        target_params = last_trial.params
        print(f"[{JOB_ID}] Found failed/interrupted Trial {last_trial.number}. Re-using params: {target_params}")

    if target_params:
        study.enqueue_trial(target_params) # basically the next trial uses the same params and resumes epochs, loading in the checkpoint


    obj_func = create_objective(train_ds, val_ds, num_class_0, num_class_1, study)
    try:
        study.optimize(obj_func, n_trials=20)
    except KeyboardInterrupt:
        print("Got keyboard interrupt, saved progress")
        sys.exit(0)

    print(f"\nFinished trials, check db for best params.")
