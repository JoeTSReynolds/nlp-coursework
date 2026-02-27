import os
import random
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import optuna
import numpy as np
from keyword_imbalance import load_and_merge_data
import sys
import time
import subprocess
from datetime import datetime

BASE_PATH = "/vol/bitbucket/jtr23/nlp/"
STUDY_DB_PATH = os.path.join(BASE_PATH, "pcl_optimization_binary.db")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best_pcl_model_binary.pt")
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
CHECKPOINT_PATH = os.path.join(BASE_PATH, f"checkpoint_job_{JOB_ID}.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 192 #256
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

TEMPLATE_POOL = [
    "This text is patronising and condescending towards {keyword}.",
    "The author shows a superior attitude toward {keyword}.",
    "This paragraph depicts {keyword} in a way that is pitying or belittling.",
    "The writing style regarding {keyword} creates a 'saviour' dynamic.",
    "This content treats {keyword} as vulnerable or inferior."
]

class PCLDataset(Dataset):
    def __init__(self, texts, keywords, labels, is_train=True):
        self.texts, self.keywords, self.labels, self.is_train = texts, keywords, labels, is_train
    def __len__(self): 
        return len(self.texts)
    def __getitem__(self, idx):
        text, kw = str(self.texts[idx]), str(self.keywords[idx])
        hypo = random.choice(TEMPLATE_POOL).format(keyword=kw) if self.is_train else TEMPLATE_POOL[0].format(keyword=kw)
        enc = TOKENIZER(text, hypo, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].flatten(), 'attention_mask': enc['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
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
            # Query nvidia-smi for free memory in MiB
            res = subprocess.check_output([
                'nvidia-smi', '--query-gpu=memory.free', 
                '--format=csv,nounits,noheader'
            ])
            # If multiple GPUs, this takes the first one (index 0)
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

def create_objective(train_ds, val_ds, weights, study):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 8e-6, log=True)
        wd = trial.suggest_float("weight_decay", 0.01, 0.05)
        bs = trial.suggest_categorical("batch_size", [8, 16])

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).float().to(DEVICE)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = 10
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss(weight=weights)
        early_stopper = EarlyStopping(patience=2)

        # resume trial if checkpoint exists
        start_epoch = 0
        if os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            if checkpoint.get('trial_number') == trial.number:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"\n[Job {JOB_ID}] Resuming Trial {trial.number} from Epoch {start_epoch}")
            del checkpoint

        # training
        final_f1 = 0.0
        for epoch in range(start_epoch, epochs):
            model.train() 
            train_pbar = tqdm(train_loader, desc=f"Job {JOB_ID} T{trial.number} E{epoch}", leave=False)
            for batch in train_pbar:
                optimizer.zero_grad()
                ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                loss = criterion(model(ids, attention_mask=mask).logits, lbls)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # evaluation
            model.eval()
            p, t = [], []
            val_loss = 0
            with torch.no_grad():
                for b in val_loader:
                    ids, mask, lbls = b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE), b['labels'].to(DEVICE)
                    out = model(ids, attention_mask=mask).logits
                    loss = criterion(out, lbls)
                    val_loss += loss.item()

                    p.extend(torch.argmax(out, dim=1).cpu().numpy())
                    t.extend(b['labels'].cpu().numpy())
            
            final_f1 = f1_score(t, p)

            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            precision, recall, f1, _ = precision_recall_fscore_support(t, p, average='binary', zero_division=0)
            acc = accuracy_score(t, p)
            avg_val_loss = val_loss / len(val_loader)

            print(f"\n[{JOB_ID}] Trial {trial.number} Epoch {epoch} Report:")
            print(f"  -> Loss: {avg_val_loss:.4f} | Acc: {acc:.4f}")
            print(f"  -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            
            # save checkpoint after each epoch
            atomic_save({
                'trial_number': trial.number,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'f1': final_f1
            }, CHECKPOINT_PATH)

            # save best model globally across all trials
            try:
                best_so_far = study.best_value
            except ValueError:
                best_so_far = -1.0

            if final_f1 > best_so_far:
                atomic_save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"[Job {JOB_ID}] *** New Global Best F1: {final_f1:.4f} (Saved to {MODEL_SAVE_PATH}) ***")

            trial.report(final_f1, epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.exceptions.TrialPruned()
            
            if epoch >= 3: # grace period
                early_stopper(final_f1)
                if early_stopper.early_stop:
                    print(f"[{JOB_ID}] Early stopping triggered at epoch {epoch}")
                    break

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return final_f1
    
    return objective

if __name__ == "__main__":
    # wait until space to run
    wait_for_free_vram(min_gb_required=6)

    df = load_and_merge_data()
    
    counts = df['orig_label'].value_counts().sort_index()
    weights = torch.tensor([1.0, 4.0], dtype=torch.float).to(DEVICE)#torch.tensor((len(df) / (len(counts) * counts)).values, dtype=torch.float).to(DEVICE)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['orig_label'], random_state=SEED)
    
    train_ds = PCLDataset(train_df['text'].tolist(), train_df['keyword'].tolist(), train_df['orig_label'].tolist(), is_train=True)
    val_ds = PCLDataset(val_df['text'].tolist(), val_df['keyword'].tolist(), val_df['orig_label'].tolist(), is_train=False)

    study = optuna.create_study(
        study_name="pcl_nli_deberta_binary",
        storage=f"sqlite:///{os.path.abspath(STUDY_DB_PATH)}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    existing_trials = [t for t in study.trials if t.state.name == "FAIL" or t.state.name == "RUNNING"]
    
    target_params = None
    if existing_trials:
        last_trial = sorted(existing_trials, key=lambda x: x.number)[-1]
        target_params = last_trial.params
        print(f"[{JOB_ID}] Found failed/interrupted Trial {last_trial.number}. Re-using params: {target_params}")

    # use these specific params for the next trial
    if target_params:
        study.enqueue_trial(target_params)

    obj_func = create_objective(train_ds, val_ds, weights, study)
    try:
        study.optimize(obj_func, n_trials=20)
    except KeyboardInterrupt:
        print("Got keyboard interrupt, saved progress")
        sys.exit(0)

    print(f"\nBest Params: {study.best_params}")
