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

BASE_PATH = "./Training"
STUDY_DB_PATH = os.path.join(BASE_PATH, "pcl_optimization.db")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best_pcl_model.pt")
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
CHECKPOINT_PATH = os.path.join(BASE_PATH, f"checkpoint_job_{JOB_ID}.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256
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

# prevents partial writes
def atomic_save(state, path):
    tmp_path = path + f".tmp_{JOB_ID}"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

def create_objective(train_ds, val_ds, weights, study):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
        wd = trial.suggest_float("weight_decay", 0.01, 0.1)
        bs = trial.suggest_categorical("batch_size", [8, 16])

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5).float().to(DEVICE)
        
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        epochs = 3
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss(weight=weights)

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
            for batch in tqdm(train_loader, desc=f"Job {JOB_ID} T{trial.number} E{epoch}", leave=False):
                optimizer.zero_grad()
                ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                loss = criterion(model(ids, attention_mask=mask).logits, lbls)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # evaluation
            model.eval()
            p, t = [], []
            with torch.no_grad():
                for b in val_loader:
                    out = model(b['input_ids'].to(DEVICE), attention_mask=b['attention_mask'].to(DEVICE)).logits
                    p.extend([1 if x >= 2 else 0 for x in torch.argmax(out, dim=1).cpu().numpy()])
                    t.extend([1 if x >= 2 else 0 for x in b['labels'].cpu().numpy()])
            
            final_f1 = f1_score(t, p)
            
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
                print(f"\n[Job {JOB_ID}] *** New Global Best F1: {final_f1:.4f} (Saved to {MODEL_SAVE_PATH}) ***")

            trial.report(final_f1, epoch)
            if trial.should_prune():
                del model
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.exceptions.TrialPruned()

        del model
        torch.cuda.empty_cache()
        gc.collect()
        return final_f1
    
    return objective

if __name__ == "__main__":
    df = load_and_merge_data()
    
    counts = df['orig_label'].value_counts().sort_index()
    weights = torch.tensor((len(df) / (len(counts) * counts)).values, dtype=torch.float).to(DEVICE)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['orig_label'], random_state=SEED)
    
    train_ds = PCLDataset(train_df['text'].tolist(), train_df['keyword'].tolist(), train_df['orig_label'].tolist(), is_train=True)
    val_ds = PCLDataset(val_df['text'].tolist(), val_df['keyword'].tolist(), val_df['orig_label'].tolist(), is_train=False)

    study = optuna.create_study(
        study_name="pcl_nli_deberta",
        storage=f"sqlite:///{os.path.abspath(STUDY_DB_PATH)}",
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1)
    )

    obj_func = create_objective(train_ds, val_ds, weights, study)
    study.optimize(obj_func, n_trials=10)

    print(f"\nBest Params: {study.best_params}")