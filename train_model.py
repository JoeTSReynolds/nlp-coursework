import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import optuna
from keyword_imbalance import load_and_merge_data

BASE_PATH = "./Training" 

STUDY_DB_PATH = os.path.join(BASE_PATH, "pcl_optimization.db")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best_pcl_model.pt")

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
        self.texts = texts
        self.keywords = keywords
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        keyword = str(self.keywords[idx])
        
        if self.is_train:
            hypothesis = random.choice(TEMPLATE_POOL).format(keyword=keyword)
        else:
            hypothesis = TEMPLATE_POOL[0].format(keyword=keyword)

        encoding = TOKENIZER(
            text, hypothesis,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_class_weights(df):
    counts = df['orig_label'].value_counts().sort_index()
    total = len(df)
    weights = total / (len(counts) * counts)
    weights_tensor = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)
    
    print("\n--- IN-PRACTICE LABEL DISTRIBUTION ---")
    for i, w in enumerate(weights.values):
        print(f"Label {i} (Count: {counts[i]}): Weight = {w:.4f}")
    return weights_tensor

def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 3e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    model = model.float().to(DEVICE) 
    
    # use the global datasets
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = 3 
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"T{trial.number} E{epoch}", leave=False):
            optimizer.zero_grad()
            ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            outputs = model(ids, attention_mask=mask)
            loss = criterion(outputs.logits, lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Intermediate Validation for Pruning
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, lbls = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                logits = model(ids, attention_mask=mask).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend([1 if p >= 2 else 0 for p in preds])
                val_labels.extend([1 if l >= 2 else 0 for l in lbls.cpu().numpy()])
        
        epoch_f1 = f1_score(val_labels, val_preds)
        
        # REPORT to Optuna for pruning
        trial.report(epoch_f1, epoch)
        
        # If this trial is performing poorly compared to others, KILL IT
        if trial.should_prune():
            print(f" -> Trial {trial.number} pruned at epoch {epoch}. F1: {epoch_f1:.4f}")
            raise optuna.exceptions.TrialPruned()

    final_score = epoch_f1 # Use the last calculated F1
    
    # Save best weights logic
    try:
        if final_score > study.best_value:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
    except:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return final_score

# --- 7. MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_and_merge_data()

    weights = get_class_weights(df)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['orig_label'], random_state=SEED)
    
    train_ds = PCLDataset(train_df['text'].tolist(), train_df['keyword'].tolist(), train_df['orig_label'].tolist(), is_train=True)
    val_ds = PCLDataset(val_df['text'].tolist(), val_df['keyword'].tolist(), val_df['orig_label'].tolist(), is_train=False)

    # persistent storage for Optuna study
    storage_url = f"sqlite:///{os.path.abspath(STUDY_DB_PATH)}"
    
    study = optuna.create_study(
        study_name="pcl_nli_deberta",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        direction="maximize"
    )

    print(f"\nTraining starting. Database: {STUDY_DB_PATH}")
    print(f"Artifacts will be saved in: {BASE_PATH}")
    
    study.optimize(objective, n_trials=20)

    print("\n--- OPTIMIZATION COMPLETE ---")
    print(f"Top F1 Score: {study.best_value:.4f}")
    print("Top Hyperparameters:", study.best_params)