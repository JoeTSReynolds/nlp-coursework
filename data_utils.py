import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')

def load_training_and_validation():
    train_path = os.path.join(DATA_DIR, 'train_ready.csv')
    val_path = os.path.join(DATA_DIR, 'val_ready.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Ready datasets not found! Please run the split-then-augment prep script first."
        )
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    return train_df, val_df

def load_official_dev_set():
    tsv_path = os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv")
    df_tsv = pd.read_csv(tsv_path, sep='\t', skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    
    df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()

    ids_path = os.path.join(DATA_DIR, "dev_semeval_parids-labels.csv")
    df_dev_ids = pd.read_csv(ids_path)
    df_dev_ids = df_dev_ids.rename(columns={df_dev_ids.columns[0]: 'par_id'})
    df_dev_ids['par_id'] = df_dev_ids['par_id'].astype(str).str.strip()

    # inner merge to isolate the dev set
    df_dev = pd.merge(df_dev_ids[['par_id']], df_tsv, on='par_id', how='inner').dropna(subset=['text', 'binary_label'])
    
    return df_dev