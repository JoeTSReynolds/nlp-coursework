import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')

SEED = 42

def load_training_and_validation(augmented=True):

    if augmented:
        train_path = os.path.join(DATA_DIR, 'train_ready.csv')
        val_path = os.path.join(DATA_DIR, 'val_ready.csv')
    
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(
                "Ready datasets not found! Please run the split-then-augment prep script first."
            )
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    else:
        df_tsv = pd.read_csv(os.path.join(DATA_DIR, 'dontpatronizeme_pcl.tsv'), sep='\t', skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
        df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    
        # using the rule from the paper
        df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    
        df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()
    
        # training IDs
        df_train_ids = pd.read_csv(os.path.join(DATA_DIR, 'train_semeval_parids-labels.csv'))
        first_col = df_train_ids.columns[0]
        df_train_ids = df_train_ids.rename(columns={first_col: 'par_id'})
        df_train_ids['par_id'] = df_train_ids['par_id'].astype(str).str.strip()
    
        # inner merge to filter
        df_train = pd.merge(df_train_ids[['par_id']], df_tsv, on='par_id', how='inner')
        
        train_df, val_df = train_test_split(df_train, test_size=0.2, stratify=df_train['orig_label'], random_state=SEED)
    
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
