# data_prep.py
import pandas as pd
from textattack.augmentation import CheckListAugmenter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import os
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')
SEED = 42

def prepare_and_augment_data():
    # load original data
    tsv_path = os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv")
    df_tsv = pd.read_csv(tsv_path, sep='\t', skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    
    df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()

    # load official train IDs and merge (to isolate official training data)
    ids_path = os.path.join(DATA_DIR, "train_semeval_parids-labels.csv")
    df_train_ids = pd.read_csv(ids_path)
    df_train_ids = df_train_ids.rename(columns={df_train_ids.columns[0]: 'par_id'})
    df_train_ids['par_id'] = df_train_ids['par_id'].astype(str).str.strip()

    # inner merge to get the clean starting dataset
    df_official = pd.merge(df_train_ids[['par_id']], df_tsv, on='par_id', how='inner').dropna(subset=['text', 'binary_label'])

    # split first to prevent data leakage
    train_df, val_df = train_test_split(df_official, test_size=0.2, stratify=df_official['binary_label'], random_state=SEED)
    
    print(f"Original Train shape: {train_df.shape}")
    print(f"Original Val shape: {val_df.shape}")

    # augment ONLY the training data
    pcl_train_df = train_df[train_df['binary_label'] == 1].copy()
    augmenter = CheckListAugmenter(pct_words_to_swap=0.2, transformations_per_example=3)
    
    print(f"\nAugmenting {len(pcl_train_df)} positive PCL rows in the training set...")
    
    augmented_rows = []
    for _, row in tqdm(pcl_train_df.iterrows(), total=len(pcl_train_df)):
        text, keyword = str(row['text']), str(row['keyword']).lower()
        
        try:
            variations = augmenter.augment(text)
        except Exception:
            continue
            
        for i, v_text in enumerate(variations):
            if keyword in v_text.lower() and v_text != text:
                new_row = row.copy()
                new_row['text'] = v_text
                # create a unique ID (helpful for debugging to know its synthetic)
                new_row['par_id'] = f"{row['par_id']}_aug{i}" 
                augmented_rows.append(new_row)

    aug_df = pd.DataFrame(augmented_rows).drop_duplicates(subset=['text'])
    
    # 5. Combine and Save
    final_train_df = pd.concat([train_df, aug_df], ignore_index=True).sample(frac=1.0, random_state=SEED)
    
    train_save_path = os.path.join(DATA_DIR, "train_ready.csv")
    val_save_path = os.path.join(DATA_DIR, "val_ready.csv")
    
    final_train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)
    
    print(f"\n--- DATASET VERIFICATION ---")
    print(f"Total Synthetic Rows Added: {len(aug_df)}")
    print(f"Final Training Set Size: {len(final_train_df)}")
    print(f"Final Validation Set Size: {len(val_df)}")
    
    num_class_0 = len(final_train_df[final_train_df['binary_label'] == 0])
    num_class_1 = len(final_train_df[final_train_df['binary_label'] == 1])
    print(f"New Train Class Distribution: Class 0 = {num_class_0}, Class 1 = {num_class_1} (Ratio 1:{num_class_0/num_class_1:.2f})")

if __name__ == "__main__":
    prepare_and_augment_data()