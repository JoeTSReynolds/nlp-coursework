import pandas as pd
import spacy
from textattack.augmentation import CheckListAugmenter
from sklearn.model_selection import StratifiedKFold
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
N_FOLDS = 5

# Load spaCy for NER (Named Entity Recognition)
print("Loading spaCy NLP model...")
nlp = spacy.load("en_core_web_sm")

def prepare_kfold_data():
    tsv_path = os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv")
    df_tsv = pd.read_csv(tsv_path, sep='\t', skiprows=4, names=['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label'])
    
    df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()

    ids_path = os.path.join(DATA_DIR, "train_semeval_parids-labels.csv")
    df_train_ids = pd.read_csv(ids_path)
    df_train_ids = df_train_ids.rename(columns={df_train_ids.columns[0]: 'par_id'})
    df_train_ids['par_id'] = df_train_ids['par_id'].astype(str).str.strip()

    df_official = pd.merge(df_train_ids[['par_id']], df_tsv, on='par_id', how='inner').dropna(subset=['text', 'binary_label'])
    df_official = df_official.reset_index(drop=True)

    print(f"Total Official Training Rows: {len(df_official)}")

    # setup stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    augmenter = CheckListAugmenter(pct_words_to_swap=0.2, transformations_per_example=3)

    # generate the Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_official, df_official['binary_label'])):
        print(f"\n{'='*40}")
        print(f" GENERATING FOLD {fold + 1}/{N_FOLDS}")
        print(f"{'='*40}")
        
        train_df = df_official.iloc[train_idx].copy()
        val_df = df_official.iloc[val_idx].copy()
        
        pcl_train_df = train_df[train_df['binary_label'] == 1].copy()
        
        print(f"Original Train: {len(train_df)} | Val: {len(val_df)}")
        print(f"Augmenting {len(pcl_train_df)} PCL rows with spaCy GPE constraints...")
        
        augmented_rows = []
        for _, row in tqdm(pcl_train_df.iterrows(), total=len(pcl_train_df)):
            text, keyword = str(row['text']), str(row['keyword']).lower()
            
            # Use spaCy to find protected Geopolitical/Societal entities (useful for PCL context)
            doc = nlp(text)
            protected_entities = [
                ent.text.lower() for ent in doc.ents 
                if ent.label_ in ['GPE', 'LOC', 'NORP'] # Countries, Cities, Nationalities, Religious groups
            ]
            
            try:
                variations = augmenter.augment(text)
            except Exception:
                continue
                
            for i, v_text in enumerate(variations):
                v_text_lower = v_text.lower()
                
                # Check 1: Did CheckList actually change the text?
                # Check 2: Is the target keyword still there?
                if v_text != text and keyword in v_text_lower:
                    
                    # ensure all protected entities survived the augmentation
                    entities_survived = all(ent in v_text_lower for ent in protected_entities)
                    
                    if entities_survived:
                        new_row = row.copy()
                        new_row['text'] = v_text
                        new_row['par_id'] = f"{row['par_id']}_f{fold}_aug{i}" 
                        augmented_rows.append(new_row)

        # combine and save this specific fold
        aug_df = pd.DataFrame(augmented_rows).drop_duplicates(subset=['text'])
        final_train_df = pd.concat([train_df, aug_df], ignore_index=True).sample(frac=1.0, random_state=SEED)
        
        print(f"Fold {fold + 1} Synthetic Rows Added: {len(aug_df)}, Ratio: {len(final_train_df[final_train_df['binary_label'] == 0]) / (len(final_train_df[final_train_df['binary_label'] == 1]) + 1e-8):.2f}")
        print(f"Fold {fold + 1} Final Train Size: {len(final_train_df)}, Ratio: {len(final_train_df[final_train_df['binary_label'] == 0]) / (len(final_train_df[final_train_df['binary_label'] == 1]) + 1e-8):.2f}")
        
        # Save to disk
        final_train_df.to_csv(os.path.join(DATA_DIR, f"train_fold_{fold}.csv"), index=False)
        val_df.to_csv(os.path.join(DATA_DIR, f"val_fold_{fold}.csv"), index=False)

    print("\n[SUCCESS] 5-Fold dataset generation complete!")

if __name__ == "__main__":
    prepare_kfold_data()