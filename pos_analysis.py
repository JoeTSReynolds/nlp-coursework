import os
from annotated_types import doc
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

tqdm.pandas()

DATA_DIR = 'data' 

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

def load_and_merge_data():
    tsv_path = os.path.join(DATA_DIR, 'dontpatronizeme_pcl.tsv')
    csv_path = os.path.join(DATA_DIR, 'train_semeval_parids-labels.csv')
    
    tsv_cols = ['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label']
    df_tsv = pd.read_csv(tsv_path, sep='\t', names=tsv_cols, skiprows=4)
    
    df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()
    
    df_train_ids = pd.read_csv(csv_path)
    first_col = df_train_ids.columns[0]
    df_train_ids = df_train_ids.rename(columns={first_col: 'par_id'})
    df_train_ids['par_id'] = df_train_ids['par_id'].astype(str).str.strip()
    
    df_train = pd.merge(df_train_ids[['par_id']], df_tsv, on='par_id', how='inner')
    df_train = df_train.dropna(subset=['text', 'binary_label'])
    
    print(f"Training dataset contains {len(df_train)} rows.")
    return df_train

def extract_pos_features(text):
    doc = nlp(str(text))
    
    num_words = len(doc)
    if num_words == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    num_pronouns = sum(1 for token in doc if token.pos_ == 'PRON')
    num_adjectives = sum(1 for token in doc if token.pos_ == 'ADJ')
    num_modals = sum(1 for token in doc if token.tag_ == 'MD')
    num_adverbs = sum(1 for token in doc if token.pos_ == 'ADV')
    num_determiners = sum(1 for token in doc if token.pos_ == 'DET')
    
    pron_per_100 = (num_pronouns / num_words) * 100
    adj_per_100 = (num_adjectives / num_words) * 100
    modal_per_100 = (num_modals / num_words) * 100
    adv_per_100 = (num_adverbs / num_words) * 100
    det_per_100 = (num_determiners / num_words) * 100
    
    return pron_per_100, adj_per_100, modal_per_100, adv_per_100, det_per_100

def run_pos_analysis(df):
    print("Running NLP POS tagging across all paragraphs (this will take 1-2 minutes)...")
    
    df[['pron_per_100', 'adj_per_100', 'modal_per_100', 'adv_per_100', 'det_per_100']] = df['text'].progress_apply(
        lambda x: pd.Series(extract_pos_features(x))
    )

    # calc averages
    pos_means = df.groupby('binary_label')[['pron_per_100', 'adj_per_100', 'modal_per_100', 'adv_per_100', 'det_per_100']].mean().reset_index()
    
    # Rename columns for the chart and table
    pos_means.rename(columns={
        'binary_label': 'Label',
        'pron_per_100': 'Pronouns',
        'adj_per_100': 'Adjectives',
        'modal_per_100': 'Modal Verbs',
        'adv_per_100': 'Adverbs',
        'det_per_100': 'Determiners'
    }, inplace=True)
    
    pos_means['Label'] = pos_means['Label'].map({0: 'No PCL', 1: 'PCL'})
    
    # for console output
    print("\n--- TABULAR EVIDENCE ---")
    print("Average POS count per 100 words:")
    print(pos_means.to_string(index=False))
    print("------------------------\n")
    
    melted_df = pd.melt(pos_means, id_vars='Label', var_name='POS Tag', value_name='Average per 100 words')
    
    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='POS Tag', y='Average per 100 words', hue='Label', data=melted_df, palette=['#1f77b4', '#ff7f0e'])
    
    plt.title('Part-of-Speech Frequencies: PCL vs No PCL', fontsize=14)
    plt.ylabel('Average count per 100 words', fontsize=12)
    plt.xlabel('Part of Speech', fontsize=12)
    plt.tight_layout()
    
    plt.savefig('pos_analysis.png', dpi=300)
    print("Saved chart as 'pos_analysis.png'")

if __name__ == "__main__":
    df = load_and_merge_data()
    run_pos_analysis(df)