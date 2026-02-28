import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')

def load_and_merge_data(split_path='train_semeval_parids-labels.csv'):
    tsv_path = os.path.join(DATA_DIR, 'dontpatronizeme_pcl.tsv')
    csv_path = os.path.join(DATA_DIR, split_path)
    
    tsv_cols = ['par_id', 'art_id', 'keyword', 'country', 'text', 'orig_label']
    df_tsv = pd.read_csv(tsv_path, sep='\t', names=tsv_cols, skiprows=4)
    
    df_tsv['orig_label'] = pd.to_numeric(df_tsv['orig_label'], errors='coerce')
    
    # using the rule from the paper
    df_tsv['binary_label'] = df_tsv['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    
    df_tsv['par_id'] = df_tsv['par_id'].astype(str).str.strip()
    
    # training IDs
    df_train_ids = pd.read_csv(csv_path)
    first_col = df_train_ids.columns[0]
    df_train_ids = df_train_ids.rename(columns={first_col: 'par_id'})
    df_train_ids['par_id'] = df_train_ids['par_id'].astype(str).str.strip()
    
    # inner merge to filter
    df_train = pd.merge(df_train_ids[['par_id']], df_tsv, on='par_id', how='inner')
    df_train = df_train.dropna(subset=['text', 'binary_label'])
    
    print(f"Training dataset contains {len(df_train)} rows.")
    return df_train

def plot_keyword_imbalance(df):
    df['binary_label'] = df['binary_label'].astype(int)

    counts = df.groupby(['keyword', 'binary_label']).size().unstack(fill_value=0)
    
    counts['Total'] = counts.sum(axis=1)
    counts = counts.sort_values(by='Total', ascending=False).drop(columns=['Total'])
    
    sns.set_theme(style="whitegrid")
    ax = counts.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#3274a1', '#e1812c'])
    
    plt.title('Distribution of PCL vs. No PCL by Vulnerable Community Keyword', fontsize=14)
    plt.xlabel('Keyword', fontsize=12)
    plt.ylabel('Number of Paragraphs', fontsize=12)
    
    plt.legend(['No PCL (Class 0)', 'PCL (Class 1)'], title='Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('keyword_imbalance.png', dpi=300)
    plt.show()
    
    print("Saved chart as 'keyword_imbalance.png'")

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_merge_data()
    print("Plotting keyword imbalance...")
    plot_keyword_imbalance(df)
