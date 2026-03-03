import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import precision_recall_fscore_support, classification_report
from tqdm.auto import tqdm
from pathlib import Path
from dotenv import load_dotenv
import argparse

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 192

parser = argparse.ArgumentParser(description="PCL Ensemble Submission")
parser.add_argument("--save_prefix", type=str, required=True, help="Prefix of models to load (e.g., 'roberta' or 'deberta')")
parser.add_argument("--num_folds", type=int, default=5, help="Number of ensemble folds (default: 5) (0 will mean no ensembling, just load a single model)")
parser.add_argument("--model_name", type=str, required=True, help="Base model to load (e.g., 'roberta-large')")
args = parser.parse_args()

TOKENIZER = AutoTokenizer.from_pretrained(args.model_name)
c0_id = TOKENIZER(" No", add_special_tokens=False)['input_ids'][0]
c1_id = TOKENIZER(" Yes", add_special_tokens=False)['input_ids'][0]

def load_data():
    print("Loading datasets...")
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv"), sep='\t', skiprows=4, names=['par_id', 'art_id', 'kw', 'country', 'text', 'orig_label'])
    raw_df['orig_label'] = pd.to_numeric(raw_df['orig_label'], errors='coerce')
    raw_df['binary_label'] = raw_df['orig_label'].apply(lambda x: 1 if x >= 2 else 0)
    raw_df['par_id'] = raw_df['par_id'].astype(str).str.strip()

    dev_labels = pd.read_csv(os.path.join(DATA_DIR, "dev_semeval_parids-labels.csv"))
    dev_labels = dev_labels.rename(columns={dev_labels.columns[0]: 'par_id'})
    dev_labels['par_id'] = dev_labels['par_id'].astype(str).str.strip()
    
    dev_df = dev_labels.merge(raw_df[['par_id', 'text', 'kw', 'binary_label']], on='par_id', how='left')
    dev_df['text'] = dev_df['text'].fillna("")
    dev_df['kw'] = dev_df['kw'].fillna("")
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, "task4_test.tsv"), sep='\t', names=['par_id', 'art_id', 'kw', 'country', 'text'])
    test_df['par_id'] = test_df['par_id'].astype(str).str.strip()
    test_df['kw'] = test_df['kw'].fillna("")
    
    return dev_df, test_df

def get_predictions(model, texts, keywords):
    model.eval()
    all_probs = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing", leave=False):
            batch_texts = texts[i:i+batch_size]
            batch_kws = keywords[i:i+batch_size]
            
            prompts = []
            for t, kw in zip(batch_texts, batch_kws):
                t = str(t).replace(TOKENIZER.mask_token, "")
                prompts.append(f"Question: Is the author being patronizing or condescending towards {kw}? Answer: {TOKENIZER.mask_token}")
                
            enc = TOKENIZER(batch_texts, prompts, add_special_tokens=True, max_length=MAX_LEN, padding=True, truncation='only_first', return_tensors='pt').to(DEVICE)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(**enc)
                seq_logits = outputs.logits
                mask_indices = (enc['input_ids'] == TOKENIZER.mask_token_id).long().argmax(dim=-1)
                
                verb_logits = seq_logits[torch.arange(seq_logits.size(0), device=DEVICE), mask_indices][:, [c0_id, c1_id]]
                probs = torch.softmax(verb_logits, dim=1)[:, 1] 
                all_probs.extend(probs.cpu().numpy())
                
    return np.array(all_probs)

def perform_error_analysis(df, probs, thresh, prefix):
    """Generates Error Analysis CSVs for Exercise 5.2"""
    print("\n--- Performing Local Error Analysis (Exercise 5.2) ---")
    df = df.copy()
    df['pred'] = (probs >= thresh).astype(int)
    df['prob'] = probs
    
    df['error_type'] = 'True Negative'
    df.loc[(df['binary_label'] == 1) & (df['pred'] == 1), 'error_type'] = 'True Positive'
    df.loc[(df['binary_label'] == 1) & (df['pred'] == 0), 'error_type'] = 'False Negative (Missed PCL)'
    df.loc[(df['binary_label'] == 0) & (df['pred'] == 1), 'error_type'] = 'False Positive (Hallucinated PCL)'
    
    print("\nConfusion Matrix Counts:")
    print(df['error_type'].value_counts())
    
    # Extract Top 20 Confident Errors
    fps = df[df['error_type'] == 'False Positive (Hallucinated PCL)'].sort_values(by='prob', ascending=False)
    fps[['text', 'kw', 'prob']].head(20).to_csv(os.path.join(BASE_PATH, f"{prefix}_error_FalsePositives.csv"), index=False)
    
    fns = df[df['error_type'] == 'False Negative (Missed PCL)'].sort_values(by='prob', ascending=True)
    fns[['text', 'kw', 'prob']].head(20).to_csv(os.path.join(BASE_PATH, f"{prefix}_error_FalseNegatives.csv"), index=False)
    
    # Keyword Performance Analysis
    print("\nCalculating F1 Score by Keyword...")
    kw_stats = []
    for kw in df['kw'].unique():
        kw_df = df[df['kw'] == kw]
        if len(kw_df) > 0:
            _, _, f1, support = precision_recall_fscore_support(kw_df['binary_label'], kw_df['pred'], average='binary', zero_division=0)
            kw_stats.append({'Keyword': kw, 'F1_Score': f1, 'Support': len(kw_df)})
            
    pd.DataFrame(kw_stats).sort_values(by='F1_Score', ascending=False).to_csv(os.path.join(BASE_PATH, f"{prefix}_error_KeywordPerformance.csv"), index=False)
    
    print(f"[SUCCESS] Error Analysis files saved.")

def main():
    dev_df, test_df = load_data()
    
    true_labels = dev_df['binary_label'].values

    print(f"Target Dev Lines: {len(dev_df)}")
    print(f"Target Test Lines: {len(test_df)}\n")

    # First do analysis on the model before training:
    print("\n--- Performing Initial Error Analysis with Untrained Model ---")
    untrained_model = AutoModelForMaskedLM.from_pretrained(args.model_name).float().to(DEVICE)
    untrained_probs = get_predictions(untrained_model, dev_df['text'].tolist(), dev_df['kw'].tolist())

    print("\n--- Finding Optimal Threshold on Dev Set (Untrained) ---")
    best_thresh_untrained = 0.5
    best_f1_untrained = 0.0

    for thresh in np.arange(0.1, 0.9, 0.02):
        preds_untrained = (untrained_probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(true_labels, preds_untrained, average='binary', zero_division=0)
        if f1 > best_f1_untrained:
            best_f1_untrained = f1
            best_thresh_untrained = thresh

    print(f"Optimal Untrained Threshold: {best_thresh_untrained:.2f} (Dev F1: {best_f1_untrained:.4f})")

    perform_error_analysis(dev_df, untrained_probs, best_thresh_untrained, f"{args.save_prefix}_untrained")
    del untrained_model
    torch.cuda.empty_cache()

    
    dev_ensemble_probs = np.zeros(len(dev_df))
    test_ensemble_probs = np.zeros(len(test_df))
    
    if args.num_folds == 0:
        print(f"--- Loading Single Model {args.save_prefix} ---")
        model = AutoModelForMaskedLM.from_pretrained(args.model_name).float().to(DEVICE)
        
        weight_path = os.path.join(BASE_PATH, f"best_pcl_model_{args.save_prefix}.pt")
            
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
        
        dev_probs = get_predictions(model, dev_df['text'].tolist(), dev_df['kw'].tolist())
        dev_ensemble_probs += dev_probs
        
        test_probs = get_predictions(model, test_df['text'].tolist(), test_df['kw'].tolist())
        test_ensemble_probs += test_probs
        
        del model
        torch.cuda.empty_cache()
    
    else:
        for fold in range(args.num_folds):
            print(f"--- Loading {args.save_prefix} Fold {fold} ---")
            model = AutoModelForMaskedLM.from_pretrained(args.model_name).float().to(DEVICE)
            
            weight_path = os.path.join(BASE_PATH, f"{args.save_prefix}_deberta_fold_{fold}.pt")
            if not os.path.exists(weight_path):
                weight_path = os.path.join(BASE_PATH, f"{args.save_prefix}_fold_{fold}.pt")
                
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
            
            dev_probs = get_predictions(model, dev_df['text'].tolist(), dev_df['kw'].tolist())
            dev_ensemble_probs += dev_probs
            
            test_probs = get_predictions(model, test_df['text'].tolist(), test_df['kw'].tolist())
            test_ensemble_probs += test_probs
            
            del model
            torch.cuda.empty_cache()
            
        # ensemble by averaging
        dev_ensemble_probs /= args.num_folds
        test_ensemble_probs /= args.num_folds
    
    print("\n--- Finding Optimal Threshold on Dev Set (Trained) ---")
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (dev_ensemble_probs >= thresh).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Optimal Threshold: {best_thresh:.2f} (Dev Ensemble F1: {best_f1:.4f})")
    
    perform_error_analysis(dev_df, dev_ensemble_probs, best_thresh, args.save_prefix)
    
    print("\n--- Generating Submission Files ---")
    final_dev_preds = (dev_ensemble_probs >= best_thresh).astype(int)
    dev_out_path = os.path.join(BASE_PATH, f"{args.save_prefix}_dev.txt")
    with open(dev_out_path, "w") as f:
        for pred in final_dev_preds:
            f.write(f"{pred}\n")
    print(f"[SUCCESS] Wrote {len(final_dev_preds)} lines to {dev_out_path}")
            
    final_test_preds = (test_ensemble_probs >= best_thresh).astype(int)
    test_out_path = os.path.join(BASE_PATH, f"{args.save_prefix}_test.txt")
    with open(test_out_path, "w") as f:
        for pred in final_test_preds:
            f.write(f"{pred}\n")
    print(f"[SUCCESS] Wrote {len(final_test_preds)} lines to {test_out_path}")

if __name__ == "__main__":
    main()
