import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

BASE_PATH = os.getenv('BASE_PATH', "/vol/bitbucket/jtr23/nlp/")
DATA_DIR = os.path.join(BASE_PATH, 'data')

MODEL_NAME = "microsoft/deberta-v3-large"
TAPT_SAVE_PATH = os.path.join(BASE_PATH, "deberta-v3-tapt")

# Note: DeBERTa requires sentencepiece. Run: pip install sentencepiece

def run_tapt():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the RAW text (we don't care about labels or splits for TAPT)
    tsv_path = os.path.join(DATA_DIR, "dontpatronizeme_pcl.tsv")
    df = pd.read_csv(tsv_path, sep='\t', skiprows=4, names=['par_id', 'art_id', 'kw', 'country', 'text', 'label'])
    df = df.dropna(subset=['text'])
    
    # Create HF Dataset
    hf_dataset = Dataset.from_pandas(df[['text']])
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=192, padding="max_length")

    print("Tokenizing dataset for MLM...")
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # This automatically masks 15% of the words dynamically
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    # If your cluster has A100 GPUs, change fp16=True to bf16=True for better DeBERTa stability
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_PATH, "tapt_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=3,          # 3 epochs is standard for TAPT
        per_device_train_batch_size=8,
        save_steps=1000,
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=True,                   
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )
    
    print("Starting Task-Adaptive Pretraining (TAPT)...")
    trainer.train()
    
    print(f"Saving TAPT model to {TAPT_SAVE_PATH}...")
    trainer.save_model(TAPT_SAVE_PATH)
    tokenizer.save_pretrained(TAPT_SAVE_PATH)
    print("[SUCCESS] TAPT Complete!")

if __name__ == "__main__":
    run_tapt()