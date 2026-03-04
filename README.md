# Detecting Patronising and Condescending Language (PCL)

This repository contains my submission for the Natural Language Processing Coursework (2026), based on Task 4 (Subtask 1) from the SemEval 2022 competition.

**Name / Leaderboard Name:** Joe Reynolds

## Architecture

### Training Pipeline
```mermaid
graph LR
    A[Training Data] --> B[Data Augmentation]
    B --> C{5-Fold Stratified<br/>Cross-Validation Split}
    
    C --> D[RoBERTa-large Fold 0<br/>Optuna Tuning]
    C --> E[RoBERTa-large Fold 1]
    C --> F[RoBERTa-large Fold 2]
    C --> G[RoBERTa-large Fold 3]
    C --> H[RoBERTa-large Fold 4]

    D -. Optimal Params .-> E
    D -. Optimal Params .-> F
    D -. Optimal Params .-> G
    D -. Optimal Params .-> H

    classDef data fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff;
    classDef process fill:#f5a623,stroke:#333,stroke-width:2px,color:#fff;
    classDef split fill:#7ed321,stroke:#333,stroke-width:2px,color:#fff;
    classDef ensemble fill:#bd10e0,stroke:#333,stroke-width:2px,color:#fff;

    class A data;
    class B process;
    class C split;
    class D,E,F,G,H ensemble;
```

### Inference Pipeline
```mermaid
graph TB
    In[Input Text<br/>e.g., Test Set] --> D[Trained RoBERTa-large<br/>Fold 0]
    In --> E[Trained RoBERTa-large<br/>Fold 1]
    In --> F[Trained RoBERTa-large<br/>Fold 2]
    In --> G[Trained RoBERTa-large<br/>Fold 3]
    In --> H[Trained RoBERTa-large<br/>Fold 4]

    D --> I[Ensemble Aggregation<br/>Soft Voting]
    E --> I
    F --> I
    G --> I
    H --> I
    
    I --> J[Final Prediction<br/>PCL / No PCL]

    classDef data fill:#4a90e2,stroke:#333,stroke-width:2px,color:#fff;
    classDef process fill:#f5a623,stroke:#333,stroke-width:2px,color:#fff;
    classDef split fill:#7ed321,stroke:#333,stroke-width:2px,color:#fff;
    classDef ensemble fill:#bd10e0,stroke:#333,stroke-width:2px,color:#fff;

    class In,J data;
    class D,E,F,G,H,I ensemble;
```

## Repo Structure

```text
├── BestModel/                              # Contains the five model ensemble (check `generate_submission.py` for how inference works)
├── error_analysis/                         # Contains CSVs of error analysis from ablation study
├── logs/                                   # Contains logs from training and running `generate_submission.py`
├── plots/                                  # Contains plots for EDA and code used to generate them
├── scripts/                                # Contains python scripts covering augmentation, training, and evaluation
│   ├── cluster_utils.py                    # Useful functions for running on DOC cluster
│   ├── data_utils.py                       # Useful functions for data loading
│   ├── fold_dataset_gen.py                 # Generates the 5-fold stratified cross validation split augmented datasets from original unaugmented dataset
│   ├── generate_submission.py              # Runs evaluation and error analysis on the model and generates `dev.txt` and `test.txt`
│   └── train_kfold_optuna.py               # Trains the 5 folds of the model with optuna hyperparam search
├── .gitignore                              # Git ignore file
├── requirements.txt                        # Python dependencies
├── dev.txt                                 # Final model predictions on the official dev set (1 per line)
├── test.txt                                # Final model predictions on the official hidden test set (1 per line)
├── NLP_PCL_Coursework-Joe_Reynolds.pdf     # Critical review, EDA, and local evaluation write-up
└── README.md                               # This file
```

The BestModel ensemble achieved an F1 score of **0.6250** on the dev set.