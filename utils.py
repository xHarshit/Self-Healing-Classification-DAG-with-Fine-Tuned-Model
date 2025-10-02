# utils.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional

def load_dataset(csv_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load CSV (expects 'review' and 'sentiment' columns). Optionally sample for quick runs."""
    print(f"[INFO] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"[INFO] Sampled {sample_size} rows for demo/training.")
    print(f"[INFO] Dataset shape: {df.shape}")
    return df

def prepare_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 256, test_size: float = 0.1):
    """Tokenize and return HuggingFace Dataset split."""
    print("[INFO] Tokenizing dataset...")
    texts = df['review'].astype(str).tolist()
    labels = df['sentiment'].map({'negative': 0, 'positive': 1}).tolist()

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    tokenized["labels"] = labels

    ds = Dataset.from_dict(tokenized)
    ds = ds.train_test_split(test_size=test_size)
    print(f"[INFO] Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")
    return ds
