# backup_model.py
from transformers import pipeline
from typing import Tuple

def load_backup_model(model_name: str = "facebook/bart-large-mnli"):
    """Load zero-shot classifier (may download model if not cached)."""
    print(f"[INFO] Loading backup zero-shot model: {model_name} ...")
    return pipeline("zero-shot-classification", model=model_name)

def backup_predict(classifier, text: str, labels=("negative", "positive")) -> Tuple[str, float]:
    out = classifier(text, candidate_labels=list(labels), multi_label=False)
    return out["labels"][0], float(out["scores"][0])
