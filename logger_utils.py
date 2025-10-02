# logger_utils.py
import os
import csv
from datetime import datetime

LOGS_DIR = "logs"
LOG_FILE = os.path.join(LOGS_DIR, "classification.log")
os.makedirs(LOGS_DIR, exist_ok=True)

CSV_HEADER = ["Timestamp", "Event", "Text", "Prediction", "Confidence", "Note"]

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log_event(event_type: str, text: str, prediction: str, confidence: float, note: str = ""):
    """Append a timed row to CSV log."""
    write_header = not os.path.exists(LOG_FILE)
    row = [_now_str(), event_type, text, prediction, f"{confidence:.4f}", note]
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)
