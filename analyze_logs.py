# analyze_logs.py
import os
import csv
from collections import Counter
import statistics
import matplotlib.pyplot as plt

LOG_FILE = os.path.join("logs", "classification.log")


def read_logs():
    """Read the structured classification log into a list of dicts."""
    if not os.path.exists(LOG_FILE):
        print("[WARN] No log file found at", LOG_FILE)
        return []
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_stats(rows):
    """Compute confidence statistics and event breakdowns."""
    confidences = []
    events = Counter()
    for r in rows:
        events[r["Event"]] += 1
        try:
            confidences.append(float(r["Confidence"]))
        except:
            pass
    stats = {
        "total_events": len(rows),
        "events_breakdown": dict(events),
        "mean_confidence": statistics.mean(confidences) if confidences else None,
        "median_confidence": statistics.median(confidences) if confidences else None
    }
    return stats, confidences, events


def plot_confidence_histogram(confidences, out="logs/confidence_histogram.png"):
    """Plot histogram of model confidences."""
    if not confidences:
        print("[WARN] No confidences to plot.")
        return
    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=20, color="skyblue", edgecolor="black")
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[INFO] Saved {out}")


def plot_confidence_curve(confidences, out="logs/confidence_curve.png"):
    """Plot line curve of confidence across inputs."""
    if not confidences:
        print("[WARN] No confidences to plot.")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(confidences) + 1), confidences, marker="o", linestyle="-", color="green")
    plt.title("Confidence Curve Over Inputs")
    plt.xlabel("Input Number")
    plt.ylabel("Confidence")
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[INFO] Saved {out}")


def plot_fallback_stats(events, out="logs/fallback_stats.png"):
    """Plot bar chart showing fallback frequency statistics."""
    if not events:
        print("[WARN] No events to plot.")
        return

    # Aggregate fallback-related events
    fallback_count = events.get("ConfidenceCheck", 0) + events.get("FallbackUser", 0) + events.get("BackupModel", 0)
    normal_count = events.get("Inference", 0)

    plt.figure(figsize=(6, 4))
    plt.bar(["Normal Predictions", "Fallbacks Triggered"], [normal_count, fallback_count],
            color=["skyblue", "orange"], edgecolor="black")
    plt.title("Fallback Frequency Statistics")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"[INFO] Saved {out}")


if __name__ == "__main__":
    rows = read_logs()
    stats, confs, events = compute_stats(rows)

    print("\n[STATS]")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Generate plots
    plot_confidence_histogram(confs)
    plot_confidence_curve(confs)
    plot_fallback_stats(events)
