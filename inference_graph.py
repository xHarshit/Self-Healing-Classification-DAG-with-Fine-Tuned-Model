# inference_graph.py
import argparse
import torch
import warnings
from typing import TypedDict, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from logger_utils import log_event
from backup_model import load_backup_model, backup_predict

# Suppress Hugging Face warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
hf_logging.set_verbosity_error()

# ====== State definition ======
class State(TypedDict):
    text: str
    prediction: str
    confidence: float

# ====== Load fine-tuned model ======
def load_finetuned_model(save_path: str = "./lora-finetuned-model"):
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    model.eval()
    return model, tokenizer

# ====== Inference Node ======
def inference_node(state: State, model, tokenizer) -> State:
    inputs = tokenizer(state["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence = float(torch.max(probs).cpu().item())
        label_idx = int(torch.argmax(probs, dim=-1).cpu().item())
        prediction = ["negative", "positive"][label_idx]

    print(f"[InferenceNode] Predicted label: {prediction.capitalize()} | Confidence: {confidence:.0%}")
    log_event("Inference", state["text"], prediction, confidence)
    return {"text": state["text"], "prediction": prediction, "confidence": confidence}

# ====== Confidence Check Node ======
def confidence_check_node(state: State, threshold: float = 0.7) -> State:
    if state["confidence"] < threshold:
        print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
        log_event("ConfidenceCheck", state["text"], state["prediction"], state["confidence"], "LOW")
    return state

def should_fallback(state: State, threshold: float = 0.7) -> str:
    return "fallback" if state["confidence"] < threshold else END

# ====== Fallback with Backup ======
def fallback_node_with_backup(state: State, backup_classifier, threshold: float = 0.7) -> State:
    backup_pred, backup_conf = backup_predict(backup_classifier, state["text"])
    print(f"[BackupModel] Prediction: {backup_pred.capitalize()} | Confidence: {backup_conf:.0%} (Backup model used)")

    if backup_conf >= threshold:
        print(f"\nFinal Label: {backup_pred.capitalize()} (Backup model used)\n")
        log_event("BackupModel", state["text"], backup_pred, backup_conf, "backup accepted")
        return {"text": state["text"], "prediction": backup_pred, "confidence": backup_conf}

    return fallback_node_user_clarify(state)

# ====== Fallback User Clarification ======
def fallback_node_user_clarify(state: State) -> State:
    print(f"[FallbackNode] Could you clarify your intent? Was this a {state['prediction']} review?")
    user_input = input("\nUser: ").strip()

    if "yes" in user_input.lower():
        final_prediction = state['prediction']
    elif "no" in user_input.lower():
        final_prediction = "positive" if state['prediction'] == "negative" else "negative"
    elif "negative" in user_input.lower():
        final_prediction = "negative"
    elif "positive" in user_input.lower():
        final_prediction = "positive"
    else:
        final_prediction = state['prediction']

    print(f"\nFinal Label: {final_prediction.capitalize()} (Corrected via user clarification)\n")
    log_event("FallbackUser", state["text"], final_prediction, state["confidence"], f"User input: {user_input}")
    log_event("FinalDecision", state["text"], final_prediction, state["confidence"], "user clarified")
    return {"text": state["text"], "prediction": final_prediction, "confidence": state["confidence"]}

# ====== Build Graph ======
def build_graph(model, tokenizer, use_backup: bool = False, backup_classifier: Optional[object] = None, threshold: float = 0.7):
    graph = StateGraph(State)

    graph.add_node("inference", RunnableLambda(lambda s: inference_node(s, model, tokenizer)))
    graph.add_node("confidence_check", RunnableLambda(lambda s: confidence_check_node(s, threshold)))

    if use_backup:
        graph.add_node("fallback", RunnableLambda(lambda s: fallback_node_with_backup(s, backup_classifier, threshold)))
    else:
        graph.add_node("fallback", RunnableLambda(fallback_node_user_clarify))

    graph.set_entry_point("inference")
    graph.add_edge("inference", "confidence_check")
    graph.add_conditional_edges(
        "confidence_check",
        lambda s: should_fallback(s, threshold),
        {"fallback": "fallback", END: END}
    )
    graph.add_edge("fallback", END)

    return graph.compile()

# ====== CLI Runner ======
def run_cli(model_path: str = "./lora-finetuned-model", threshold: float = 0.7, use_backup: bool = False):
    model, tokenizer = load_finetuned_model(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model.to("cuda")

    backup_classifier = load_backup_model() if use_backup else None
    app = build_graph(model, tokenizer, use_backup, backup_classifier, threshold)

    print("=== Self-Healing Classification CLI ===")
    print("Type 'exit' or 'quit' to close.\n")

    while True:
        raw = input("\nInput: ")
        if raw.strip().lower() in {"quit", "exit"}:
            print("\nExiting... Goodbye!")
            break

        print()
        state = {"text": raw, "prediction": "", "confidence": 0.0}
        _ = app.invoke(state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./lora-finetuned-model")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--use_backup", action="store_true", help="Enable backup zero-shot model")
    args = parser.parse_args()
    run_cli(args.model_path, args.threshold, args.use_backup)
