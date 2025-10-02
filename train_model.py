# train_model.py
import argparse
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from utils import load_dataset, prepare_dataset

def train_with_lora(csv_path: str, sample_size: int = None, output_dir: str = "./lora-finetuned-model",
                    num_train_epochs: int = 2, batch_size: int = 16):
    df = load_dataset(csv_path, sample_size)
    print("[INFO] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    dataset = prepare_dataset(df, tokenizer)

    print("[INFO] Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],
        task_type=TaskType.SEQ_CLS,
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "results"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )

    print("[INFO] Starting training...")
    trainer.train()

    print(f"[INFO] Saving model to {output_dir} ...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT with LoRA on IMDB dataset")
    parser.add_argument("--csv", type=str, default="IMDB Dataset.csv")
    parser.add_argument("--sample_size", type=int, default=None, help="Use smaller sample for fast demo")
    parser.add_argument("--output_dir", type=str, default="./lora-finetuned-model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_with_lora(args.csv, args.sample_size, args.output_dir, args.epochs, args.batch_size)
