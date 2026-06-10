import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mental_health_dataset import LABEL2ID, load_data, load_vocab_and_vectors


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "src/model/mentalbert_final.pt")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "output/classification_report.json")
DEFAULT_VOCAB_PATH = os.path.join(BASE_DIR, "data/processed/vocab.txt")
DEFAULT_VECTORS_PATH = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LEN = 256
DEFAULT_MODEL_NAME = "mental/mental-bert-base-uncased"
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN")


def load_split_from_existing_loader(data_path, vocab_path, vectors_path, max_len, batch_size, split="val"):
    """
    Reuse the project's existing split logic so this evaluator reads the same
    validation rows as the training scripts.

    Note: we pass vocab/vectors only because load_data(...) requires them to
    construct its dataset objects. MentalBERT itself does not use these vectors.
    """
    vocab, _, unk_id, pad_id = load_vocab_and_vectors(vocab_path, vectors_path)
    train_loader, val_loader, test_loader = load_data(data_path, vocab, max_len, batch_size, pad_id, unk_id)
    if split == "val":
        loader = val_loader
    elif split == "test":
        loader = test_loader
    elif split == "train":
        loader = train_loader
    else:
        raise ValueError(f"Unknown split: {split}")
    df = loader.dataset.df
    texts = df["statement"].tolist()
    labels = df["status"].map(LABEL2ID).tolist()
    return texts, labels


def evaluate(model, tokenizer, texts, labels, device, batch_size, max_len):
    """Run batched validation and return loss, accuracy, labels, and predictions."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]

            # Tokenize raw statements with the same Hugging Face tokenizer used
            # during MentalBERT fine-tuning.
            enc = tokenizer(
                batch_texts,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            y = torch.tensor(batch_labels, dtype=torch.long, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, y)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_labels.extend(y.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(1, (len(texts) + batch_size - 1) // batch_size)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy, all_labels, all_preds


def main():
    # CLI defaults match this repository's standard training/eval locations.
    parser = argparse.ArgumentParser(description="Evaluate a MentalBERT model.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--vectors", type=str, default=DEFAULT_VECTORS_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--token", type=str, default=DEFAULT_HF_TOKEN)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate (train, val, test)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build base MentalBERT architecture, then load the fine-tuned checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.token,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        token=args.token,
        local_files_only=args.local_files_only,
    ).to(device)

    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)


    # Pull split text/labels from the shared project split pipeline.
    split_texts, split_labels = load_split_from_existing_loader(
        args.data,
        args.vocab,
        args.vectors,
        args.max_len,
        args.batch_size,
        split=args.split,
    )
    avg_loss, accuracy, labels, preds = evaluate(
        model,
        tokenizer,
        split_texts,
        split_labels,
        device,
        args.batch_size,
        args.max_len,
    )

    id_to_label = {v: k for k, v in LABEL2ID.items()}
    ordered_labels = [id_to_label[i] for i in range(len(id_to_label))]

    # Keep both console report (human-readable) and JSON output (script-friendly).
    report = classification_report(labels, preds, target_names=ordered_labels, digits=4, zero_division=0)
    print(report)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "avg_loss": avg_loss,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
