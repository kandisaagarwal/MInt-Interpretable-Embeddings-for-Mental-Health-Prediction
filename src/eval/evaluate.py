

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mental_health_dataset import load_data, load_vocab_and_vectors
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
from src.train.train_with_glove import load_vocab_and_glove_vectors


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DEFAULT_VOCAB_PATH = os.path.join(BASE_DIR, "data/processed/vocab.txt")
DEFAULT_VECTORS_PATH = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "src/model/model-best-quiz.pt")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "output/classification_report.json")
DEFAULT_METADATA_PATH = os.path.join(BASE_DIR, "src/model/metadata.json")
DEFAULT_BATCH_SIZE = 32


def evaluate(model: nn.Module, loader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, _, _ = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy, all_labels, all_preds




def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model and output classification report.")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to model .pt file')
    parser.add_argument('--metadata', type=str, default=DEFAULT_METADATA_PATH, help='Path to metadata.json')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_PATH, help='Path to output classification report JSON')
    parser.add_argument('--vocab', type=str, default=DEFAULT_VOCAB_PATH, help='Path to vocab.txt')
    parser.add_argument('--vectors', type=str, default=DEFAULT_VECTORS_PATH, help='Path to vectors file (imbued layout; ignored if --glove-path is set)')
    parser.add_argument(
        '--glove-path',
        type=str,
        default=None,
        help='GloVe word-vector file (e.g. glove_subset_for_vocab.txt). Uses same alignment as glove_embeddings.py.',
    )
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH, help='Path to data CSV')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata for model hyperparameters and label mapping
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # Extract hyperparameters
    num_heads = metadata.get("num_heads", 4)
    num_layers = metadata.get("num_layers", 2)
    ff_dim = metadata.get("ff_dim", 256)
    mlp_hidden_dim = metadata.get("mlp_hidden_dim", 128)
    dropout = metadata.get("dropout", 0.1)
    max_len = metadata.get("max_len", 512)
    label2id = metadata.get("labels")
    if label2id is None:
        raise ValueError("labels mapping not found in metadata.json")
    # Ensure label2id is label->id, and build id->label
    id_to_label = {v: k for k, v in label2id.items()}
    ordered_labels = [id_to_label[i] for i in range(len(id_to_label))]

    # Load vocab and initial embedding table (checkpoint overwrites after build)
    # Auto-select GloVe loader when vectors path points to a GloVe file.
    glove_like_vectors = "glove" in os.path.basename(args.vectors).lower()
    if args.glove_path is not None:
        vocab, vectors, unk_id, pad_id = load_vocab_and_glove_vectors(args.vocab, args.glove_path)
    elif glove_like_vectors:
        vocab, vectors, unk_id, pad_id = load_vocab_and_glove_vectors(args.vocab, args.vectors)
    else:
        vocab, vectors, unk_id, pad_id = load_vocab_and_vectors(args.vocab, args.vectors)

    # Use the same deterministic split. We keep only the test set
    _, val_loader, test_loader = load_data(
        args.data,
        vocab,
        max_len,
        args.batch_size,
        pad_id,
        unk_id,
    )

    model = TransformerEncoderClassifier(
        embedding_matrix=vectors,
        num_classes=len(label2id),
        pad_idx=pad_id,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        max_len=max_len,
    ).to(device)

    # Load the saved model
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = nn.CrossEntropyLoss()
    avg_loss, accuracy, labels, preds = evaluate(model, test_loader, loss_fn, device)
    
    print(classification_report(labels, preds, target_names=ordered_labels, digits=4, zero_division=0))

    # Save classification report to file
    print("Saving classification report to file...")
    with open(args.output, "w") as f:
        json.dump({
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "classification_report": classification_report(labels, preds, target_names=ordered_labels, digits=4, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        }, f, indent=2)


if __name__ == "__main__":
    main()