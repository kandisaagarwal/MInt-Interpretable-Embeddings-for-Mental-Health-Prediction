import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mental_health_dataset import load_vocab_and_vectors
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Fixed project paths (intentionally constants).
VOCAB_PATH = os.path.join(BASE_DIR, "data/processed/vocab.txt")
VECTORS_PATH = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
MODEL_PATH = os.path.join(BASE_DIR, "src/model/model.pt")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
CONCEPTS_PATH = os.path.join(BASE_DIR, "imparting-interpretability/concepts.txt")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "output/prepool_topk_by_dimension.json")

MAX_LEN = 128


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text, max_len):
    tokens = clean_text(text).split()
    return tokens[:max_len]


def encode_tokens(tokens, vocab, max_len, pad_id, unk_id):
    token_ids = [vocab.get(tok, unk_id) for tok in tokens]
    attention_mask = [1] * len(token_ids)

    while len(token_ids) < max_len:
        token_ids.append(pad_id)
        attention_mask.append(0)

    return token_ids, attention_mask


def read_statements(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("statement", "")
            rows.append(text)
    return rows


def read_concepts(path):
    concepts = []
    if not os.path.exists(path):
        return concepts
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            concepts.append(line)
    return concepts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export top-k input sequences per pre-pooled embedding dimension as JSON."
    )
    parser.add_argument("--k", default=10, type=int, help="Top k sequences per dimension")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size for model inference")
    parser.add_argument("--max-len", default=MAX_LEN, type=int, help="Token length cap")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_PATH, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.k <= 0:
        raise ValueError("k must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, vectors, unk_id, pad_id = load_vocab_and_vectors(VOCAB_PATH, VECTORS_PATH)
    concepts = read_concepts(CONCEPTS_PATH)

    model = TransformerEncoderClassifier(
        embedding_matrix=vectors,
        num_classes=7,
        pad_idx=pad_id,
    ).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    statements = read_statements(DATA_PATH)
    if not statements:
        raise ValueError("No statements were found in the dataset.")

    embedding_dim = vectors.shape[1]
    all_scores = []

    with torch.no_grad():
        for start in range(0, len(statements), args.batch_size):
            batch_texts = statements[start : start + args.batch_size]

            batch_tokens = [tokenize(text, args.max_len) for text in batch_texts]
            batch_ids = []
            batch_masks = []
            for tokens in batch_tokens:
                ids, mask = encode_tokens(tokens, vocab, args.max_len, pad_id, unk_id)
                batch_ids.append(ids)
                batch_masks.append(mask)

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_masks, dtype=torch.long, device=device)

            _, _, encoded = model(input_ids, attention_mask)

            # Sequence score per dimension = max token activation in the sequence.
            masked = encoded.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
            seq_dim_scores = masked.max(dim=1).values  # (batch, dim)

            # If a sequence has no valid tokens, make its score neutral.
            empty_rows = attention_mask.sum(dim=1) == 0
            if empty_rows.any():
                seq_dim_scores[empty_rows] = 0.0

            all_scores.append(seq_dim_scores.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)  # (num_sequences, embedding_dim)

    output = {
        "dataset_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "k": int(args.k),
        "num_sequences": int(scores.shape[0]),
        "num_dimensions": int(scores.shape[1]),
        "results": [],
    }

    for dim in range(embedding_dim):
        ranked = np.argsort(scores[:, dim])
        top_indices = ranked[-args.k:][::-1]

        dim_name = concepts[dim] if dim < len(concepts) else f"dim_{dim}"
        dim_result = {
            "dimension": int(dim),
            "concept_word": dim_name,
            "top_sequences": [],
        }

        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            text = statements[idx]
            tokens = tokenize(text, args.max_len)
            token_ids, token_mask = encode_tokens(tokens, vocab, args.max_len, pad_id, unk_id)

            with torch.no_grad():
                single_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                single_mask = torch.tensor([token_mask], dtype=torch.long, device=device)
                _, _, single_encoded = model(single_ids, single_mask)

            token_values = single_encoded[0, : len(tokens), dim].detach().cpu().numpy() if tokens else np.array([])
            if token_values.size > 0:
                top_pos = int(np.argmax(token_values))
                top_token = tokens[top_pos]
                top_token_value = float(token_values[top_pos])
            else:
                top_pos = -1
                top_token = ""
                top_token_value = 0.0

            dim_result["top_sequences"].append(
                {
                    "rank": int(rank),
                    "dataset_index": idx,
                    "score": float(scores[idx, dim]),
                    "text": text,
                    "top_token_for_dimension": {
                        "position": top_pos,
                        "token": top_token,
                        "value": top_token_value,
                    },
                }
            )

        output["results"].append(dim_result)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"Wrote JSON output to: {args.output_file}")


if __name__ == "__main__":
    main()