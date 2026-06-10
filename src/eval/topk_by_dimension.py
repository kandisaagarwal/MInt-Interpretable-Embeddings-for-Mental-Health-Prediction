import argparse
import csv
import json
import os
import re
import sys

import numpy as np
import torch

# Adds the project root to the system path so we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mental_health_dataset import LABEL2ID, load_vocab_and_vectors
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
from src.train.train_with_glove import load_vocab_and_glove_vectors

# --- FILE PATH SETUP ---
# Locates where data, the trained model, and results should be stored

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VOCAB_PATH = os.path.join(BASE_DIR, "data/processed/vocab.txt")
VECTORS_PATH = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
MODEL_PATH = os.path.join(BASE_DIR, "src/model/model.pt")
METADATA_PATH = os.path.join(BASE_DIR, "src/model/metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
CONCEPTS_PATH = os.path.join(BASE_DIR, "imparting-interpretability/concepts.txt")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "output/topk_by_dimension.json")

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        return json.load(f)

# --- TEXT UTILITIES ---

def clean_text(text):
    """Removes punctuation and converts text to lowercase for consistency."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def tokenize(text, max_len):
    """Splits a sentence into a list of individual words."""
    return clean_text(text).split()[:max_len]

def encode_tokens(tokens, vocab, max_len, pad_id, unk_id):
    """
    Converts words into numbers (IDs) that the computer can understand.
    Also creates a 'mask' so the model knows which parts are real words vs padding.
    """
    token_ids = [vocab.get(tok, unk_id) for tok in tokens]
    attention_mask = [1] * len(token_ids)

    # If the sentence is too short, add 'padding' (empty slots) to reach max_len
    while len(token_ids) < max_len:
        token_ids.append(pad_id)
        attention_mask.append(0)

    return token_ids, attention_mask

# --- DATA LOADING ---

def read_examples(csv_path):
    """Loads the mental health statements and their known labels from a CSV file."""
    examples = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("statement", "")
            label_name = row.get("status", "")
            label_id = LABEL2ID.get(label_name)
            if label_id is None:
                raise ValueError(f"Unknown label '{label_name}' found in dataset.")
            examples.append({
                "text": text,
                "label_name": label_name,
                "label_id": int(label_id),
            })
    return examples

def read_concepts(path):
    """Reads the list of human-understandable concepts (like 'anxiety' or 'hope')."""
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
    """Sets up the command line options (e.g., how many examples to show: --k 10)."""
    parser = argparse.ArgumentParser(
        description="Extract top-k sequences that trigger specific model dimensions."
    )
    parser.add_argument("--vocab", default=VOCAB_PATH, type=str, help="Path to vocab.txt")
    parser.add_argument("--vectors", default=VECTORS_PATH, type=str, help="Path to vectors file (ignored if --glove-path is set)")
    parser.add_argument(
        "--glove-path",
        default=None,
        type=str,
        help="GloVe file (e.g. glove_subset_for_vocab.txt); same loader as glove_embeddings.py",
    )
    parser.add_argument("--model", default=MODEL_PATH, type=str, help="Path to trained model .pt file")
    parser.add_argument("--metadata", default=METADATA_PATH, type=str, help="Path to metadata.json file")
    parser.add_argument("--data", default=DATA_PATH, type=str, help="Path to data CSV file")
    parser.add_argument("--concepts", default=CONCEPTS_PATH, type=str, help="Path to concepts.txt file")
    parser.add_argument("--k", default=5, type=int, help="Number of top examples to show per concept")
    parser.add_argument("--batch-size", default=32, type=int, help="How many sentences to process at once")
    parser.add_argument("--max-len", default=512, type=int, help="Token length cap")
    parser.add_argument("--mode", default="post", choices=["pre", "post", "both"], help="Look at individual words (pre) or whole sentences (post)")
    parser.add_argument("--post-length-adjustment", default="none", choices=["none", "sqrt"], help="Adjust scores of based on sentence length")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument("--min-length", default=1, type=int, help="Minimum sequence length for top/bottom k")
    return parser.parse_args()

# --- MODEL PROCESSING ---

def pooled_from_encoded(encoded, attention_mask, use_max_pooling):
    """
    Combines the importance of individual words into a single score for the whole sentence.
    It either picks the strongest word (max) or takes the average (mean).
    """
    mask = attention_mask.unsqueeze(-1).type_as(encoded)
    if use_max_pooling:
        pooled = encoded.masked_fill(mask == 0, -1e9).max(dim=1).values
    else:
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled

def make_common_fields(idx, examples, pred_ids, id_to_label):
    """Helper to format the data for the final JSON output."""
    true_id = examples[idx]["label_id"]
    pred_id = int(pred_ids[idx])
    return {
        "dataset_index": int(idx),
        "text": examples[idx]["text"],
        "true_label_id": int(true_id),
        "true_label": examples[idx]["label_name"],
        "predicted_label_id": int(pred_id),
        "predicted_label": id_to_label[pred_id],
        "is_correct": bool(pred_id == true_id),
    }

def main():
    args = parse_args()
    

    # Setup hardware (use Graphics Card if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata for model hyperparameters
    metadata = load_metadata(args.metadata)
    num_heads = metadata.get("num_heads", 4)
    num_layers = metadata.get("num_layers", 2)
    ff_dim = metadata.get("ff_dim", 256)
    dropout = metadata.get("dropout", 0.1)
    mlp_hidden_dim = metadata.get("mlp_hidden_dim", 128)
    max_len = metadata.get("max_len", 512)
    freeze_embeddings = metadata.get("freeze_embeddings", False)
    max_pooling = metadata.get("max_pooling", True)
    num_classes = metadata.get("num_classes", len(LABEL2ID))

    # Load vocabulary and the actual model
    if args.glove_path:
        vocab, vectors, unk_id, pad_id = load_vocab_and_glove_vectors(args.vocab, args.glove_path)
    else:
        vocab, vectors, unk_id, pad_id = load_vocab_and_vectors(args.vocab, args.vectors)
    concepts = read_concepts(args.concepts)
    id_to_label = {idx: label for label, idx in LABEL2ID.items()}

    model = TransformerEncoderClassifier(
        embedding_matrix=vectors,
        num_classes=num_classes,
        pad_idx=pad_id,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        max_len=max_len,
        freeze_embeddings=freeze_embeddings,
        max_pooling=max_pooling,
    ).to(device)

    # Load the trained "brain" weights into the model
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval() # Put model in evaluation mode (no learning, just reflecting)

    examples = read_examples(args.data)

    all_pred_ids = []
    pre_rows = []
    post_rows = []

    # --- STEP 1: RUN THE DATA THROUGH THE MODEL ---
    with torch.no_grad(): # Don't calculate gradients (saves memory)
        for start in range(0, len(examples), args.batch_size):
            batch_examples = examples[start : start + args.batch_size]
            batch_tokens = [tokenize(ex["text"], args.max_len) for ex in batch_examples]

            batch_ids = []
            batch_masks = []
            for tokens in batch_tokens:
                ids, mask = encode_tokens(tokens, vocab, args.max_len, pad_id, unk_id)
                batch_ids.append(ids)
                batch_masks.append(mask)

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(batch_masks, dtype=torch.long, device=device)

            # Get the model's prediction (logits) and its internal activations (encoded)
            logits, _, encoded = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            all_pred_ids.append(preds.detach().cpu().numpy())

            # Store scores for individual words
            if args.mode in {"pre", "both"}:
                masked = encoded.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
                seq_dim_scores = masked.max(dim=1).values
                pre_rows.append(seq_dim_scores.detach().cpu().numpy())

            # Store scores for the whole sentence
            if args.mode in {"post", "both"}:
                pooled = pooled_from_encoded(encoded, attention_mask, model.max_pooling)
                post_rows.append(pooled.detach().cpu().numpy())

    pred_ids = np.concatenate(all_pred_ids, axis=0)

    # Prepare the header of our results file
    output = {
        "dataset_path": args.data,
        "model_path": args.model,
        "k": int(args.k),
        "mode": args.mode,
        "num_sequences": int(len(examples)),
        "num_dimensions": int(vectors.shape[1]),
    }


    # --- STEP 2: FIND THE TOP-K EXAMPLES PER CONCEPT ---
    # This loop goes through each concept and finds which text snippets 
    # triggered that concept most strongly (top_sequences) and least strongly (bottom_sequences).
    if args.mode in {"pre", "both"}:
        pre_scores = np.concatenate(pre_rows, axis=0)
        num_concepts = len(concepts)
        pre_results = []

        # Compute sequence lengths for each example
        sequence_lengths = np.array([max(1, len(tokenize(ex["text"], args.max_len))) for ex in examples], dtype=np.float32)

        for dim in range(num_concepts):
            # Adjust scores based on length
            if args.post_length_adjustment == "sqrt":
                adjusted_scores = pre_scores[:, dim] / np.sqrt(sequence_lengths)
            else:
                adjusted_scores = pre_scores[:, dim]

            ranked = np.argsort(adjusted_scores)
            top_indices = ranked[-args.k:][::-1]
            bottom_indices = ranked[:args.k]

            dim_result = {
                "dimension": int(dim),
                "concept_word": concepts[dim],
                "top_sequences": [],
                "bottom_sequences": [],
            }

            # For each top example, find the specific word (token) that caused the reaction
            for rank, idx in enumerate(top_indices, start=1):
                idx = int(idx)
                tokens = tokenize(examples[idx]["text"], args.max_len)
                token_ids, token_mask = encode_tokens(tokens, vocab, args.max_len, pad_id, unk_id)
                # Run the model on this single example to get per-token activations
                with torch.no_grad():
                    single_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                    single_mask = torch.tensor([token_mask], dtype=torch.long, device=device)
                    _, _, single_encoded = model(single_ids, single_mask)
                # Get the activations for this dimension (up to the real token length)
                token_values = single_encoded[0, :len(tokens), dim].detach().cpu().numpy() if tokens else np.array([])
                if token_values.size > 0:
                    top_pos = int(np.argmax(token_values))
                    top_token = tokens[top_pos]
                    top_token_value = float(token_values[top_pos])
                else:
                    top_pos = -1
                    top_token = ""
                    top_token_value = 0.0
                row = {
                    "rank": rank,
                    "score": float(pre_scores[idx, dim]),
                    "top_token_for_dimension": {
                        "position": top_pos,
                        "token": top_token,
                        "value": top_token_value,
                    },
                }
                row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
                dim_result["top_sequences"].append(row)

            # For each bottom example, do the same
            for rank, idx in enumerate(bottom_indices, start=1):
                idx = int(idx)
                tokens = tokenize(examples[idx]["text"], args.max_len)
                token_ids, token_mask = encode_tokens(tokens, vocab, args.max_len, pad_id, unk_id)
                with torch.no_grad():
                    single_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                    single_mask = torch.tensor([token_mask], dtype=torch.long, device=device)
                    _, _, single_encoded = model(single_ids, single_mask)
                token_values = single_encoded[0, :len(tokens), dim].detach().cpu().numpy() if tokens else np.array([])
                if token_values.size > 0:
                    top_pos = int(np.argmax(token_values))
                    top_token = tokens[top_pos]
                    top_token_value = float(token_values[top_pos])
                else:
                    top_pos = -1
                    top_token = ""
                    top_token_value = 0.0
                row = {
                    "rank": rank,
                    "score": float(pre_scores[idx, dim]),
                    "top_token_for_dimension": {
                        "position": top_pos,
                        "token": top_token,
                        "value": top_token_value,
                    },
                }
                row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
                dim_result["bottom_sequences"].append(row)

            pre_results.append(dim_result)
        output["prepool"] = {"results": pre_results}

    # --- POSTPOOL RESULTS (restore this section) ---
    if args.mode in {"post", "both"}:
        post_scores = np.concatenate(post_rows, axis=0)
        num_concepts = len(concepts)
        post_results = []

        # Compute sequence lengths for each example
        sequence_lengths = np.array([max(1, len(tokenize(ex["text"], args.max_len))) for ex in examples], dtype=np.float32)

        for dim in range(num_concepts):
            # Adjust scores based on length
            if args.post_length_adjustment == "sqrt":
                adjusted_scores = post_scores[:, dim] / np.sqrt(sequence_lengths)
            else:
                adjusted_scores = post_scores[:, dim]

            # Top k: use all indices
            ranked_all = np.argsort(adjusted_scores)
            top_indices = ranked_all[-args.k:][::-1]

            # Bottom k: filter by min_length
            min_length = args.min_length
            valid_indices = np.where(sequence_lengths >= min_length)[0]
            if valid_indices.size > 0:
                filtered_scores = adjusted_scores[valid_indices]
                ranked_bottom = valid_indices[np.argsort(filtered_scores)]
                bottom_indices = ranked_bottom[:args.k]
            else:
                # Fallback: if no sequences meet the minimum, use all
                bottom_indices = ranked_all[:args.k]

            dim_result = {
                "dimension": int(dim),
                "concept_word": concepts[dim],
                "top_sequences": [],
                "bottom_sequences": [],
            }

            # Top k sequences 
            for rank, idx in enumerate(top_indices, start=1):
                idx = int(idx)
                row = {
                    "rank": rank,
                    "score": float(adjusted_scores[idx]),
                    "raw_pooled_score": float(post_scores[idx, dim]),
                    "sequence_length": int(sequence_lengths[idx]),
                }
                row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
                dim_result["top_sequences"].append(row)

            # Bottom k sequences
            for rank, idx in enumerate(bottom_indices, start=1):
                idx = int(idx)
                row = {
                    "rank": rank,
                    "score": float(adjusted_scores[idx]),
                    "raw_pooled_score": float(post_scores[idx, dim]),
                    "sequence_length": int(sequence_lengths[idx]),
                }
                row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
                dim_result["bottom_sequences"].append(row)

            post_results.append(dim_result)
        output["postpool"] = {
            "length_adjustment": args.post_length_adjustment,
            "results": post_results
        }

    # --- STEP 3: SAVE EVERYTHING ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"Wrote JSON output to: {args.output_file}")

if __name__ == "__main__":
    main()