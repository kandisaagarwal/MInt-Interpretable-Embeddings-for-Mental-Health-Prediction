import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.mental_health_dataset import LABEL2ID


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "src/model/mentalbert_final.pt")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "output/topk_by_dimension.json")
DEFAULT_MODEL_NAME = "mental/mental-bert-base-uncased"
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN")


def parse_args():
    # CLI flags let this script run against different checkpoints/splits
    # without editing code.
    parser = argparse.ArgumentParser(
        description="Extract top-k sequences by hidden-state dimensions for MentalBERT."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--token", type=str, default=DEFAULT_HF_TOKEN)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--post-length-adjustment", default="none", choices=["none", "sqrt"])
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--num-dimensions", type=int, default=None, help="Limit to first N dimensions (None = all)")
    return parser.parse_args()


def read_examples(csv_path):
    # Read the dataset as raw text + labels. We keep text intact here because
    # tokenization is performed by the Hugging Face tokenizer later.
    examples = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("statement", "")
            label_name = row.get("status", "")
            label_id = LABEL2ID.get(label_name)
            if label_id is None:
                continue
            examples.append(
                {
                    "text": text,
                    "label_name": label_name,
                    "label_id": int(label_id),
                }
            )
    return examples


def make_common_fields(idx, examples, pred_ids, id_to_label):
    # Shared metadata included for both top and bottom examples per dimension.
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


# Pooling is now done inline in the main loop; no separate function needed.



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer + classification model architecture, then inject the
    # fine-tuned checkpoint weights.
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
    model.eval()

    examples = read_examples(args.data)
    id_to_label = {v: k for k, v in LABEL2ID.items()}

    # We collect predictions and pooled vectors in separate arrays, then merge
    # once at the end for ranking.
    pred_rows = []
    pooled_rows = []
    lengths = []

    with torch.no_grad():
        for start in tqdm(range(0, len(examples), args.batch_size), desc="Scoring sequences"):
            batch_examples = examples[start : start + args.batch_size]
            batch_texts = [ex["text"] for ex in batch_examples]

            enc = tokenizer(
                batch_texts,
                max_length=args.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            # Single forward pass: extract base model outputs (pooler_output).
            base_outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            pooler_output = base_outputs.pooler_output
            
            # Compute logits from pooler_output using the classifier head.
            logits = model.classifier(pooler_output)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            pred_rows.append(preds)

            # Reuse pooler_output for dimension-based ranking.
            pooled = pooler_output.detach().cpu().numpy()
            pooled_rows.append(pooled)

            lengths.extend(attention_mask.sum(dim=1).detach().cpu().numpy().tolist())

    pred_ids = np.concatenate(pred_rows, axis=0)
    post_scores = np.concatenate(pooled_rows, axis=0)
    sequence_lengths = np.array(lengths, dtype=np.float32)

    num_dimensions = int(post_scores.shape[1])
    # Limit to --num-dimensions if specified.
    dimensions_to_process = args.num_dimensions if args.num_dimensions is not None else num_dimensions
    dimensions_to_process = min(dimensions_to_process, num_dimensions)
    
    results = []
    for dim in range(dimensions_to_process):
        # Optional length normalization reduces bias toward longer texts.
        if args.post_length_adjustment == "sqrt":
            adjusted_scores = post_scores[:, dim] / np.sqrt(np.maximum(sequence_lengths, 1.0))
        else:
            adjusted_scores = post_scores[:, dim]

        # Top-k = largest scores on this dimension.
        ranked_all = np.argsort(adjusted_scores)
        top_indices = ranked_all[-args.k :][::-1]

        # Bottom-k can ignore very short sequences (often noisy).
        valid_indices = np.where(sequence_lengths >= args.min_length)[0]
        if valid_indices.size > 0:
            filtered_scores = adjusted_scores[valid_indices]
            ranked_bottom = valid_indices[np.argsort(filtered_scores)]
            bottom_indices = ranked_bottom[: args.k]
        else:
            bottom_indices = ranked_all[: args.k]

        dim_result = {
            "dimension": int(dim),
            "concept_word": f"dim_{dim}",
            "top_sequences": [],
            "bottom_sequences": [],
        }

        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            row = {
                "rank": rank,
                "score": float(post_scores[idx, dim]),
                "adjusted_score": float(adjusted_scores[idx]),
            }
            row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
            dim_result["top_sequences"].append(row)

        for rank, idx in enumerate(bottom_indices, start=1):
            idx = int(idx)
            row = {
                "rank": rank,
                "score": float(post_scores[idx, dim]),
                "adjusted_score": float(adjusted_scores[idx]),
            }
            row.update(make_common_fields(idx, examples, pred_ids, id_to_label))
            dim_result["bottom_sequences"].append(row)

        results.append(dim_result)

    output = {
        # "postpool" mirrors the naming used by the rest of this repo's
        # interpretability pipeline.
        "dataset_path": args.data,
        "model_path": args.model,
        "k": int(args.k),
        "mode": "post",
        "num_sequences": int(len(examples)),
        "num_dimensions": num_dimensions,
        "dimensions_processed": dimensions_to_process,
        "postpool": {"results": results},
    }

    # Ensure output directory exists before writing.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote JSON output to: {args.output_file}")


if __name__ == "__main__":
    main()
