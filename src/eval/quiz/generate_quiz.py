"""
generate_quiz.py

Randomly selects 20 dimensions and generates quiz questions.
For each dimension:
  - Takes the top 3 sequences as "real" examples
  - Picks 1 random outlier from a different dimension
  - Shuffles the 4 options
  - Records the correct answer

Outputs:
  output/quiz_without_answers.json
  output/quiz_with_answers.json
"""

import csv
import json
import random
import argparse
import os


# Config
DATASET_PATH = "data/processed/trainingdata.csv"
RESULTS_PATH = "output/topk_by_dimension.json"
OUTPUT_WITHOUT_ANSWERS = "output/quiz_without_answers.json"
OUTPUT_WITH_ANSWERS = "output/quiz_with_answers.json"
REAL_SEQS = 3
RANDOM_SEED = 42

def parse_args():
    parser = argparse.ArgumentParser(description="Generate quiz questions from topk_by_dimension output.")
    parser.add_argument('--data', type=str, default=DATASET_PATH, help='Path to the CSV dataset for outlier candidates.')
    parser.add_argument('--results', type=str, default=RESULTS_PATH, help='Path to the topk_by_dimension JSON results.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for quiz generation.')
    parser.add_argument('--random-outlier', action='store_true', help='Use a random sequence from another dimension as the outlier instead of the bottom one.')
    parser.add_argument('--output-without-answers', type=str, default=OUTPUT_WITHOUT_ANSWERS, help='Path to output quiz JSON without answers.')
    parser.add_argument('--output-with-answers', type=str, default=OUTPUT_WITH_ANSWERS, help='Path to output quiz JSON with answers.')
    return parser.parse_args()

args = parse_args()
random.seed(args.seed)

# Load all possible outlier candidates from the CSV
all_statements = []
with open(args.data, "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        all_statements.append(row["statement"])


# Load results
with open(args.results, "r") as f:
    data = json.load(f)


# Use postpool results
postpool_results = data["postpool"]["results"]

# Build lookup: dim -> list of sequence texts
dim_to_sequences = {}
dim_to_concept   = {}
for result in postpool_results:
    dim = result["dimension"]
    dim_to_sequences[dim] = [s["text"] for s in result["top_sequences"]]
    dim_to_concept[dim]   = result["concept_word"]

all_dims = list(dim_to_sequences.keys())

# Use all dimensions
selected_dims = sorted(all_dims)

# Generate questions
questions_without_answers = []
questions_with_answers = []


for dim in selected_dims:
    concept = dim_to_concept[dim]
    dim_result = postpool_results[selected_dims.index(dim)]
    top_seqs = dim_result.get("top_sequences", [])
    bottom_seqs = dim_result.get("bottom_sequences", [])
    real_seqs = [s["text"] for s in top_seqs[:REAL_SEQS]]

    if args.random_outlier:
        # Pick a random sequence from the entire CSV as the outlier
        outlier = random.choice(all_statements)
    else:
        # Use the bottom sequence as the outlier (default)
        outlier = bottom_seqs[0]["text"] if bottom_seqs else (top_seqs[-1]["text"] if top_seqs else "")

    # Shuffle options and track outlier position
    options = real_seqs + [outlier]
    combined = list(enumerate(options))
    random.shuffle(combined)
    orig_indices, shuffled_options = zip(*combined)
    correct_answer = list(orig_indices).index(REAL_SEQS)  # outlier is always last in options before shuffle

    q_id = len(questions_without_answers) + 1

    # Version without answer
    questions_without_answers.append({
        "question_id": q_id,
        "dimension":   dim,
        "concept":     concept,
        "options":     list(shuffled_options),
    })

    # Version with answer
    questions_with_answers.append({
        "question_id":   q_id,
        "dimension":     dim,
        "concept":       concept,
        "options":       list(shuffled_options),
        "correct_answer": correct_answer,        # 0-indexed
    })

# Save both files
output_without_dir = os.path.dirname(args.output_without_answers)
if output_without_dir:
    os.makedirs(output_without_dir, exist_ok=True)

output_with_dir = os.path.dirname(args.output_with_answers)
if output_with_dir:
    os.makedirs(output_with_dir, exist_ok=True)

with open(args.output_without_answers, "w") as f:
    json.dump(questions_without_answers, f, indent=2)

with open(args.output_with_answers, "w") as f:
    json.dump(questions_with_answers, f, indent=2)

print(f"\nSaved {len(questions_without_answers)} questions to:")
print(f"  {args.output_without_answers}")
print(f"  {args.output_with_answers}")