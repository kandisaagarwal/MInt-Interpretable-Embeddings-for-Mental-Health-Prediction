# Usage:
#   Imbued / train.py checkpoint (default):
#     bash eval.bash [model_path] [output_dir] [quiz_seed]
#   GloVe checkpoint from src/train/glove_embeddings.py:
#     EVAL_GLOVE=1 bash eval.bash [model_path] [output_dir] [quiz_seed]
#     (defaults: model_path=src/model/glove_model.pt, metadata=glove_metadata.json)

# --------- Configurable Paths ---------
OUTPUT_DIR=${2:-output}
QUIZ_SEED=${3:-42}
mkdir -p "$OUTPUT_DIR"

# Stop on failure (e.g. generate_quiz must succeed before evaluate_quiz).
# pipefail: failure in evaluate.py is not hidden by `| tee`.
set -eo pipefail

# Clean up old outputs for a fresh run
rm -f "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*.log 2>/dev/null || true

VOCAB_PATH=data/processed/vocab.txt
DATASET_PATH=data/processed/trainingdata.csv
CONCEPTS_PATH=imparting-interpretability/concepts.txt

if [ "${EVAL_GLOVE:-0}" = "1" ]; then
	MODEL_PATH=${1:-src/model/glove_model.pt}
	METADATA_PATH=src/model/glove_metadata.json
	GLOVE_VEC_PATH=data/processed/glove_subset_for_vocab.txt
	GLOVE_FLAGS=(--glove-path "$GLOVE_VEC_PATH")
	# Imbued file unused when --glove-path is set; kept for argparse default
	VECTORS_PATH=data/processed/vectors_imbued.txt
else
	MODEL_PATH=${1:-src/model/model.pt}
	METADATA_PATH=src/model/metadata.json
	VECTORS_PATH=data/processed/vectors_imbued.txt
	GLOVE_FLAGS=()
fi
CLASSIFICATION_REPORT_PATH="$OUTPUT_DIR/classification_report.json"
BATCH_SIZE=32
QUIZ_QUESTIONS=100

# --------- 1. Run evaluation ---------
EVAL_LOG="$OUTPUT_DIR/evaluate.log"
EVAL_SCORE_FILE="$OUTPUT_DIR/evaluate_score.txt"
python3 ./src/eval/evaluate.py \
	--model "$MODEL_PATH" \
	--metadata "$METADATA_PATH" \
	--vocab "$VOCAB_PATH" \
	--vectors "$VECTORS_PATH" \
	"${GLOVE_FLAGS[@]}" \
	--data "$DATASET_PATH" \
	--batch-size "$BATCH_SIZE" \
	--output "$CLASSIFICATION_REPORT_PATH" | tee "$EVAL_LOG"

# --------- 2. Interpretability analysis ---------
python3 ./src/eval/topk_by_dimension.py \
	--output-file "$OUTPUT_DIR/topk_by_dimension.json" \
	--vocab "$VOCAB_PATH" \
	--vectors "$VECTORS_PATH" \
	"${GLOVE_FLAGS[@]}" \
	--concepts "$CONCEPTS_PATH" \
	--model "$MODEL_PATH" \
	--metadata "$METADATA_PATH" \
	--batch-size "$BATCH_SIZE" \
	--data "$DATASET_PATH"

# --------- 3. Generate quiz ---------
python3 ./src/eval/quiz/generate_quiz.py \
	--data "$DATASET_PATH" \
	--results "$OUTPUT_DIR/topk_by_dimension.json" \
	--seed "$QUIZ_SEED" \
	--random-outlier \
	--output-without-answers "$OUTPUT_DIR/quiz_without_answers.json" \
	--output-with-answers "$OUTPUT_DIR/quiz_with_answers.json"

# --------- 4. Evaluate quiz ---------
QUIZ_SCORE_FILE="$OUTPUT_DIR/quiz_score.txt"
python3 ./src/eval/quiz/evaluate_quiz.py \
	--quiz_path "$OUTPUT_DIR/quiz_with_answers.json" \
	--score_path "$QUIZ_SCORE_FILE" \
	--results_path "$OUTPUT_DIR/model_quiz_results.json" \
    --first_n "$QUIZ_QUESTIONS"

# --------- 5. Print summary ---------
echo -e "\n--- Summary ---"
echo "Classification report (accuracy and loss):"
if [ -f "$CLASSIFICATION_REPORT_PATH" ]; then
	jq -r '.classification_report' "$CLASSIFICATION_REPORT_PATH"
else
	echo "No classification_report.json found."
fi

echo "Interpretability (quiz) score:"
if [ -f "$QUIZ_SCORE_FILE" ]; then
	cat "$QUIZ_SCORE_FILE"
else
	echo "No quiz_score.txt found."
fi
