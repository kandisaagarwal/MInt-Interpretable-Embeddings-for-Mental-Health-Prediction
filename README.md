# MInt: Mental Health Interpretable Tranformer Multi-Class Classification

## Abstract
We address the black box nature of word embeddings in mental health text classification by learning representations whose dimensions correspond to human interpretable linguistic concepts used to predict one of seven labels. We re-implement and adapt Şenel et al. (2020), modifying a GloVe-style objective so embedding dimensions align with predefined mental health and affect concepts. The embeddings are put into a shallow transformer model and then through a classification head to output the classification label.

We define embedding interpretability as dimension-concept alignment, measured by overlap between each dimension's top-ranked words and curated concept word groups, and we test faithfulness by identifying which concepts drive predictions and how outputs change under dimension ablation. We evaluate Accuracy and Macro-F1 and compare against standard GloVe (Pennington et al., 2014), MentalBERT (Ji et al., 2022) and an alternative interpretable embedding baseline (NNSE; Murphy et al., 2012) to quantify tradeoffs between accuracy, semantic quality, and transparency.

## Project Structure

### Training Files

#### `src/train/train.py`
Main training script that implements the `Trainer` class. This file handles:
- Loading vocabulary and pretrained embeddings
- Loading training data from CSV
- Setting up the model with hyperparameters (batch size, learning rate, epochs, etc.)
- Training loop with loss computation (including optional interpretability loss for regularization)
- Validation during training
- Early stopping and learning rate scheduling
- Model checkpointing

**How to run:**
```bash
python src/train/train.py
```

### Model Files

#### `src/model/transformer_encoder_classifier.py`
Defines the `TransformerEncoderClassifier` class - a neural network architecture consisting of:
- Token embedding layer initialized from pretrained interpretable embeddings
- Positional embeddings to encode sequence positions
- Stacked transformer encoder layers (default 2 layers, 4 attention heads)
- Feed-forward networks within each transformer layer
- Mean or max pooling over the sequence
- MLP classification head that outputs logits for 7 mental health condition classes

**Key features:**
- Supports freezing embeddings to preserve interpretability
- Configurable number of transformer layers, attention heads, and hidden dimensions
- Max sequence length of 512 tokens (handles 99th percentile of sequences)

#### `src/model/mental_health_dataset.py`
Utility module containing:
- `load_vocab_and_vectors()` - Loads vocabulary and embedding vectors from text files
- `load_data()` - Loads mental health dataset from CSV and creates PyTorch DataLoaders
- `LABEL2ID` - Mapping of 7 mental health condition labels to class IDs


### Evaluation Files

#### Overview
The evaluation scripts compute various metrics and analyses:

MInt:
- **evaluate.py** - Computes classification accuracy, loss, per-class precision/recall/F1, and confusion matrix
- **evaluate_quiz.py** - Tests human alignment on interpretability quiz
- **generate_quiz.py** - Creates interpretability quiz (20 random dimensions, 4 options each)
- **topk_by_dimension.py** - Identifies top sequences that activate each embedding dimension
- **prepool_topk.py** - Extracts top sequences before pooling layer

MentalBERT:
- **evaluate_mentalbert.py** - Evaluates MentalBERT baseline for comparison
- **topk_by_dimension_mentalbert.py** - Identifies top sequences that activate each embedding dimension MentalBERT

Standard GloVe Embeddings:
- **glove_embeddings.py** - Loads standard GloVe embeddings (comparison baseline)

Computing Most important dimensions for each concept:
- **gradient.py** - Computes gradient-based attribution for dimension importance

**How to run all evaluations and generate quizzes:**
```bash
bash src/eval/eval.bash
```

This runs all evaluation scripts in sequence and generates:
- `output/classification_report.json` - Performance metrics
- `output/topk_by_dimension.json` - Top sequences per dimension
- `output/quiz_without_answers.json` - Quiz for human evaluation
- `output/quiz_with_answers.json` - Quiz with correct answers

### Interpretability Files

#### `imparting-interpretability/`
Companion C/C++ project for creating interpretable embeddings using modified GloVe algorith, adopted by Şenel et al. (2020).

**How to run the full pipeline:**
```bash
bash imparting-interpretability/imparting_interpretability_demo.sh
```
This generates the interpretable embeddings used to initialize the transformer classifier.

## Full Pipeline

1. **Generate interpretable embeddings:**
   ```bash
   cd imparting-interpretability
   bash imparting_interpretability_demo.sh
   ```

2. **Train the model:**
   ```bash
   python src/train/train.py
   ```

3. **Evaluate model performance:**
   ```bash
   python src/eval/evaluate.py
   ```

4. **Generate interpretability quiz:**
   ```bash
   python src/eval/generate_quiz.py
   python src/eval/evaluate_quiz.py
   ```

5. **Run all evaluations:**
   ```bash
   bash src/eval/eval.bash
   ```

## Data

- `data/processed/vocab.txt` - Vocabulary of unique tokens
- `data/processed/vectors_imbued.txt` - Interpretable embedding vectors (from imparting_interpretability)
- `data/processed/trainingdata.csv` - Mental health labeled dataset
- `data/processed/word_group_ids.txt` - Mapping of concept words to dimension indices

## Output

- `output/classification_report.json` - Per-class metrics (precision, recall, F1)
- `output/topk_by_dimension.json` - Top sequences for each dimension
- `output/quiz_without_answers.json` - Human interpretability quiz
- `output/quiz_with_answers.json` - Quiz with correct answers