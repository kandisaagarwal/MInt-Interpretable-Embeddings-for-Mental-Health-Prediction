import sys
import os
import math
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn as nn
from src.model.mental_health_dataset import load_data, LABEL2ID
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

vocab_path = os.path.join(BASE_DIR, "data/processed/vocab.txt")
glove_path = os.path.join(BASE_DIR, "data/processed/glove_subset_for_vocab.txt")
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
forced_ids_path = os.path.join(BASE_DIR, "data/processed/word_group_ids.txt")


def _infer_glove_dim(parts):
    """Last ``dim`` fields must be floats; everything before is the word (may contain spaces)."""
    # Try each possible split point from right to left until the trailing tokens
    # all parse cleanly as floating-point numbers.
    for dim in range(len(parts) - 1, 0, -1):
        try:
            [float(x) for x in parts[-dim:]]
        except ValueError:
            continue
        if parts[:-dim]:
            return dim
    return None


def load_vocab_and_glove_vectors(vocab_path, glove_path):
    """
    Same vocab / UNK / PAD convention as load_vocab_and_vectors, but embedding rows
    come from a GloVe word2vec-style file (word + floats per line).

    Words in vocab.txt missing from the GloVe file use the mean GloVe vector.
    """
    # Read the GloVe file into a lookup table: word -> embedding vector.
    glove = {}
    dim = None
    with open(glove_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if dim is None:
                dim = _infer_glove_dim(parts)
                if dim is None:
                    continue
            if len(parts) < dim + 1:
                continue
            try:
                vec = np.array([float(x) for x in parts[-dim:]], dtype=np.float64)
            except ValueError:
                continue
            word = " ".join(parts[:-dim]).strip()
            if not word:
                continue
            glove[word] = vec

    if not glove or dim is None:
        raise ValueError(f"No vectors loaded from {glove_path} (check format: word(s) + {dim or '?'} floats per line)")

    # Use the mean embedding for OOV words so the model still has a reasonable
    # starting point for vocabulary items that never appear in the GloVe file.
    dim = next(iter(glove.values())).shape[0]
    mean_vec = np.mean(np.stack(list(glove.values())), axis=0)

    # Build a vocab mapping and a dense embedding matrix in the same order.
    vocab = {}
    rows = []
    with open(vocab_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            word = line.split()[0]
            vocab[word] = idx
            rows.append(glove[word].copy() if word in glove else mean_vec.copy())

    # Add explicit special tokens used by the dataset loader.
    vocab["<unk>"] = len(vocab)
    vocab["<PAD>"] = len(vocab)
    unk_row = mean_vec.copy()
    rows.append(unk_row)
    rows.append(np.zeros((dim,), dtype=np.float64))

    vectors = torch.FloatTensor(np.stack(rows))
    UNK_ID = vocab["<unk>"]
    PAD_ID = vocab["<PAD>"]
    return vocab, vectors, UNK_ID, PAD_ID


class GloveTrainer:
    """Train/evaluate a Transformer classifier initialized from GloVe embeddings."""

    def __init__(
        self,
        vocab_path,
        glove_path,
        data_path,
        batch_size=32,
        epochs=25,
        lr=1e-4,
        max_len=512,
        num_classes=7,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        mlp_hidden_dim=256,
        freeze_embeddings=False,
        max_pooling=True,
        interpretability_loss=False,
        interpretability_k=1e-2,
        use_weighted_loss=True,
        early_stopping_patience=5,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.5,
        min_lr=5e-7,
    ):
        self.vocab_path = vocab_path
        self.glove_path = glove_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.mlp_hidden_dim = mlp_hidden_dim
        self.freeze_embeddings = freeze_embeddings
        self.max_pooling = max_pooling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_interpretability_loss = interpretability_loss
        self.interpretability_k = interpretability_k
        self.use_weighted_loss = use_weighted_loss
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr

    def _build_forced_dim_mask(self, ids_path, vocab_size, embedding_dim):
        """
        Build a boolean mask with shape [vocab_size, embedding_dim].

        mask[word_id, dim] is True when that token should be encouraged to have
        positive activation in that embedding/hidden dimension.
        """
        # Each line in the ids file corresponds to one constrained dimension.
        dim_to_ids = []
        with open(ids_path, "r") as f:
            for line in f:
                line = line.strip()
                ids = [int(tok) for tok in line.split()]
                dim_to_ids.append(ids)

        # Limit to the overlap between available embedding dimensions and file rows.
        num_dims = min(embedding_dim, len(dim_to_ids))
        mask = torch.zeros((vocab_size, embedding_dim), dtype=torch.bool)
        for dim_idx in range(num_dims):
            for word_id in dim_to_ids[dim_idx]:
                if 0 <= word_id < vocab_size:
                    mask[word_id, dim_idx] = True
        return mask

    def _interpretability_loss(self, encoded, input_ids, attention_mask, forced_mask):
        """Compute interpretability penalty on selected token-dimension pairs."""
        # Ignore padding positions because they are not real tokens.
        valid_positions = attention_mask == 1
        if not valid_positions.any():
            return encoded.new_tensor(0.0)

        # Keep only the token ids and hidden states for actual words.
        valid_ids = input_ids[valid_positions]
        valid_encoded = encoded[valid_positions]
        selected_dims = forced_mask[valid_ids]

        if not selected_dims.any():
            return encoded.new_tensor(0.0)

        # Penalize activations in the dimensions that are supposed to stay small
        # or positive, using the same shaping function as the main trainer.
        x = valid_encoded[selected_dims]
        alpha = 0.5
        eps = 1e-8
        cost = torch.where(
            x < 0.5,
            alpha * torch.exp(-x / alpha),
            (1.0 / math.e) * alpha * alpha / (x.abs() + eps),
        )
        return cost.mean()

    def train(self):
        # 1) Load vocab and build embedding matrix from the GloVe file.
        vocab, vectors, UNK_ID, PAD_ID = load_vocab_and_glove_vectors(
            self.vocab_path, self.glove_path
        )

        # 2) Build train/val/test loaders with the same tokenization pipeline.
        train_loader, val_loader, test_loader = load_data(
            self.data_path, vocab, self.max_len, self.batch_size, PAD_ID, UNK_ID
        )

        # Count labels in the training split so we can rebalance the loss.
        label_counts = Counter()
        for _, _, Y in train_loader:
            label_counts.update(Y.tolist())

        # Instantiate the transformer classifier with the preloaded GloVe matrix.
        model = TransformerEncoderClassifier(
            embedding_matrix=vectors,
            num_classes=self.num_classes,
            pad_idx=PAD_ID,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            mlp_hidden_dim=self.mlp_hidden_dim,
            max_len=self.max_len,
            freeze_embeddings=self.freeze_embeddings,
            max_pooling=self.max_pooling,
        ).to(self.device)

        forced_mask = None
        if self.use_interpretability_loss:
            # The optional regularizer needs to know which embedding dimensions
            # are associated with which vocabulary ids.
            forced_mask = self._build_forced_dim_mask(
                forced_ids_path,
                vocab_size=vectors.shape[0],
                embedding_dim=vectors.shape[1],
            ).to(self.device)

        # Weight classes inversely proportional to how often they appear.
        total = sum(label_counts.values())
        num_classes = len(LABEL2ID)
        weights = torch.tensor(
            [
                total / (num_classes * label_counts[i])
                for i in range(num_classes)
            ],
            dtype=torch.float,
        ).to(self.device)

        if self.use_weighted_loss:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        # Adam is used for parameter updates during training.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Reduce LR when validation macro-F1 stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.min_lr,
        )

        # Early-stopping state tracked on validation macro-F1.
        best_val_macro_f1 = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            interpretability_loss = None

            # Iterate over mini-batches and update the model after each one.
            for X, attention_mask, Y in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"
            ):
                X = X.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # Forward pass returns class scores plus intermediate encoded states.
                logits, _, encoded = model(X, attention_mask)
                classification_loss = loss_fn(logits, Y)

                if self.use_interpretability_loss:
                    # Add a secondary penalty on selected hidden dimensions.
                    interpretability_loss = self._interpretability_loss(
                        encoded,
                        X,
                        attention_mask,
                        forced_mask,
                    )
                    loss = classification_loss + self.interpretability_k * interpretability_loss
                else:
                    loss = classification_loss

                # Standard gradient update step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # These per-epoch prints summarize the most recent batch values and
            # the average training loss across the full epoch.
            print(f"Classification loss: {classification_loss.item():.4f}")
            print(
                f"Interpretability loss: {interpretability_loss.item() if interpretability_loss is not None else 'N/A'}"
            )
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            # Validate each epoch and use macro-F1 for tuning decisions.
            val_loss, val_acc, val_macro_f1 = self.evaluate(
                model, val_loader, loss_fn, return_loss=True
            )

            # Update scheduler from macro-F1 (higher is better).
            scheduler.step(val_macro_f1)

            if val_macro_f1 > best_val_macro_f1:
                # Keep a CPU copy of the best model so the final save uses the
                # best validation checkpoint rather than the last epoch.
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                print(f"New best val_macro_f1: {best_val_macro_f1:.4f} at epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= self.early_stopping_patience:
                print(
                    f"Early stopping triggered after {epoch+1} epochs. "
                    f"Best val_macro_f1: {best_val_macro_f1:.4f} at epoch {best_epoch+1}"
                )
                break

        # Save the best-performing model found during training.
        model_save_path = os.path.join(BASE_DIR, "src/model", "glove_model.pt")
        # Save best model found during training (fallback to final model if needed).
        if best_model_state is not None:
            torch.save(best_model_state, model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)

        # Store enough metadata to reproduce the model shape and training setup.
        metadata = {
            "labels": LABEL2ID,
            "vocab_size": len(vocab),
            "max_len": self.max_len,
            "num_classes": self.num_classes,
            "pad_idx": PAD_ID,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "freeze_embeddings": self.freeze_embeddings,
            "max_pooling": self.max_pooling,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "use_interpretability_loss": self.use_interpretability_loss,
            "interpretability_k": self.interpretability_k,
            "use_weighted_loss": self.use_weighted_loss,
            "early_stopping_patience": self.early_stopping_patience,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "lr_scheduler_factor": self.lr_scheduler_factor,
            "min_lr": self.min_lr,
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
            "embedding_source": "glove_subset_for_vocab.txt",
        }
        metadata_path = os.path.join(BASE_DIR, "src/model", "glove_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved model to {model_save_path}")
        print(f"Saved metadata to {metadata_path}")

    def evaluate(self, model, loader, loss_fn, return_loss=False):
        # Evaluation runs with model in inference mode and no gradient tracking.
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, attention_mask, Y in loader:
                X = X.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # Compute predictions only; no parameter updates happen here.
                logits, _, _ = model(X, attention_mask)
                loss = loss_fn(logits, Y)
                total_loss += loss.item()

                # Convert scores to predicted class ids.
                preds = logits.argmax(dim=-1)
                correct += (preds == Y).sum().item()
                total += Y.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(Y.cpu().tolist())

        # Aggregate metrics across the full loader.
        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f" val_loss={avg_loss:.4f}  val_acc={accuracy:.4f}  val_macro_f1={macro_f1:.4f}")
        print(
            classification_report(
                all_labels, all_preds, target_names=list(LABEL2ID.keys())
            )
        )

        if return_loss:
            return avg_loss, accuracy, macro_f1
        return accuracy


if __name__ == "__main__":
    # Default entry point: train from configured dataset + GloVe paths above.
    trainer = GloveTrainer(vocab_path, glove_path, data_path)
    trainer.train()
