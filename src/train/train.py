import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from src.model.mental_health_dataset import load_vocab_and_vectors, load_data, LABEL2ID
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

vocab_path = os.path.join(BASE_DIR, "data/processed/vocab_fixed.txt")
vectors_path = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
forced_ids_path = os.path.join(BASE_DIR,"data/processed/word_group_ids.txt",)

class Trainer:
    """End-to-end training helper for the mental health text classifier."""

    def __init__(
            self, 
            vocab_path, # path to pretrained vocab
            vectors_path, # path to pretrained vectors
            data_path, # path to training data
            batch_size=32, # batch size for training
            epochs=45, # number of training epochs
            lr=1e-4, # learning rate for Adam optimizer
            max_len=512, # Most inputs will fit in this context window  # maximum sequence length for padding/truncation
            interpretability_loss=True, # whether to include the interpretability regularization term in the loss function
            interpretability_k=1e-2, # single knob controlling regularization strength
            use_weighted_loss=True, # whether to use weighted loss to handle class imbalance
            early_stopping_patience=5, # number of epochs to wait for improvement before stopping
            lr_scheduler_patience=2, # number of epochs to wait before reducing LR
            lr_scheduler_factor=0.5, # factor to reduce LR by
            min_lr=5e-7, # minimum learning rate
            num_layers=4, # number of transformer encoder layers
            ff_dim=512, # feedforward dimension in transformer
            mlp_hidden_dim=256, # hidden dimension in MLP classifier head
        ):
        self.vocab_path = vocab_path
        self.vectors_path = vectors_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_interpretability_loss = interpretability_loss
        self.interpretability_k = interpretability_k
        self.use_weighted_loss = use_weighted_loss
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.min_lr = min_lr
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.mlp_hidden_dim = mlp_hidden_dim


    def _build_forced_dim_mask(self, ids_path, vocab_size, embedding_dim):
        """
        Build a mask M (vocab size x embedding dim) from word_group_ids.txt.
        M[word_id, dim] is True when that (word, embedding-dimension) pair
        should be encouraged by the interpretability regularizer.

        File format: each line lists the word IDs tied to one embedding dimension.
        """
        # Read "dimension -> list of word ids" from disk.
        dim_to_ids = []
        with open(ids_path, "r") as f:
            for line in f:
                line = line.strip()
                ids = [int(tok) for tok in line.split()]
                dim_to_ids.append(ids)

        # Build a boolean lookup table with shape [vocab_size, embedding_dim].
        # Only dimensions that exist in both the embedding and ids file are used.
        num_dims = min(embedding_dim, len(dim_to_ids))
        mask = torch.zeros((vocab_size, embedding_dim), dtype=torch.bool)
        for dim_idx in range(num_dims):
            for word_id in dim_to_ids[dim_idx]:
                if 0 <= word_id < vocab_size:
                    mask[word_id, dim_idx] = True
        return mask

    def _interpretability_loss(self, encoded, input_ids, attention_mask, forced_mask):
        """
        Compute interpretability penalty on token-level encoder outputs.

        We only score real tokens (not padding), and only the dimensions marked
        as important by forced_mask for each token id.
        """

        valid_positions = attention_mask == 1 # True for real tokens, False for padding
        if not valid_positions.any():
            return encoded.new_tensor(0.0)

        # Keep only non-padding tokens in this batch.
        valid_ids = input_ids[valid_positions]
        valid_encoded = encoded[valid_positions]
        selected_dims = forced_mask[valid_ids]

        if not selected_dims.any():
            return encoded.new_tensor(0.0)

        # x contains only encoder values at constrained (token, dimension) pairs.
        x = valid_encoded[selected_dims]
        alpha = 0.5 # controls how quickly the penalty decays as values approach zero
        eps = 1e-8 # to prevent division by zero
        # with alpha=0.5, below is the g(x) from https://arxiv.org/pdf/1807.07279
        cost = torch.where(
            x < 0.5,
            alpha * torch.exp(-x / alpha),
            (1.0 / math.e) * alpha * alpha / (x.abs() + eps),
        )
        return cost.mean()

    def train(self):
        # 1) Load vocabulary and pretrained embeddings used by the model's embedding layer.
        # Load vocab and vectors
        vocab, vectors, UNK_ID, PAD_ID = load_vocab_and_vectors(
            self.vocab_path, self.vectors_path
        )

        # 2) Build train/val/test DataLoaders with tokenization + padding/truncation.
        # Load data
        train_loader, val_loader, test_loader = load_data(
            self.data_path, vocab, self.max_len, self.batch_size, PAD_ID, UNK_ID
        )

        # count labels in training set
        label_counts = Counter()
        for _, _, Y in train_loader:
            label_counts.update(Y.tolist())

        # Build model
        model = TransformerEncoderClassifier(
            embedding_matrix=vectors,
            num_classes=7,
            pad_idx=PAD_ID,
            num_layers=self.num_layers,
            ff_dim=self.ff_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
        ).to(self.device)

        # Build constraints only when interpretability regularization is enabled
        forced_mask = None
        if self.use_interpretability_loss:
            # forced_mask[word_id, dim] tells the regularizer which embedding
            # dimensions should be active for each word id.
            forced_mask = self._build_forced_dim_mask(
                forced_ids_path,
                vocab_size=vectors.shape[0],
                embedding_dim=vectors.shape[1],
            ).to(self.device)
        
        # weighted loss to handle class imbalance, weight for class i is total_samples / (num_classes * count_i)
        total = sum(label_counts.values())
        num_classes = len(LABEL2ID)
        weights = torch.tensor([
            total / (num_classes * label_counts[i]) for i in range(num_classes)
        ], dtype=torch.float).to(self.device)

        # Loss and optimizer
        if self.use_weighted_loss:
            loss_fn = nn.CrossEntropyLoss(weight=weights) # weighted loss to handle class imbalance
        else:
            loss_fn = nn.CrossEntropyLoss() # standard unweighted loss
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        # Learning rate scheduler: Reduce LR on plateau (based on macro-f1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.min_lr,
        )

        # Early stopping variables
        best_val_macro_f1 = -1.0
        best_epoch = -1
        epochs_no_improve = 0
        best_model_state = None

        # --- Checkpoint loading logic ---
        checkpoint_dir = os.path.join(BASE_DIR, "src/model")
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pt"))
        start_epoch = 0
        if checkpoint_files:
            # If prior checkpoints exist, continue from the most recent epoch file.
            # Find latest checkpoint by epoch number
            def extract_epoch_num(f):
                import re
                m = re.search(r"model_epoch_(\d+).pt", os.path.basename(f))
                return int(m.group(1)) if m else -1
            checkpoint_files = sorted(checkpoint_files, key=extract_epoch_num)
            latest_ckpt = checkpoint_files[-1]
            latest_epoch = extract_epoch_num(latest_ckpt)
            print(f"Loading checkpoint {latest_ckpt} (epoch {latest_epoch})...")
            model.load_state_dict(torch.load(latest_ckpt, map_location=self.device))
            start_epoch = latest_epoch
            print(f"Resuming training from epoch {start_epoch+1}")
            # Optionally, load best model state if available
            best_model_path = os.path.join(checkpoint_dir, "model.pt")
            if os.path.exists(best_model_path):
                best_model_state = torch.load(best_model_path, map_location=self.device)

        for epoch in range(start_epoch, self.epochs):
            model.train()
            total_loss = 0.0

            for X, attention_mask, Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # move batch to device
                X = X.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # forward pass
                logits, _, encoded = model(X, attention_mask)

                # Standard classification loss
                classification_loss = loss_fn(logits, Y)

                if self.use_interpretability_loss:
                    # Add interpretability regularization from encoder outputs before pooling
                    interpretability_loss = self._interpretability_loss(
                        encoded,
                        X,
                        attention_mask,
                        forced_mask,
                    )
                    # Final objective includes both classification and interpretability terms
                    loss = classification_loss + self.interpretability_k * interpretability_loss
                else:
                    # Classification training only
                    loss = classification_loss

                # Standard optimization step.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Classification loss: {classification_loss.item():.4f}")
            print(f"Interpretability loss: {interpretability_loss.item() if self.use_interpretability_loss else 'N/A'}")
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            # Save checkpoint at each epoch
            checkpoint_path = os.path.join(BASE_DIR, "src/model", f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Evaluate on validation set after each epoch for model selection.
            val_loss, val_acc, val_macro_f1 = self.evaluate(model, val_loader, loss_fn, return_loss=True)

            # Step the scheduler based on macro-f1
            scheduler.step(val_macro_f1)

            # Early stopping check based on macro-f1
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                print(f"New best val_macro_f1: {best_val_macro_f1:.4f} at epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val_macro_f1: {best_val_macro_f1:.4f} at epoch {best_epoch+1}")
                break

        # Save the best model
        model_save_path = os.path.join(BASE_DIR, "src/model", "model.pt")
        if best_model_state is not None:
            torch.save(best_model_state, model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)

        # Save metadata (hyperparameters/config) so inference scripts know
        # how this model was trained.
        metadata = {
            "labels": LABEL2ID,
            "vocab_size": len(vocab),
            "max_len": self.max_len,
            "num_classes": 7,
            "pad_idx": PAD_ID,
            "num_heads": 4,
            "num_layers": self.num_layers,
            "ff_dim": self.ff_dim,
            "dropout": 0.1,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "freeze_embeddings": False,
            "max_pooling": True,
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
        }
        metadata_path = os.path.join(BASE_DIR, "src/model", "metadata.json")
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved model metadata to {metadata_path}")

    def evaluate(self, model, loader, loss_fn, return_loss=False):
        # Disable dropout etc. and evaluate without tracking gradients.
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

                logits, _, _ = model(X, attention_mask)
                loss = loss_fn(logits, Y)
                total_loss += loss.item()

                # Convert class scores to final predicted class IDs.
                preds = logits.argmax(dim=-1)
                correct += (preds == Y).sum().item()
                total += Y.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(Y.cpu().tolist())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0.0
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f" val_loss={avg_loss:.4f}  val_acc={accuracy:.4f}  val_macro_f1={macro_f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=list(LABEL2ID.keys())))

        if return_loss:
            return avg_loss, accuracy, macro_f1
        return accuracy

if __name__ == "__main__":
    # Default entry point: train using paths defined at the top of this file.
    trainer = Trainer(vocab_path, vectors_path, data_path)
    trainer.train()