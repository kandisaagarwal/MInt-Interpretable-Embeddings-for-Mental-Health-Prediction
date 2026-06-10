import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from src.model.mental_health_dataset import load_vocab_and_vectors, load_data, LABEL2ID
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

vocab_path = os.path.join(BASE_DIR, "data/processed/vocab.txt")
vectors_path = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")


class LinearClassifier(nn.Module):
    """Simple linear classifier on top of pretrained embeddings"""
    def __init__(self, embedding_matrix, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        embedding_dim = embedding_matrix.shape[1]
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.pad_idx = pad_idx

    def forward(self, input_ids, attention_mask):
        # Embed tokens
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Max pooling: mask out padding tokens before pooling
        embedded_masked = embedded.clone()
        mask_expanded = attention_mask.unsqueeze(-1).expand(embedded.shape).bool()
        embedded_masked[~mask_expanded] = float('-inf')
        
        # Max pool over sequence dimension
        pooled, _ = torch.max(embedded_masked, dim=1)  # (batch, embed_dim)
        
        # Handle case where all tokens are padding
        pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
        
        # Linear classification
        logits = self.linear(pooled)  # (batch, num_classes)
        return logits, pooled


class LinearTrainer:
    def __init__(
        self,
        vocab_path,
        vectors_path,
        data_path,
        batch_size=32,
        epochs=25,
        lr=3e-5,
        max_len=512,
        use_weighted_loss=True,
    ):
        self.vocab_path = vocab_path
        self.vectors_path = vectors_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.use_weighted_loss = use_weighted_loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train(self):
        # Load vocab and vectors
        vocab, vectors, UNK_ID, PAD_ID = load_vocab_and_vectors(
            self.vocab_path, self.vectors_path
        )

        # Load data
        train_loader, val_loader, test_loader = load_data(
            self.data_path, vocab, self.max_len, self.batch_size, PAD_ID, UNK_ID
        )

        # Count labels in training set
        label_counts = Counter()
        for _, _, Y in train_loader:
            label_counts.update(Y.tolist())

        # Build model
        model = LinearClassifier(
            embedding_matrix=vectors,
            num_classes=7,
            pad_idx=PAD_ID,
        ).to(self.device)

        # Weighted loss to handle class imbalance
        total = sum(label_counts.values())
        num_classes = len(LABEL2ID)
        weights = torch.tensor([
            total / (num_classes * label_counts[i]) for i in range(num_classes)
        ], dtype=torch.float).to(self.device)

        # Loss and optimizer
        if self.use_weighted_loss:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_val_acc = 0.0
        best_epoch = 0

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for X, attention_mask, Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Move batch to device
                X = X.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # Forward pass
                logits, _, _ = model(X, attention_mask)

                # Classification loss
                loss = loss_fn(logits, Y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            # Evaluate on val set after each epoch
            val_loss, val_acc = self.evaluate(model, val_loader, loss_fn)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(BASE_DIR, "src/model", "linear_best.pt"))

        # Save the final model
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "src/model", "linear_final.pt"))

        self._print_summary(best_epoch, best_val_acc)
        self._save_history()

    def evaluate(self, model, loader, loss_fn):
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

                preds = logits.argmax(dim=-1)
                correct += (preds == Y).sum().item()
                total += Y.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(Y.cpu().tolist())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        print(f" val_loss={avg_loss:.4f}  val_acc={accuracy:.4f}")
        print(classification_report(all_labels, all_preds, target_names=list(LABEL2ID.keys())))

        return avg_loss, accuracy

    def _print_summary(self, best_epoch, best_val_acc):
        print("\n" + "="*60)
        print("LINEAR CLASSIFIER SUMMARY")
        print("="*60)
        print(f"Best Epoch: {best_epoch}/{self.epochs}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
        print("="*60 + "\n")

    def _save_history(self):
        history_path = os.path.join(BASE_DIR, "linear_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    trainer = LinearTrainer(vocab_path, vectors_path, data_path)
    trainer.train()