import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from src.model.mental_health_dataset import load_data, LABEL2ID
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")


class RandomEmbeddingTrainer:
    """Baseline: Train transformer with random embeddings (no pretrained vectors)"""

    def __init__(
        self,
        data_path,
        batch_size=32,
        epochs=25,
        lr=1e-5,
        max_len=512,
        embedding_dim=200,
        use_weighted_loss=True,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.use_weighted_loss = use_weighted_loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _build_vocab_from_data(self, data_path):
        """Build vocabulary from CSV data"""
        df = pd.read_csv(data_path)
        word_counter = Counter()

        for text in df['statement']:
            words = str(text).lower().split()
            word_counter.update(words)

        PAD_ID = 0
        UNK_ID = 1
        vocab = {'<PAD>': PAD_ID, '<UNK>': UNK_ID}

        for word, _ in word_counter.most_common():
            if word not in vocab:
                vocab[word] = len(vocab)

        return vocab, PAD_ID, UNK_ID

    def _build_random_embedding_matrix(self, vocab_size, embedding_dim):
        """Build random embedding matrix with Xavier initialization"""
        embedding_matrix = torch.empty(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(embedding_matrix)
        return embedding_matrix

    def train(self):
        """Train model with random embeddings"""

        print("Building vocabulary from data...")
        vocab, PAD_ID, UNK_ID = self._build_vocab_from_data(self.data_path)
        print(f"Vocab size: {len(vocab)}")

        print("Loading data...")
        train_loader, val_loader, test_loader = load_data(
            self.data_path,
            vocab=vocab,
            max_len=self.max_len,
            batch_size=self.batch_size,
            pad_id=PAD_ID,
            unk_id=UNK_ID
        )

        embedding_matrix = self._build_random_embedding_matrix(len(vocab), self.embedding_dim)
        print(f"Created random embedding matrix: {embedding_matrix.shape}")

        label_counts = Counter()
        for _, _, Y in train_loader:
            label_counts.update(Y.tolist())

        num_layers = 6
        ff_dim = 512
        mlp_hidden_dim = 256
        model = TransformerEncoderClassifier(
            embedding_matrix=embedding_matrix,
            num_classes=7,
            pad_idx=PAD_ID,
            num_layers=num_layers,
            ff_dim=ff_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(self.device)

        total = sum(label_counts.values())
        num_classes = len(LABEL2ID)
        weights = torch.tensor([
            total / (num_classes * label_counts[i]) for i in range(num_classes)
        ], dtype=torch.float).to(self.device)

        if self.use_weighted_loss:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            for X, attention_mask, Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                X = X.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                logits, _, _ = model(X, attention_mask)
                loss = loss_fn(logits, Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            val_loss, val_acc = self.evaluate(model, val_loader, loss_fn)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(BASE_DIR, "src/model", "random_embedding_best.pt")
                )

        torch.save(
            model.state_dict(),
            os.path.join(BASE_DIR, "src/model", "random_embedding_final.pt")
        )

        self._print_summary(best_epoch, best_val_acc)
        self._save_history()

        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        test_loss, test_acc = self.evaluate(model, test_loader, loss_fn)

        # Save metadata
        metadata = {
            "labels": LABEL2ID,
            "vocab_size": len(vocab),
            "max_len": self.max_len,
            "num_classes": 7,
            "pad_idx": PAD_ID,
            "num_heads": 4,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "dropout": 0.1,
            "mlp_hidden_dim": mlp_hidden_dim,
            "embedding_type": "random",
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "use_weighted_loss": self.use_weighted_loss,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
        }
        metadata_path = os.path.join(BASE_DIR, "src/model", "random_embedding_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def evaluate(self, model, loader, loss_fn):
        """Evaluate model on validation/test set"""
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
        print(classification_report(
            all_labels, all_preds, target_names=list(LABEL2ID.keys())
        ))

        return avg_loss, accuracy

    def _print_summary(self, best_epoch, best_val_acc):
        print("\n" + "="*60)
        print("RANDOM EMBEDDING BASELINE - TRAINING SUMMARY")
        print("="*60)
        print(f"Best Epoch: {best_epoch}/{self.epochs}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
        print("="*60 + "\n")

    def _save_history(self):
        history_path = os.path.join(BASE_DIR, "random_embedding_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    trainer = RandomEmbeddingTrainer(data_path)
    trainer.train()