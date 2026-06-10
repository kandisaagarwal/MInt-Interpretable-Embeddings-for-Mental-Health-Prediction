import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(
    __file__), "../..")))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.mental_health_dataset import LABEL2ID
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")

# Hugging Face token is read from the environment so secrets are not hard-coded.
HF_TOKEN = os.environ.get("HF_TOKEN")  


class MentalBERTTrainer:
    """Fine-tune a pretrained MentalBERT model for mental health classification."""

    def __init__(
        self,
        data_path,
        model_name="mental/mental-bert-base-uncased",
        batch_size=32,
        epochs=5,
        lr=3e-5,
        max_len=256,
        use_weighted_loss=True,
        efficient_finetuning=True,
        train_last_n_layers=1,
        use_amp=True,
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.use_weighted_loss = use_weighted_loss
        self.efficient_finetuning = efficient_finetuning
        self.train_last_n_layers = train_last_n_layers
        self.use_amp = use_amp and torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
        }
        print(f"Using device: {self.device}")

    def _get_final_model_path(self):
        # Final model = the last weights seen at the end of training.
        return os.path.join(BASE_DIR, "src/model", "mentalbert_final.pt")

    def _get_best_model_path(self):
        # Best model = checkpoint with the highest validation macro-F1.
        return os.path.join(BASE_DIR, "src/model", "mentalbert_best.pt")

    def _load_existing_final_model(self, model):
        # Resume from the final model file if it already exists on disk.
        path = self._get_final_model_path()
        if not os.path.exists(path):
            return False

        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        print(f"Loaded existing model weights from {path}")
        return True

    def _configure_efficient_finetuning(self, model):
        """Freeze most layers and only train the classifier head plus last N encoder layers."""
        if not self.efficient_finetuning:
            # Full fine-tuning: every parameter remains trainable.
            for param in model.parameters():
                param.requires_grad = True
            return

        # Start by freezing everything, then selectively re-enable gradients.
        for param in model.parameters():
            param.requires_grad = False

        # Always train the classifier head.
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True

        base_model = getattr(model, "base_model", None)
        if base_model is None:
            return

        # Optional pooling layers on some architectures.
        if hasattr(base_model, "pooler") and base_model.pooler is not None:
            for param in base_model.pooler.parameters():
                param.requires_grad = True

        encoder = getattr(base_model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            layers = list(encoder.layer)
            if self.train_last_n_layers > 0:
                # Only the last few transformer blocks are updated during training.
                for layer in layers[-self.train_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def _count_trainable_parameters(self, model):
        # Useful for checking how much of the backbone is actually being updated.
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in model.parameters())
        return trainable, total

    def _load_data_for_bert(self):
        """Load the CSV split, tokenize it, and build PyTorch DataLoaders."""
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        # Split once with stratification so each split keeps similar label balance.
        df = pd.read_csv(self.data_path)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=12,
            stratify=df["status"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=12,
            stratify=temp_df["status"],
        )

        train_texts = train_df["statement"].tolist()
        train_labels = train_df["status"].map(LABEL2ID).tolist()
        val_texts = val_df["statement"].tolist()
        val_labels = val_df["status"].map(LABEL2ID).tolist()
        test_texts = test_df["statement"].tolist()
        test_labels = test_df["status"].map(LABEL2ID).tolist()

        # Tokenizer/model are loaded from Hugging Face, using a token if required.
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=HF_TOKEN
        )

        def tokenize_batch(text_list, label_list, shuffle):
            # Convert raw text into padded token tensors ready for the model.
            encodings = tokenizer(
                text_list,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            dataset = TensorDataset(
                encodings["input_ids"],
                encodings["attention_mask"],
                torch.tensor(label_list, dtype=torch.long),
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        # Build separate loaders for training, validation, and testing.
        train_loader = tokenize_batch(train_texts, train_labels, shuffle=True)
        val_loader = tokenize_batch(val_texts, val_labels, shuffle=False)
        test_loader = tokenize_batch(test_texts, test_labels, shuffle=False)

        return train_loader, val_loader, test_loader, tokenizer

    def train(self):
        # Prepare tokenized data once up front.
        train_loader, val_loader, test_loader, tokenizer = self._load_data_for_bert()

        # Count labels in the training split so we can counter class imbalance.
        label_counts = Counter()
        for _, _, Y in train_loader:
            label_counts.update(Y.tolist())

        # Load the pretrained sequence classifier and attach a 7-class head.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=7,
            token=HF_TOKEN
        ).to(self.device)

        # If a previous final checkpoint exists, use it as the starting point.
        self._load_existing_final_model(model)

        # Optionally freeze most of the backbone and only fine-tune a small part.
        self._configure_efficient_finetuning(model)
        trainable_params, total_params = self._count_trainable_parameters(model)
        print(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100.0 * trainable_params / total_params:.2f}%)"
        )

        # Class weights help the loss pay more attention to underrepresented labels.
        total = sum(label_counts.values())
        num_classes = len(LABEL2ID)
        weights = torch.tensor(
            [total / (num_classes * label_counts[i]) for i in range(num_classes)],
            dtype=torch.float,
        ).to(self.device)

        if self.use_weighted_loss:
            loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = nn.CrossEntropyLoss()

        # AdamW is the standard optimizer for transformer fine-tuning.
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=self.lr,
        )
        # Mixed precision reduces memory use and can speed up training on GPU.
        scaler = GradScaler(enabled=self.use_amp)
        # Track the best validation scores seen during training.
        best_val_acc = 0.0
        best_val_macro_f1 = float("-inf")
        best_epoch = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0

            # Iterate over mini-batches and update the model after each one.
            for input_ids, attention_mask, Y in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"
            ):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # Clear old gradients before computing the new backward pass.
                optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    # The Hugging Face model returns loss directly when labels are provided.
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=Y,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            # Average the batch losses so each epoch has a single training-loss value.
            avg_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs}  loss={avg_loss:.4f}")

            # Validation drives model selection, not the training loss.
            val_loss, val_acc, val_macro_f1 = self.evaluate(model, val_loader, loss_fn)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)

            if val_macro_f1 > best_val_macro_f1:
                # Save the best model separately so it can be reused later.
                best_val_macro_f1 = val_macro_f1
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), self._get_best_model_path())

        # Print a compact summary and persist the run history for later analysis.
        self._print_summary(best_epoch, best_val_acc, best_val_macro_f1)
        self._save_history("mentalbert_history.json")

        # Save the final checkpoint from the last epoch as a separate artifact.
        torch.save(model.state_dict(), self._get_final_model_path())

    def evaluate(self, model, loader, loss_fn):
        # Evaluation uses inference mode so dropout is disabled and gradients are off.
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, Y in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                Y = Y.to(self.device)

                # Forward pass only; no parameter updates happen during validation.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, Y)
                total_loss += loss.item()

                # Pick the class with the highest score for each example.
                preds = logits.argmax(dim=-1)
                correct += (preds == Y).sum().item()
                total += Y.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(Y.cpu().tolist())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        print(f" val_loss={avg_loss:.4f}  val_acc={accuracy:.4f}")
        # Print per-class precision/recall/F1 so mistakes are easier to inspect.
        print(
            classification_report(
                all_labels, all_preds, target_names=list(LABEL2ID.keys())
            )
        )

        # Macro-F1 treats all classes equally, which is useful for imbalanced data.
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f" val_macro_f1={macro_f1:.4f}")

        return avg_loss, accuracy, macro_f1

    def _print_summary(self, best_epoch, best_val_acc, best_val_macro_f1):
        # Final run summary for quick comparison across experiments.
        print("\n" + "="*60)
        print("MENTALBERT TRAINING SUMMARY")
        print("="*60)
        print(f"Best Epoch: {best_epoch}/{self.epochs}")
        print(f"Best Val Macro F1: {best_val_macro_f1:.4f}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
        print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
        print("="*60 + "\n")

    def _save_history(self, filename):
        # Store metrics as JSON so they can be plotted or compared later.
        history_path = os.path.join(BASE_DIR, filename)
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    # Default entry point: train using the dataset path defined above.
    trainer = MentalBERTTrainer(data_path)
    trainer.train()