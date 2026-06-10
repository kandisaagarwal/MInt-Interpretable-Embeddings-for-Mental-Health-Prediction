import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from src.model.mental_health_dataset import load_vocab_and_vectors, load_data, LABEL2ID
from src.model.transformer_encoder_classifier import TransformerEncoderClassifier
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Integer class id -> string label (LABEL2ID is name -> id)
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}


def infer_transformer_hparams(state_dict):
    """Read num_layers, ff_dim, mlp_hidden_dim from a TransformerEncoderClassifier checkpoint."""
    layer_indices = []
    for key in state_dict:
        parts = key.split(".")
        if (
            len(parts) >= 4
            and parts[0] == "encoder"
            and parts[1] == "layers"
            and parts[2].isdigit()
        ):
            layer_indices.append(int(parts[2]))
    if not layer_indices:
        raise ValueError("Could not infer num_layers from state_dict (missing encoder.layers.* keys)")
    num_layers = max(layer_indices) + 1
    ff_dim = int(state_dict["encoder.layers.0.linear1.weight"].shape[0])
    mlp_hidden_dim = int(state_dict["classifier.0.weight"].shape[0])
    return num_layers, ff_dim, mlp_hidden_dim

vocab_path = os.path.join(BASE_DIR, "data/processed/vocab.txt")
vectors_path = os.path.join(BASE_DIR, "data/processed/vectors_imbued.txt")
data_path = os.path.join(BASE_DIR, "data/processed/trainingdata.csv")
model_path = os.path.join(BASE_DIR, "src/model", "model-best-quiz.pt")


class GradientImportanceAnalyzer:
    """Compute gradient-based importance scores for interpretable dimensions"""

    def load_concepts(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def get_concept_name(self, dim_idx):
        if not hasattr(self, "concepts"):
            # Only load once
            concepts_path = os.path.join(BASE_DIR, "imparting-interpretability", "concepts.txt")
            self.concepts = self.load_concepts(concepts_path)
        if 0 <= dim_idx < len(self.concepts):
            return self.concepts[dim_idx]
        else:
            return f"dim_{dim_idx+1}"

    def __init__(self, model_path, vocab_path, vectors_path, data_path, device=None):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.vectors_path = vectors_path
        self.data_path = data_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.vectors = None

    def load_model(self):
        """Load pretrained model"""
        vocab, vectors, UNK_ID, PAD_ID = load_vocab_and_vectors(
            self.vocab_path, self.vectors_path
        )
        self.vocab = vocab
        self.vectors = vectors
        self.pad_id = PAD_ID

        state = torch.load(self.model_path, map_location=self.device)
        num_layers, ff_dim, mlp_hidden_dim = infer_transformer_hparams(state)
        print(
            f"Checkpoint architecture: num_layers={num_layers}, ff_dim={ff_dim}, "
            f"mlp_hidden_dim={mlp_hidden_dim}"
        )

        model = TransformerEncoderClassifier(
            embedding_matrix=vectors,
            num_classes=7,
            pad_idx=PAD_ID,
            num_layers=num_layers,
            ff_dim=ff_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        ).to(self.device)

        model.load_state_dict(state)
        model.eval()
        
        # Ensure parameters require gradients for gradient computation
        for param in model.parameters():
            param.requires_grad = True
        
        self.model = model

    def compute_gradients(self, input_ids, attention_mask, target_class):
        """
        Compute gradients of output logit w.r.t. pooled representation
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            target_class: class index to compute gradients for
            
        Returns:
            pooled_repr: (batch, embed_dim) pooled representation
            gradients: (batch, embed_dim) gradients of logit w.r.t. pooled repr
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Enable gradients for this computation
        with torch.enable_grad():
            # Forward pass with gradient tracking (z = pooled sentence vector)
            logits, pooled, _ = self.model(input_ids, attention_mask)
            
            # Get logit for target class
            class_logit = logits[:, target_class]
            
            # ∂y_c/∂z (mean over batch if batch > 1)
            loss = class_logit.mean()  # Average over batch
            grads = torch.autograd.grad(
                loss, pooled, create_graph=False, retain_graph=False
            )[0]

        return pooled.detach(), grads.detach()

    def compute_importance_scores(self, input_ids, attention_mask, target_class):
        """
        Sensitivity(z_i) = ∂y_c/∂z_i
        Importance(z_i) = z_i * ∂y_c/∂z_i (gradient × activation)
        """
        pooled, grads = self.compute_gradients(input_ids, attention_mask, target_class)

        sensitivity = grads
        importance = pooled * grads

        return sensitivity, importance, pooled

    def analyze_batch(self, data_loader, num_constrained_dims=100):
        """
        Analyze importance scores across a batch
        
        Args:
            data_loader: DataLoader with (input_ids, attention_mask, labels)
            num_constrained_dims: number of interpretable dimensions
            
        Returns:
            results: dict with importance scores per class
        """
        self.model.eval()
        # For each class/logit, collect all importance vectors
        importance_by_class = {label: [] for label in LABEL2ID.keys()}

        # Note: No torch.no_grad() here since we need gradients for analysis
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(data_loader):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Get predictions (detached from computation graph)
            with torch.no_grad():
                logits, _, _ = self.model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)

            # Compute importance for each sample in batch
            for sample_idx in range(input_ids.shape[0]):
                pred_class = preds[sample_idx].item()
                true_class = labels[sample_idx].item()

                # Compute gradients for predicted class
                sample_input = input_ids[sample_idx:sample_idx+1]
                sample_mask = attention_mask[sample_idx:sample_idx+1]

                sensitivity, importance, pooled_repr = self.compute_importance_scores(
                    sample_input, sample_mask, pred_class
                )

                embed_dim = int(importance.shape[-1])
                n_constrained = min(num_constrained_dims, embed_dim)

                # First n_constrained interpretable dims; force 1D so argsort is never (1, n) vs (n,)
                constrained_importance = (
                    importance[0, :n_constrained].detach().cpu().float().numpy().reshape(-1)
                )
                constrained_sensitivity = (
                    sensitivity[0, :n_constrained].detach().cpu().float().numpy().reshape(-1)
                )

                # Ranking for all dimensions (by absolute importance)
                ranking_indices = np.argsort(np.abs(constrained_importance))[::-1]
                ranking = []
                for idx in ranking_indices:
                    concept = self.get_concept_name(idx)
                    ranking.append({
                        "dimension": int(idx+1),
                        "concept": concept,
                        "importance": float(constrained_importance[idx]),
                        "sensitivity": float(constrained_sensitivity[idx])
                    })

                k = min(5, constrained_importance.size)
                if k < 1:
                    top_dims = np.array([], dtype=np.int64)
                    top_vals = []
                else:
                    # Rank by magnitude of gradient×activation; values kept signed.
                    top_dims = np.argsort(np.abs(constrained_importance))[-k:][::-1]
                    top_vals = constrained_importance[top_dims].tolist()

                # Collect importance for the predicted class
                importance_by_class[ID2LABEL[pred_class]].append(constrained_importance)

        # Aggregate: for each class/logit, compute mean importance and ranking
        summary = {}
        for class_name, importances in importance_by_class.items():
            if len(importances) == 0:
                continue
            importances = np.stack(importances, axis=0)  # (num_samples, num_dims)
            mean_importance = np.mean(importances, axis=0)
            mean_sensitivity = np.mean(np.abs(importances), axis=0)
            ranking_indices = np.argsort(np.abs(mean_importance))[::-1]
            ranking = []
            for idx in ranking_indices:
                concept = self.get_concept_name(idx)
                ranking.append({
                    "dimension": int(idx+1),
                    "concept": concept,
                    "mean_importance": float(mean_importance[idx]),
                    "mean_abs_importance": float(np.abs(mean_importance[idx])),
                    "mean_abs_sensitivity": float(mean_sensitivity[idx])
                })
            summary[class_name] = {
                "mean_importance": mean_importance.tolist(),
                "dimension_ranking": ranking
            }
        return summary

    def save_results(self, results, output_path):
        """Save importance analysis to JSON"""
        output_file = os.path.join(BASE_DIR, output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    analyzer = GradientImportanceAnalyzer(
        model_path=model_path,
        vocab_path=vocab_path,
        vectors_path=vectors_path,
        data_path=data_path,
    )

    print("Loading model...")
    analyzer.load_model()

    print("Loading data...")
    vocab, _, UNK_ID, PAD_ID = load_vocab_and_vectors(vocab_path, vectors_path)
    train_loader, val_loader, test_loader = load_data(data_path, vocab, max_len=512, batch_size=8, pad_id=PAD_ID, unk_id=UNK_ID)

    # Aggregate all splits into one
    all_importance_by_class = {}
    for split_name, loader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
        print(f"Computing gradient importance scores for {split_name} set...")
        results = analyzer.analyze_batch(loader, num_constrained_dims=200)
        for class_name, class_info in results.items():
            if class_name not in all_importance_by_class:
                all_importance_by_class[class_name] = []
            all_importance_by_class[class_name].append(class_info)

    # Combine: average mean_importance and mean_abs_importance across splits for each class
    combined_results = {}
    for class_name, split_infos in all_importance_by_class.items():
        # Stack mean_importance arrays
        mean_importances = np.stack([info["mean_importance"] for info in split_infos], axis=0)
        avg_mean_importance = np.mean(mean_importances, axis=0)
        # Use concepts from the first split
        ranking_indices = np.argsort(np.abs(avg_mean_importance))[::-1]
        ranking = []
        for idx in ranking_indices:
            concept = analyzer.get_concept_name(idx)
            ranking.append({
                "dimension": int(idx+1),
                "concept": concept,
                "mean_importance": float(avg_mean_importance[idx]),
                "mean_abs_importance": float(np.abs(avg_mean_importance[idx]))
            })
        combined_results[class_name] = {
            "mean_importance": avg_mean_importance.tolist(),
            "dimension_ranking": ranking
        }

    print("Saving combined results to gradient_importance_analysis_combined.json ...")
    analyzer.save_results(combined_results, "gradient_importance_analysis_combined.json")

    # Print summary
    for class_name, class_info in combined_results.items():
        print(f"\n{class_name}: {len(class_info['mean_importance'])} dims")
        print("  Top 5 dimensions:")
        for r in class_info["dimension_ranking"][:5]:
            print(f"    {r['dimension']}: {r['concept']} (mean_importance={r['mean_importance']:.4f})")