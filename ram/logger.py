import json
from pathlib import Path

import optuna
import wandb
import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor

from ram.model import Model
from paper_archive.utils import bootstrap_mean_ci


class Logger:
    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 trial: optuna.Trial | None,
                 hyperparameter: dict,
                 model: Model,
                 group: str | None = None,
                 threshold:float =0.5
                 ):
        """
        Initialise the logger.

        Args:
            trial: Optuna trial.
            hyperparameter: Dict of hyperparameters for metadata and model loading.
            model: Model for saving.
            group: W&B group.
            threshold: Binary classification threshold
        """
        self.threshold = threshold
        metadata = {"hyperparameter": hyperparameter}
        self.run = self.setup_wandb(trial, metadata, group)

        parts = self.run.name.split("-")
        self.folder = Path(__file__).parent.parent / "data" / "trained_models" / f"{parts[-1]}-{'-'.join(parts[:-1])}"
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        json.dump(metadata, open(self.folder / 'metadata.json', 'w'), indent=4)

        self.model = model
        self.buffer = {}

    @jaxtyped(typechecker=beartype)
    def setup_wandb(self, trial: optuna.Trial | None, metadata: dict, group: str | None = None) -> wandb.Run:
        """
        Set up the Weights & Biases run.

        Args:
            trial: Optuna trial for naming the run.
            metadata: metadata.
            group: W&B group.

        Return:
            Weights & Biases run
        """
        # wandb.login(key="")
        run = wandb.init(project="RAM", config=metadata, group=group,
                         dir=Path(__file__).parent.parent / "data" / "wandb")
        if trial is not None:
            run.name = f"trial/{trial.number}/{run.name}"

        return run

    @jaxtyped(typechecker=beartype)
    def save_model(self):
        """
        Save the model as "model.pth".
        """
        torch.save(self.model.state_dict(), self.folder / "model.pth")

    @jaxtyped(typechecker=beartype)
    def checkpoint(self):
        """
        Save the model as "checkpoint.pth".
        """
        torch.save(self.model.state_dict(), self.folder / "checkpoint.pth")

    @jaxtyped(typechecker=beartype)
    def __del__(self):
        """
        Upon deletion ensure the W&B run finished.
        """
        self.run.finish()

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def log_training(self,
                     label: Bool[Tensor, "batch"],
                     logit: Float[Tensor, "batch"],
                     loss: Float[Tensor, ""]
                     ):
        """
        Create the log of a training step and post it to W&B.

        Args:
            label: Reachability labels.
            logit: Predicted logits.
            loss: Loss on the batch.
        """
        data = {"Loss": loss,
                "Reachable [%]": label.sum().item() / label.shape[0] * 100}
        data |= self.compute_metrics(logit, label, threshold=self.threshold)
        data = self.assign_space(data, "Training")

        self.run.log(data=data, commit=True)

    @jaxtyped(typechecker=beartype)
    def log_validation(self,
                       morph_index: Int[Tensor, "batch"],
                       label: Bool[Tensor, "batch"],
                       logit: Float[Tensor, "batch"],
                       loss: float | Float[Tensor, ""]| Float[Tensor, "1"],
                       ):
        """
        Create the log of a validation step. Requires a call to aggregate_validation to post to W&B.

        Args:
            morph_index: Morphology indices.
            label: Reachability labels.
            logit: Predicted logits.
            loss: Loss on the batch.

        """
        if "morph_index" not in self.buffer:
            self.buffer["morph_index"] = []
        if "label" not in self.buffer:
            self.buffer["label"] = []
        if "logit" not in self.buffer:
            self.buffer["logit"] = []
        if "loss" not in self.buffer:
            self.buffer["loss"] = 0.0

        self.buffer["morph_index"] += [morph_index.cpu()]
        self.buffer["label"] += [label.cpu()]
        self.buffer["logit"] += [logit.cpu()]
        self.buffer["loss"] += loss

    @jaxtyped(typechecker=beartype)
    def aggregate_validation(self, boundary: bool):
        """
        Aggregate validation steps and post to W&B.

        Args:
            boundary: Whether we evaluated random or boundary samples.
        """
        logit = torch.cat(self.buffer["logit"])
        label = torch.cat(self.buffer["label"])
        morph_index = torch.cat(self.buffer["morph_index"])

        data = self.compute_metrics(logit, label, morph_index, self.threshold)
        data |= {"Loss": self.buffer["loss"] / logit.shape[0]}

        data = self.assign_space(data, "Boundary" if boundary else "Validation")
        self.run.log(data=data, commit=False)
        self.buffer = {}

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def assign_space(data: dict, space: str) -> dict:
        """
        Assign data to a panel in W&B.

        Args:
            data: Data to assign.
            space: Name of the panel.
        Returns:
            Assigned data.
        """
        for key in list(data.keys()):
            data[f"{space}/{key}"] = data.pop(key)
        return data

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def compute_metrics(
            logit: Float[Tensor, "batch"],
            label: Bool[Tensor, "batch"],
            morph_index: Tensor | None = None,
            threshold: float = 0.5
    ) -> dict:
        """
        Compute logging metrics.

        Args:
            logit: Predicted logits.
            label: Reachability labels.
            morph_index: Morphology indices.
            threshold: Binary classification threshold.
        Returns:
            Metrics
        """

        metrics = {'Confidence': 2 * (torch.nn.Sigmoid()(logit) - 0.5).abs().mean().item()}

        confusion_matrix = binary_confusion_matrix(logit, label, morph_index, threshold)
        tp = confusion_matrix[:, 0, 0]
        fn = confusion_matrix[:, 0, 1]
        fp = confusion_matrix[:, 1, 0]
        tn = confusion_matrix[:, 1, 1]

        f1 = 2 * tp / (2 * tp + fp + fn + 1e-6) * 100
        perfect_negatives = (tp == 0.0) & (fn == 0.0) & (fp == 0.0) & (tn > 0.0)
        f1[perfect_negatives] = 100.0

        p_total = torch.clamp(tp + fn, min=1.0)
        n_total = torch.clamp(fp + tn, min=1.0)

        tpr = (tp / p_total) * 100
        fnr = (fn / p_total) * 100
        fpr = (fp / n_total) * 100
        tnr = (tn / n_total) * 100

        balanced_accuracy = 0.5 * (tpr + tnr)

        morph_metrics = torch.stack([balanced_accuracy, tpr, fnr, fpr, tnr, f1], dim=-1)
        mean_vals, ci_lower, ci_upper = bootstrap_mean_ci(morph_metrics, n_bootstraps=1000, ci=95)
        metric_names = ['Balanced Accuracy',
                        'True Positive Rate',
                        'False Negative Rate',
                        'False Positive Rate',
                        'True Negative Rate',
                        'F1 Score']
        for i, name in enumerate(metric_names):
            metrics[f'{name} (Mean)'] = mean_vals[i].item()
            metrics[f'{name} (CI Lower)'] = ci_lower[i].item()
            metrics[f'{name} (CI Upper)'] = ci_upper[i].item()

        return metrics


@jaxtyped(typechecker=beartype)
def binary_confusion_matrix(logit: Float[Tensor, "batch"],
                            label: Bool[Tensor, "batch"],
                            morph_index: Int[Tensor, "batch"] | None = None,
                            threshold:float = 0.5) \
        -> Float[Tensor, "n_morphs 2 2"]:
    """
    Compute the binary confusion matrix. Macro-averaged if morph_index is not None.

    Args:
        logit: Predicted logits.
        label: Label.
        morph_index: Morphology indices.
        threshold: Binary classification threshold.

    Returns:
         Binary confusion matrix.
    """
    mask = morph_index
    if morph_index is None:
        mask = torch.zeros_like(label).int()
    unique_morphs, mapped_indices = torch.unique(mask, return_inverse=True)
    num_morphs = len(unique_morphs)

    predicted = torch.sigmoid(logit) > threshold
    tp_count = torch.bincount(mapped_indices, weights=(predicted & label).float(), minlength=num_morphs)
    fn_count = torch.bincount(mapped_indices, weights=(~predicted & label).float(), minlength=num_morphs)
    fp_count = torch.bincount(mapped_indices, weights=(predicted & ~label).float(), minlength=num_morphs)
    tn_count = torch.bincount(mapped_indices, weights=(~predicted & ~label).float(), minlength=num_morphs)

    confusion_matrix = torch.zeros(num_morphs, 2, 2)
    confusion_matrix[:, 0, 0] = tp_count
    confusion_matrix[:, 0, 1] = fn_count
    confusion_matrix[:, 1, 0] = fp_count
    confusion_matrix[:, 1, 1] = tn_count

    return confusion_matrix
