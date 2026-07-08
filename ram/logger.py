import json
from pathlib import Path

import optuna
import wandb
import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor

from ram.model import Model
from ram.dataset.loader import TrainingSet, ValidationSet
from paper_archive.utils import bootstrap_mean_ci


class Logger:
    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 trial: optuna.Trial | None,
                 training_set: TrainingSet,
                 validation_set: ValidationSet,
                 boundary_set: ValidationSet,
                 hyperparameter: dict,
                 epochs: int,
                 early_stopping: int,
                 lr: float,
                 model: Model,
                 ):
        """
        Initialise the logger.

        Args:
            trial: Optuna trial.
            training_set: Training set.
            validation_set: Validation set.
            boundary_set: Validation set with boundary poses.
            hyperparameter: Dict of hyperparameters for metadata.
            epochs: Epochs for metadata.
            early_stopping: Early stopping criteria for metadata.
            lr: Learning rate for metadata.
            model: Model for saving.
        """
        self.training_set = training_set
        self.validation_set = validation_set
        self.boundary_set = boundary_set
        metadata = {"num_training_samples": len(training_set) * training_set.batch_size,
                    "num_validation_samples": len(validation_set) * validation_set.batch_size,
                    "num_boundary_samples": len(boundary_set) * boundary_set.batch_size,
                    "hyperparameter": hyperparameter,
                    "epochs": epochs,
                    "batch_size": training_set.batch_size,
                    "early_stopping": early_stopping,
                    "lr": lr}
        self.run = self.setup_wandb(trial, metadata)

        parts = self.run.name.split("-")
        self.folder = Path(__file__).parent.parent / "data" / "trained_models" / f"{parts[-1]}-{'-'.join(parts[:-1])}"
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        json.dump(metadata, open(self.folder / 'metadata.json', 'w'), indent=4)

        self.model = model
        self.buffer = {}
        self.step = 0

    @jaxtyped(typechecker=beartype)
    def setup_wandb(self, trial: optuna.Trial | None, metadata: dict) -> wandb.Run:
        """
        Set up the Weights & Biases run.

        Args:
            trial: Optuna trial for naming the run.
            metadata: metadata.

        Return:
            Weights & Biases run
        """
        # wandb.login(key="")
        run = wandb.init(project="RAM", config=metadata, dir=Path(__file__).parent.parent / "data" /"wandb")
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
        self.run.finish()

    @jaxtyped(typechecker=beartype)
    @torch.no_grad()
    def log_training(self,
                     epoch: int,
                     batch_idx: int,
                     label: Bool[Tensor, "batch"],
                     logit: Float[Tensor, "batch"],
                     loss: Float[Tensor, ""]
                     ):
        """
        Create the log of a training step and post it to W&B.

        Args:
            epoch: Current epoch, used to calculate the overall training step.
            batch_idx: Current batch idx, used to calculate the overall training step.
            label: Reachability labels.
            logit: Predicted logits.
            loss: Loss on the batch.
        """
        data = {"Loss": loss / self.training_set.batch_size,
                "Reachable [%]": label.sum().item() / label.shape[0] * 100}
        data |= self.compute_metrics(logit, label)
        data = self.assign_space(data, "Training")

        self.step = epoch * len(self.training_set) + batch_idx
        self.run.log(data=data, step=self.step, commit=True)

    @jaxtyped(typechecker=beartype)
    def log_intermediate_validation(self,
                                    label: Bool[Tensor, "batch"],
                                    logit: Float[Tensor, "batch"],
                                    loss: Float[Tensor, ""]
                                    ):
        """
        Create the log of an intermediate validation step and post it to W&B.

        Args:
            label: Reachability labels.
            logit: Predicted logits.
            loss: Loss on the batch.
        """

        data = {"Loss": loss / self.validation_set.batch_size,
                "Reachable [%]": label.sum().item() / label.shape[0] * 100}
        data |= self.compute_metrics(logit, label)
        data = self.assign_space(data, "Intermediate Validation")

        self.run.log(data=data, step=self.step + 1, commit=False)

    @jaxtyped(typechecker=beartype)
    def log_validation(self,
                       batch_idx: int,
                       label: Bool[Tensor, "batch"],
                       logit: Float[Tensor, "batch"],
                       loss: Float[Tensor, ""],
                       boundary: bool = False):
        """
        Log a validation step.

        """
        if "loss" not in self.buffer:
            self.buffer["loss"] = 0.0
        if "logit" not in self.buffer:
            self.buffer["logit"] = []
        if "label" not in self.buffer:
            self.buffer["label"] = []
        if "morph_index" not in self.buffer:
            self.buffer["morph_index"] = []

        self.buffer["label"] += [label.cpu()]
        self.buffer["logit"] += [logit.cpu()]
        self.buffer["loss"] += loss

        active_set = self.boundary_set if boundary else self.validation_set
        self.buffer["morph_index"] += [active_set._get_batch(batch_idx)[:, 0].long().cpu()]
        if batch_idx + 1 == len(active_set):
            data = {"Loss": self.buffer["loss"] / len(active_set) / active_set.batch_size}
            data |= self.compute_metrics(torch.cat(self.buffer["logit"]),
                                         torch.cat(self.buffer["label"]),
                                         torch.cat(self.buffer["morph_index"]))


            data = self.assign_space(data, "Validation")
            if boundary:
                data = self.assign_space(data, "Boundary")

            self.run.log(data=data, step=self.step + 1, commit=False)
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
            morph_index: Tensor | None = None
    ) -> dict:
        """
        Compute classification metrics.

        Args:
            logit: Predicted logits.
            label: Reachability labels.
            morph_index: If grouped by morphology index, we calculate mean and confidence interval.

        Returns:
            Dictionary with binary confusion matrix, F1 Score, and a prediction histogram.
        """
        metrics = {'Confidence': 2*(torch.nn.Sigmoid()(logit)-0.5).abs().mean()}

        confusion_matrix = binary_confusion_matrix(logit, label, morph_index)
        tp = confusion_matrix[:, 0, 0]
        fn = confusion_matrix[:, 0, 1]
        fp = confusion_matrix[:, 1, 0]
        tn = confusion_matrix[:, 1, 1]

        valid_mask = (tp + fn > 0.0) & (fp + tn > 0.0)
        tp = tp[valid_mask]
        fn = fn[valid_mask]
        fp = fp[valid_mask]
        tn = tn[valid_mask]

        morph_metrics = torch.stack([
            tp, fn, fp, tn,
            2 * tp / (2 * tp + fp + fn + 1e-6) * 100
        ], dim=-1)

        mean_vals, ci_lower, ci_upper = bootstrap_mean_ci(morph_metrics, n_bootstraps=1000, ci=95)
        metric_names = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives', 'F1 Score']
        for i, name in enumerate(metric_names):
            metrics[f'{name} (Mean)'] = mean_vals[i].item()
            metrics[f'{name} (CI Lower)'] = ci_lower[i].item()
            metrics[f'{name} (CI Upper)'] = ci_upper[i].item()

        return metrics


@jaxtyped(typechecker=beartype)
def binary_confusion_matrix(logit: Float[Tensor, "batch"],
                            label: Bool[Tensor, "batch"],
                            morph_index:Int[Tensor, "batch"]|None = None) \
        -> Float[Tensor, "2 2"] | Float[Tensor, "n_morphs 2 2"]:
    """
    Compute the binary confusion matrix.

    Args:
        logit: Predicted logits.
        label: Label.
        morph_index: Morphology indices.

    Returns:
         Binary confusion matrix.
    """
    if morph_index is None:
        morph_index = torch.zeros_like(label).int()

    predicted = torch.nn.Sigmoid()(logit) > 0.5

    unique_morphs, mapped_indices = torch.unique(morph_index, return_inverse=True)
    num_morphs = len(unique_morphs)

    tp_count = torch.bincount(mapped_indices, weights=(predicted & label).float(), minlength=num_morphs)
    fn_count = torch.bincount(mapped_indices, weights=(~predicted & label).float(), minlength=num_morphs)
    fp_count = torch.bincount(mapped_indices, weights=(predicted & ~label).float(), minlength=num_morphs)
    tn_count = torch.bincount(mapped_indices, weights=(~predicted & ~label).float(), minlength=num_morphs)

    p_total = tp_count + fn_count
    n_total = fp_count + tn_count

    confusion_matrix = torch.zeros(num_morphs, 2, 2)
    confusion_matrix[:, 0, 0] = (tp_count / p_total) * 100
    confusion_matrix[:, 0, 1] = (fn_count / p_total) * 100
    confusion_matrix[:, 1, 0] = (fp_count / n_total) * 100
    confusion_matrix[:, 1, 1] = (tn_count / n_total) * 100

    return confusion_matrix
