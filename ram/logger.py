import json
from pathlib import Path

import optuna
import wandb
import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor

from ram.model import Model
from ram.metrics import compute_metrics


class Logger:
    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 trial: optuna.Trial | None,
                 hyperparameter: dict,
                 model: Model,
                 group: str | None = None,
                 threshold: float = 0.5
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
        data |= compute_metrics(logit, label, threshold=self.threshold)
        data = self.assign_space(data, "Training")

        self.run.log(data=data, commit=True)

    @jaxtyped(typechecker=beartype)
    def log_validation(self,
                       morph_index: Int[Tensor, "batch"],
                       label: Bool[Tensor, "batch"],
                       logit: Float[Tensor, "batch"],
                       loss: float | Float[Tensor, ""] | Float[Tensor, "1"],
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
    def aggregate_validation(self, boundary: bool) -> float:
        """
        Aggregate validation steps and post to W&B.

        Args:
            boundary: Whether we evaluated random or boundary samples.

        Returns:
            Mean balanced accuracy.
        """
        logit = torch.cat(self.buffer["logit"])
        label = torch.cat(self.buffer["label"])
        morph_index = torch.cat(self.buffer["morph_index"])

        data = compute_metrics(logit, label, morph_index, self.threshold)
        data |= {"Loss": self.buffer["loss"] / logit.shape[0]}

        output = data["Balanced Accuracy (Mean)"]
        data = self.assign_space(data, "Boundary" if boundary else "Validation")
        self.run.log(data=data, commit=False)
        self.buffer = {}
        return output

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

