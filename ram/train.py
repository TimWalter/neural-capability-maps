import random
import argparse

import torch
import optuna
from tqdm import tqdm

from ram.logger import Logger
from ram.model import Model
from ram.dataset.loader import IndexedCellSet, HomogeneousPoseSet


def validation(model: Model, logger: Logger, validation_set: HomogeneousPoseSet, loss_function) -> float:
    """
    Validate a model on the given set.

    Args:
        model: Model to evaluate.
        logger: Logger to document.
        validation_set: Data provider.
        loss_function: Loss function.

    Returns:
        Average loss.
    """
    loss = 0.0
    model.eval()
    for batch_idx, (morph, pose, label, morph_index) in enumerate(validation_set):
        logit = model.predict(morph, pose)
        loss += loss_function(logit, label.float())
        logger.log_validation(morph_index, label, logit, loss)
    loss /= len(validation_set)
    return loss.cpu().item()


def train(training_set_path: str,
          validation_set_path: str | None,
          pretrain: int,
          epochs: int,
          batch_size: int,
          early_stopping: int,
          validation_interval: int,
          lr: float,
          hyperparameter: dict,
          group: str | None,
          trial: optuna.Trial | None = None,
          stop_after: int | None = None) -> float:
    """
    Train a model

    Args:
        training_set_path: Path to the training set.
        validation_set_path: Path to the validation set can be None.
        pretrain: If unequal to -1, specifies the model id to initialise with.
        epochs: Number of epochs.
        batch_size: Batch size.
        early_stopping: If unequal to -1, specifies the number of underperforming validations that trigger early stopping.
        validation_interval: After how many batches should there be validation.
        lr: Learning rate.
        hyperparameter: Hyperparameters of the model.
        group: W&B group.
        trial: Optuna trial.
        stop_after: Determines the number of batches to run per epoch can be None.

    Returns:
        Minimum validation loss.
    """
    device = torch.device("cuda")

    model = Model(**hyperparameter)
    if pretrain != -1:
        model = model.from_id(pretrain)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    training_set = IndexedCellSet(batch_size, True, training_set_path, device)
    if validation_set_path is not None:
        validation_set = HomogeneousPoseSet(batch_size, False, validation_set_path, device)
        boundary_set = HomogeneousPoseSet(batch_size, False, validation_set.path + "_boundary", device)

    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
    min_loss = torch.inf
    early_stopping_counter = 0

    logger = Logger(trial, hyperparameter, model, group)
    for e in range(epochs):
        model.train()
        for batch_idx, (morph, pose, label, _) in enumerate(tqdm(training_set, desc=f"Training")):
            model.zero_grad()
            logit = model(morph, pose)
            loss = loss_function(logit, label.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logger.log_training(label, logit, loss)

            if stop_after is not None and stop_after <= batch_idx:
                break

            if validation_set_path is not None and batch_idx % validation_interval == 0:
                logger.checkpoint()
                with torch.no_grad():
                    validation(model, logger, boundary_set, loss_function)
                    logger.aggregate_validation(True)
                    loss = validation(model, logger, validation_set, loss_function)
                    logger.aggregate_validation(False)
                    if loss < min_loss:
                        min_loss = loss
                        early_stopping_counter = 0
                        logger.save_model()
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter == early_stopping:
                            (print('Early Stopping'))
                            return min_loss
                    if trial is not None:
                        trial.report(loss, e)
                        if trial.should_prune():
                            logger.run.finish()
                            raise optuna.TrialPruned()
                    model.train()
        logger.checkpoint()
    return min_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--training_set_path", type=str, default="train")
    parser.add_argument("--validation_set_path", type=str, default="val")
    parser.add_argument("--pretrain", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=4)
    parser.add_argument("--validation_interval", type=int, default=500_000)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--group", type=str, default=None, help="W&B group")
    args = parser.parse_args()

    if args.seed != -1:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    del args.seed

    train(**vars(args), hyperparameter={})
