import argparse

import torch
import optuna
from tqdm import tqdm

from ram.logger import Logger
from ram.model import Model
from ram.dataset.loader import TrainingSet, ValidationSet


def main(epochs: int,
         batch_size: int,
         early_stopping: int,
         lr: float,
         pretrain: int,
         hyperparameter: dict,
         trial: optuna.Trial = None):
    device = torch.device("cuda")

    training_set = TrainingSet(batch_size, True)
    validation_set = ValidationSet(batch_size, False)
    boundary_set = ValidationSet(batch_size, False, validation_set.path + "_boundary")

    model = Model(**hyperparameter)
    if pretrain != -1:
        model = model.from_id(pretrain)
    model = model.to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    logger = Logger(trial, training_set, validation_set, boundary_set, hyperparameter, epochs, early_stopping, lr, model)

    min_loss = torch.inf
    early_stopping_counter = 0
    for e in range(epochs):
        # Training
        model.train()
        for batch_idx, (morph, pose, label) in enumerate(tqdm(training_set, desc=f"Training")):
            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            model.zero_grad()

            logit = model(morph, pose)
            loss = loss_function(logit, label.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logger.log_training(e, batch_idx, label, logit, loss)

            # Intermediate Validation
            if batch_idx % 100000 == 0:
                with torch.no_grad():
                    logger.checkpoint()
                    morph, pose, label = validation_set.get_semi_random_batch()
                    morph = morph.to(device, non_blocking=True)
                    pose = pose.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    logit = model.predict(morph, pose)
                    loss = loss_function(logit, label.float())
                    logger.log_intermediate_validation(label, logit, loss)

        # Validation
        model.eval()
        for batch_idx, (morph, pose, label) in enumerate(tqdm(boundary_set, desc=f"Validation - Boundary")):
            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logit = model.predict(morph, pose)
            loss = loss_function(logit, label.float())

            logger.log_validation(batch_idx, label, logit, loss, True)

        loss = 0.0
        for batch_idx, (morph, pose, label) in enumerate(tqdm(validation_set, desc=f"Validation")):
            morph = morph.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logit = model.predict(morph, pose)
            loss += loss_function(logit, label.float())

            logger.log_validation(batch_idx, label, logit, loss)
        loss /= len(validation_set) * batch_size

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
                del logger
                raise optuna.TrialPruned()

    return min_loss


if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--early_stopping", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrain", type=int, default=-1)
    args = parser.parse_args()

    main(**vars(args), hyperparameter={})
