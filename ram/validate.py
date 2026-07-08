import argparse

import torch
from tqdm import tqdm

from ram.logger import Logger
from ram.model import Model
from ram.dataset.loader import ValidationSet, TrainingSet


def main(model_id: int, batch_size: int):
    device = torch.device("cuda")

    training_set = TrainingSet(batch_size, True)
    validation_set = ValidationSet(batch_size, False, "test")
    boundary_set = ValidationSet(batch_size, False, validation_set.path + "_boundary")

    model = Model.from_id(model_id).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    logger = Logger(None, training_set, validation_set, boundary_set, {}, 1, -1, 3e-4, model)

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

        logger.log_validation(batch_idx, label, logit, loss, False)
    loss /= len(validation_set) * batch_size
    logger.run.log(data={}, step=logger.step+1, commit=True)

if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=142)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    main(**vars(args))
