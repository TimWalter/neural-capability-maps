import argparse

import torch

from ram.logger import Logger
from ram.model import Model
from ram.dataset.loader import ValidationSet, TrainingSet
from ram.train import validate_boundary, validate


def main(model_id: int, batch_size: int, set_name: str):
    device = torch.device("cuda")

    training_set = TrainingSet(batch_size, True)
    validation_set = ValidationSet(batch_size, False, set_name)
    boundary_set = ValidationSet(batch_size, False, validation_set.path + "_boundary")

    model = Model.from_id(model_id).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    logger = Logger(None, training_set, validation_set, boundary_set, {}, 1, -1, 3e-4, model)

    validate_boundary(model, logger, device, boundary_set, loss_function)
    validate(model, logger, device, validation_set, loss_function)
    logger.run.log(data={}, step=logger.step + 1, commit=True)


if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=53)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--set_name", type=str, default="test", help="Set type to validate on")
    args = parser.parse_args()

    main(**vars(args))
