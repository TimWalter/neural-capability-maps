import argparse

import torch
from tqdm import tqdm

from ram.logger import Logger
from ram.model import Model
from ram.dataset.loader import HomogeneousPoseSet
from ram.train import validate_boundary, validate

def best_threshold(model: Model, val_set: HomogeneousPoseSet, n_grid: int = 1024) -> float:
    """
    Pick the confidence threshold that maximises the macro-averaged balanced accuracy on a set.

    Args:
        model: Model to threshold.
        val_set: Set to optimise the threshold on.
        n_grid: Number of candidate thresholds.
    Returns:
        Confidence threshold, in probability space as used by the logger.
    """
    logits, labels, morph_indices = [], [], []
    model.eval()
    with torch.no_grad():
        for morph, pose, label, morph_index in tqdm(val_set, desc="Thresholding"):
            logits.append(model.predict(morph, pose).cpu())
            labels.append(label.cpu())
            morph_indices.append(morph_index.cpu())
    logit, label = torch.cat(logits), torch.cat(labels)
    morph_index = torch.cat(morph_indices)

    order = torch.linspace(0, logit.numel() - 1, n_grid, dtype=torch.float64).long()
    grid = logit.sort().values[order].unique()
    n_thresh = grid.numel()

    _, morph_inv = morph_index.unique(return_inverse=True)
    n_morphs = int(morph_inv.max()) + 1

    bin_index = torch.bucketize(logit, grid)
    flat = (morph_inv * 2 + label) * (n_thresh + 1) + bin_index
    counts = torch.bincount(flat, minlength=n_morphs * 2 * (n_thresh + 1))
    neg, pos = counts.view(n_morphs, 2, n_thresh + 1).unbind(1)

    tp = pos.flip(-1).cumsum(-1).flip(-1)[:, 1:].double()
    fp = neg.flip(-1).cumsum(-1).flip(-1)[:, 1:].double()
    n_pos, n_neg = pos.sum(-1, keepdim=True), neg.sum(-1, keepdim=True)

    tpr = tp / n_pos.clamp(min=1)
    tnr = 1 - fp / n_neg.clamp(min=1)
    balanced_accuracy = 0.5 * (tpr + tnr)

    valid = (n_pos > 0) & (n_neg > 0)
    macro = (balanced_accuracy * valid).sum(0) / valid.sum(0).clamp(min=1)

    return torch.sigmoid(grid[macro.argmax()].double()).item()

def main(model_id: int, batch_size: int, val_set_path: str, test_set_path: str, group: str):
    device = torch.device("cuda")

    model = Model.from_id(model_id).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean')

    val_set = HomogeneousPoseSet(batch_size, False, val_set_path, device)
    threshold = best_threshold(model, val_set)
    print(f"Determined threshold: {threshold}")

    test_set = HomogeneousPoseSet(batch_size, False, test_set_path, device)
    boundary_set = HomogeneousPoseSet(batch_size, False, test_set.path + "_boundary", device)
    logger = Logger(None, {}, model, group, threshold=threshold)

    validate_boundary(model, logger, boundary_set, loss_function)
    validate(model, logger, test_set, loss_function)
    logger.run.log(data={}, commit=True)


if __name__ == '__main__':
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, default=772)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--val_set_path", type=str, default="val")
    parser.add_argument("--test_set_path", type=str, default="test")
    parser.add_argument("--group", type=str, default=None, help="W&B group")
    args = parser.parse_args()

    main(**vars(args))
