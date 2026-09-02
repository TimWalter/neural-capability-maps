import pickle
from pathlib import Path

import torch

from ram.metrics import compute_metrics
from ram.validate import best_confidence_threshold
from ram.dataset.loader import HomogeneousPoseSet

DOF_KEYS = ["All", "7", "6", "5"]


def load_split(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the cached GGIK predictions of a split.

    Args:
        path: Name of the dataset split.
    Returns:
        Logits, reachability labels and morphology indices.
    """
    directory = Path(__file__).parent / "cache" / "ggik" / path

    distance = torch.load(directory / "distances.pth").flatten()
    label = torch.load(directory / "labels.pth")
    morph_index = torch.load(directory / "morph_indices.pth")

    logit = -distance

    return logit, label, morph_index


def dof_masks(path: str, morph_index: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Build the per-DoF sample masks of a split.

    Args:
        path: Name of the dataset split.
        morph_index: Morphology indices of the samples.
    Returns:
        Mask per DoF key.
    """
    morphologies = HomogeneousPoseSet(1, False, path, torch.device("cpu")).morphologies
    # The morphologies are zero-padded DH tables with one row per joint plus the base transform.
    dof = (morphologies.abs().sum(dim=2) != 0).sum(dim=1) - 1
    sample_dof = dof[morph_index]

    masks = {"All": torch.ones_like(morph_index, dtype=torch.bool)}
    for dof_key in DOF_KEYS[1:]:
        masks[dof_key] = sample_dof == int(dof_key)
    return masks


if __name__ == "__main__":
    torch.manual_seed(0)

    splits = {path: load_split(path) for path in ["test", "test_boundary"]}

    # Mirrors ram.validate.validate: a single threshold is shared by the random and boundary split.
    threshold = best_confidence_threshold(*splits["test"])
    print(f"Determined threshold: {threshold}")

    for path, (logit, label, morph_index) in splits.items():
        masks = dof_masks(path, morph_index)

        split_results = {}
        for dof_key, mask in masks.items():
            split_results[dof_key] = compute_metrics(logit[mask], label[mask], morph_index[mask], threshold)

        directory = Path(__file__).parent / "cache" / "ggik" / path
        pickle.dump(split_results, open(directory / "results.pickle", "wb"))
        print(path, split_results)
