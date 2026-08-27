import torch
from beartype import beartype
from jaxtyping import Float, Bool, jaxtyped, Int
from torch import Tensor


@jaxtyped(typechecker=beartype)
def compute_metrics(
        logit: Float[Tensor, "batch"],
        label: Bool[Tensor, "batch"],
        morph_index: Int[Tensor, "batch"] | None = None,
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

    tpr, fnr, fpr, tnr = counts_to_rates(tp, fn, fp, tn)
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
                            threshold: float = 0.5) \
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


@jaxtyped(typechecker=beartype)
def counts_to_rates(tp: Float[Tensor, "n_morphs"],
                    fn: Float[Tensor, "n_morphs"],
                    fp: Float[Tensor, "n_morphs"],
                    tn: Float[Tensor, "n_morphs"]) -> tuple[
    Float[Tensor, "n_morphs"],
    Float[Tensor, "n_morphs"],
    Float[Tensor, "n_morphs"],
    Float[Tensor, "n_morphs"]
]:
    """
    Transform binary confusion matrix counts into rates

    Args:
        tp: True positives.
        fn: False negatives.
        fp: False positives.
        tn: True negatives.

    Returns:
        True positive rate, false negative rate, false positive rate, true negative rate.
    """
    p_total = torch.clamp(tp + fn, min=1.0)
    n_total = torch.clamp(fp + tn, min=1.0)

    tpr = (tp / p_total) * 100
    fnr = (fn / p_total) * 100
    fpr = (fp / n_total) * 100
    tnr = (tn / n_total) * 100
    return tpr, fnr, fpr, tnr


@jaxtyped(typechecker=beartype)
def bootstrap_mean_ci(trajectories: Float[Tensor, "n_trajectories n_timepoints"], n_bootstraps: int = 1000,
                      ci: int = 95) \
        -> tuple[
            Float[Tensor, "n_timepoints"],
            Float[Tensor, "n_timepoints"],
            Float[Tensor, "n_timepoints"]
        ]:
    """
    Calculates the mean and confidence interval for a set of trajectories.

    Args:
        trajectories: A 2D tensor where each row is a trajectory.
                                     Shape: (n_trajectories, n_timepoints).
        n_bootstraps: The number of bootstrap samples to generate.
        ci: The desired confidence interval in percent.

    Returns:
        The mean trajectory and the confidence interval.
    """
    n_trajectories, n_timepoints = trajectories.shape

    boot_indices = torch.randint(
        low=0,
        high=n_trajectories,
        size=(n_bootstraps, n_trajectories),
        device=trajectories.device
    )

    bootstrap_samples = trajectories[boot_indices]
    bootstrap_means = torch.mean(bootstrap_samples, dim=1)
    mean_trajectory = torch.mean(trajectories, dim=0)

    lower_percentile = ((100 - ci) / 2) / 100
    upper_percentile = (100 - (100 - ci) / 2) / 100

    quantiles = torch.tensor([lower_percentile, upper_percentile], device=trajectories.device)
    ci_bounds = torch.quantile(bootstrap_means, quantiles, dim=0)

    # Basic (reverse-percentile) bootstrap: reflect quantiles around the observed mean
    ci_lower = 2 * mean_trajectory - ci_bounds[1]
    ci_upper = 2 * mean_trajectory - ci_bounds[0]

    return mean_trajectory, ci_lower, ci_upper
