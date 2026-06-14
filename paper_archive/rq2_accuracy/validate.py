import torch
from pathlib import Path
from tqdm import tqdm
from ram.logger import Logger
from ram.dataset.loader import ValidationSet

def evaluate_ggik_split(path_name: str):
    """Loads data for a specific split and computes metrics for All, 7, 6, and 5 DoF."""
    print(f"--- Processing GGIK split: {path_name} ---")

    # Load data
    directory = Path.cwd() / "data" / path_name
    se3_dist = torch.load(directory / "se3_dist.pth")
    labels = torch.load(directory/ "labels.pth")

    if path_name == "test_boundary":
        morph_indices = []
        mask5 = []
        mask6 = []
        mask7 = []

        eval_set = ValidationSet(1000, False, path_name)
        for batch_idx, (morph, pose, label) in enumerate(tqdm(eval_set, desc=path_name)):
            morph_indices += [eval_set._get_batch(batch_idx)[:, 0].long()]
            mask5 += [(morph.abs().sum(dim=2) != 0).sum(dim=1) == 6]
            mask6 += [(morph.abs().sum(dim=2) != 0).sum(dim=1) == 7]
            mask7 += [(morph.abs().sum(dim=2) != 0).sum(dim=1) == 8]

        morph_indices = torch.cat(morph_indices)
        mask5 = torch.cat(mask5)
        mask6 = torch.cat(mask6)
        mask7 = torch.cat(mask7)
    else:
        morph_indices = torch.load(directory / "morph_indices.pth")
        ones = torch.ones(100* 1000, dtype=torch.bool)
        zeros = torch.zeros(100* 1000, dtype=torch.bool)
        mask5 = torch.cat([ones, zeros, zeros])
        mask6 = torch.cat([zeros, ones, zeros])
        mask7 = torch.cat([zeros, zeros, ones])

    # Optimize the distance threshold using the grid search logic
    min_unreachable_distance = se3_dist[~labels].min()
    max_reachable_distance = se3_dist[labels].max()

    if max_reachable_distance > min_unreachable_distance:
        threshold = min_unreachable_distance
        best_f1 = 0.0
        for candidate in torch.linspace(min_unreachable_distance, max_reachable_distance, 100):
            logit = se3_dist < candidate
            f1 = Logger.compute_metrics(logit.float()[:, 0], labels, morph_indices)["F1 Score (Mean)"]
            if f1 > best_f1:
                threshold = candidate
                best_f1 = f1
    else:
        threshold = (max_reachable_distance + min_unreachable_distance) / 2

    # Final threshold prediction vector
    logit = se3_dist < threshold
    logit_flat = logit.float()[:, 0]

    # Define masks map
    dof_masks = {
        "All": torch.ones_like(labels, dtype=torch.bool),
        "7": mask7,
        "6": mask6,
        "5": mask5
    }

    split_results = {}
    for dof_key, mask in dof_masks.items():
        # Apply the current DoF mask slice
        m_logit = logit_flat[mask]
        m_labels = labels[mask]
        m_morphs = morph_indices[mask]

        # Calculate bootstrap metrics using our macro-filtered compute_metrics
        split_results[dof_key] = Logger.compute_metrics(m_logit, m_labels, m_morphs)

    return split_results

# --- Execute Evaluation for Both Splits ---
# Mapping: 'test' -> Random (%), 'boundary' -> Boundary (%)
ggik_random = evaluate_ggik_split("test")
ggik_boundary = evaluate_ggik_split("test_boundary")

# --- Generate LaTeX Output Snippet ---
def format_latex_cell(metrics_dict, metric_name):
    """Helper to cleanly format values into \num{mean(low:high)} blocks."""
    mean_key = f"{metric_name} (Mean)"
    low_key = f"{metric_name} (CI Lower)"
    high_key = f"{metric_name} (CI Upper)"

    if mean_key in metrics_dict:
        mean = metrics_dict[mean_key]
        low = metrics_dict.get(low_key, mean)
        high = metrics_dict.get(high_key, mean)
        return f"\\num{{{mean:.0f}({high-mean:.0f}:{mean-low:.0f})}}"
    return "\\num{}"

print("Random")
print(f"\\num{{{ggik_random["All"]["F1 Score (Mean)"]:.0f}({ggik_random["All"]["F1 Score (CI Upper)"]-ggik_random["All"]["F1 Score (Mean)"]:.0f}:{ ggik_random["All"]["F1 Score (Mean)"]- ggik_random["All"]["F1 Score (CI Lower)"]:.0f})}}")

print("Boundary")
print(f"\\num{{{ggik_boundary["All"]["F1 Score (Mean)"]:.0f}({ggik_boundary["All"]["F1 Score (CI Upper)"]-ggik_boundary["All"]["F1 Score (Mean)"]:.0f}:{ ggik_boundary["All"]["F1 Score (Mean)"]- ggik_boundary["All"]["F1 Score (CI Lower)"]:.0f})}}")

print("\n=== GENERATED LATEX ROWS FOR GGIK ===")
metric_keys = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives']

for idx, dof in enumerate(["All", "7", "6", "5"]):
    cls_label = "GGIK" if idx == 0 else ""

    # Grab metrics dictionaries for the specific DoF
    r_metrics = ggik_random.get(dof, {})
    b_metrics = ggik_boundary.get(dof, {})

    # Format cells for Random split
    r_cells = [format_latex_cell(r_metrics, k) for k in metric_keys]
    # Format cells for Boundary split
    b_cells = [format_latex_cell(b_metrics, k) for k in metric_keys]

    # Construct LaTeX formatting string matching your tabularray structure
    row_str = (
        f"        {cls_label:<10} & {dof:<4} "
        f"& {r_cells[0]} & {r_cells[1]} & {r_cells[2]} & {r_cells[3]} "
        f"& {b_cells[0]} & {b_cells[1]} & {b_cells[2]} & {b_cells[3]} \\\\"
    )
    print(row_str)