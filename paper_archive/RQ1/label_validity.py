import pickle
from pathlib import Path

import torch
from tqdm import tqdm

import ram.dataset.r3 as r3
import ram.dataset.so3 as so3
import ram.dataset.se3 as se3

from ram.dataset.morphology import sample_morph
from ram.dataset.workspace import fk_approximation, synthesise_data
from ram.dataset.boundaries import sample_boundary
from ram.metrics import binary_confusion_matrix, bootstrap_mean_ci, counts_to_rates
from paper_archive.utils import latex_mean_and_ci

torch.manual_seed(0)
device = torch.device("cuda")

cache = Path(__file__).parent / "cache"
cache.mkdir(parents=True, exist_ok=True)

# Config
levels = [1, 2, 3, 4]
intervals = [1, 1, 10, 30]
num_robots = 50
num_samples = 100_000
num_geodesic = 1000
num_geodesic_samples = 100

# Data Generation
morphs = {}
labels = {}
cell_indices = {}
boundary_labels = {}
boundary_cell_indices = {}
for dof in [5, 6, 7]:
    morphs[dof] = sample_morph(num_robots, dof, False, device)
    labels[dof] = []
    cell_indices[dof] = []
    boundary_labels[dof] = []
    boundary_cell_indices[dof] = []
    for m in morphs[dof]:
        # Random
        p, l = synthesise_data(m, num_samples, True, True)
        c = se3.index(p)
        labels[dof] += [l]
        cell_indices[dof] += [c]
        # Boundary
        bp, bl = sample_boundary(m, num_geodesic, num_geodesic_samples)
        bl = bl.cpu()
        bc = se3.index(bp).cpu()
        boundary_labels[dof] += [bl]
        boundary_cell_indices[dof] += [bc]

if not any(cache.iterdir()):
    cell_distance = []
    benchmarks = []
    # Table 1
    n_cells = []
    runtime = []
    balanced_accuracy_random = []
    balanced_accuracy_boundary = []

    # Table A1
    confusion_matrix_random = {"All": [], 7: [], 6: [], 5: []}
    confusion_matrix_boundary = {"All": [], 7: [], 6: [], 5: []}

    for level, interval in zip(levels, intervals):
        se3.set_level(level)
        print(f"[LEVEL{se3.LEVEL}]")

        cell_distance += [[se3.MIN_DISTANCE_BETWEEN_CELLS, se3.MAX_DISTANCE_BETWEEN_CELLS]]
        n_cells += [so3.N_CELLS * (torch.linalg.norm(r3.cell(torch.arange(0, r3.N_CELLS)), dim=1) < 1.0).sum()]

        level_benchmarks = []
        # Table 1
        level_runtime = []
        level_balanced_accuracy_random = []
        level_balanced_accuracy_boundary = []

        # Table A1
        level_confusion_matrix_random = {"All": [], 7: [], 6: [], 5: []}
        level_confusion_matrix_boundary = {"All": [], 7: [], 6: [], 5: []}

        for dof in [5, 6, 7]:
            print(f"[DOF{dof}]")
            batch_size = None

            for (m, l, c, bl, bc) in tqdm(zip(morphs[dof],
                                              labels[dof], cell_indices[dof],
                                              boundary_labels[dof], boundary_cell_indices[dof]
                                              ), desc="Looping Morphologies"):
                # Random
                morph_runtime = 0.0
                morph_benchmarks = []

                r_indices = torch.empty(0, dtype=torch.int64, device="cpu")
                while True:
                    new_r_indices, benchmark, batch_size = fk_approximation(m, True,
                                                                            seconds=interval,
                                                                            batch_size=batch_size)
                    r_indices = torch.cat([r_indices, new_r_indices]).unique()
                    morph_benchmarks.append(torch.tensor(benchmark))
                    morph_runtime += interval

                    logit = torch.isin(c, r_indices)
                    tpr, fnr, fpr, tnr = counts_to_rates(
                        *binary_confusion_matrix(logit.float(), l).flatten(start_dim=1).unbind(dim=-1))

                    if tpr > 95.0 or morph_runtime > 600:
                        break

                # Boundary
                boundary_logit = torch.isin(bc, r_indices)
                tpr_boundary, fnr_boundary, fpr_boundary, tnr_boundary = counts_to_rates(
                    *binary_confusion_matrix(boundary_logit.float(), bl).flatten(start_dim=1).unbind(dim=-1))

                # Aggregate
                morph_benchmarks = torch.stack(morph_benchmarks)
                aggregated_morph_benchmark = morph_benchmarks.sum(dim=0)
                aggregated_morph_benchmark[0] = r_indices.shape[0]
                aggregated_morph_benchmark[2:] = morph_benchmarks.mean(dim=0)[2:]
                level_benchmarks += [aggregated_morph_benchmark]

                level_runtime += [morph_runtime]

                level_balanced_accuracy_random += [(tpr + tnr) / 2]
                level_balanced_accuracy_boundary += [(tpr_boundary + tnr_boundary) / 2]

                level_confusion_matrix_random["All"] += [[tpr, fnr, fpr, tnr]]
                level_confusion_matrix_boundary["All"] += [[tpr_boundary, fnr_boundary, fpr_boundary, tnr_boundary]]
                level_confusion_matrix_random[dof] += [[tpr, fnr, fpr, tnr]]
                level_confusion_matrix_boundary[dof] += [[tpr_boundary, fnr_boundary, fpr_boundary, tnr_boundary]]

        # Aggregate (As this is final compute CIs
        benchmarks += [bootstrap_mean_ci(torch.stack(level_benchmarks))]

        runtime += [bootstrap_mean_ci(torch.tensor(level_runtime).unsqueeze(1))]
        balanced_accuracy_random += [bootstrap_mean_ci(torch.tensor(level_balanced_accuracy_random).unsqueeze(1))]
        balanced_accuracy_boundary += [bootstrap_mean_ci(torch.tensor(level_balanced_accuracy_boundary).unsqueeze(1))]

        for key in level_confusion_matrix_random.keys():
            confusion_matrix_random[key] += [bootstrap_mean_ci(torch.tensor(level_confusion_matrix_random[key]))]
            confusion_matrix_boundary[key] += [bootstrap_mean_ci(torch.tensor(level_confusion_matrix_boundary[key]))]

    with open(cache / "cache.pkl", "wb") as file:
        pickle.dump([cell_distance, n_cells, runtime, balanced_accuracy_random, balanced_accuracy_boundary,
                     confusion_matrix_random, confusion_matrix_boundary, benchmarks], file)
else:
    (cell_distance, n_cells, runtime, balanced_accuracy_random, balanced_accuracy_boundary,
     confusion_matrix_random, confusion_matrix_boundary, benchmarks) = pickle.load(open(cache / "cache.pkl", "rb"))

# Table 1
print(r"""
\begin{tblr}{
            colspec = {l r r r},
            row{1} = {font=\bfseries}, 
        }
        \toprule
        Cell Spacing & \# Cells & Runtime (s) & Balanced Accuracy (\%) \\
        \midrule""")
for i, level in enumerate(levels):
    current_runtime = ""
    if runtime[i][0] == 1.0:
        current_runtime = r"$\leq 1$"
    else:
        current_runtime = latex_mean_and_ci(*runtime[i])
    random = latex_mean_and_ci(*balanced_accuracy_random[i])
    print(
        rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$ & ${int(n_cells[i]):,}$ & {current_runtime} & {random} \\\addlinespace")
print(r"""\bottomrule
    \end{tblr}""")

# Table A1
print(r"""
    \begin{tblr}{
                colspec = {l r r r r r},
                row{1} = {font=\bfseries},
            }
            \toprule
            Cell Spacing & DoF & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)\\
            \midrule""")
for i, l in enumerate(levels):
    print(rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$")
    for dof in ("All", 5, 6, 7):
        row = rf"& {dof} \\"
        for cm in [confusion_matrix_random]:
            cm_inner = torch.stack(cm[dof][i], dim=-1)
            tpr = latex_mean_and_ci(*cm_inner[0])
            fnr = latex_mean_and_ci(*cm_inner[1])
            fpr = latex_mean_and_ci(*cm_inner[2])
            tnr = latex_mean_and_ci(*cm_inner[3])
            row += rf"& {tpr} & {fnr} & {fpr} & {tnr}"
        row += r"\\"
        print(row)
print(r"""\bottomrule
    \end{tblr}""")



# Table A3
print(r"""
    \begin{tblr}{
                colspec = {l r r r r r},
                row{1} = {font=\bfseries},
            }
            \toprule
            Cell Spacing & DoF & TPR (\%) & FNR (\%) & FPR (\%) & TNR (\%)\\
            \midrule""")
for i, l in enumerate(levels):
    print(rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$")
    for dof in ("All", 5, 6, 7):
        row = rf"& {dof} \\"
        for cm in [confusion_matrix_boundary]:
            cm_inner = torch.stack(cm[dof][i], dim=-1)
            tpr = latex_mean_and_ci(*cm_inner[0])
            fnr = latex_mean_and_ci(*cm_inner[1])
            fpr = latex_mean_and_ci(*cm_inner[2])
            tnr = latex_mean_and_ci(*cm_inner[3])
            row += rf"& {tpr} & {fnr} & {fpr} & {tnr}"
        row += r"\\"
        print(row)
print(r"""\bottomrule
    \end{tblr}""")

# Table A3
print(r"""
\begin{tblr}{
            colspec = {l r r r},
            row{1} = {font=\bfseries}, 
        }
        \toprule
        Cell Spacing & \# Cells & Runtime (s) & Balanced Accuracy (\%) \\
        \midrule""")
for i, level in enumerate(levels):
    current_runtime = ""
    if runtime[i][0] == 1.0:
        current_runtime = r"$\leq 1$"
    else:
        current_runtime = latex_mean_and_ci(*runtime[i])
    boundary = latex_mean_and_ci(*balanced_accuracy_boundary[i])
    print(
        rf"$\left[{cell_distance[i][0]:.3f}, {cell_distance[i][1]:.3f}\right]$ & ${int(n_cells[i]):,}$ & {current_runtime} & {boundary} \\\addlinespace")
print(r"""\bottomrule
    \end{tblr}""")

# Benchmark summary (debug-only)
metric_names = ["filled_cells", "total_samples",
                "total_efficiency (%)", "unique_efficiency (%)",
                "collision_efficiency (%)"]
for i, level in enumerate(levels):
    print(f"[LEVEL {level}]")
    mean, lower, upper = benchmarks[i]
    for name, m, lo, up in zip(metric_names, mean, lower, upper):
        print(f"  {name:<26} {latex_mean_and_ci(m, lo, up)}")
