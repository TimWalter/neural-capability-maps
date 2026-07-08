import torch
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import numpy as np

from paper_archive.utils import get_plt_colour

# --- CONFIGURATION ---
WANDB_ENTITY = "tim-walter-tum"
WANDB_PROJECT = "RAM"

# Map each DoF (1 to 9) to its corresponding W&B Run ID
dof_run_mapping = {
    1: "dcpxw4zi",
    2: "zid8a8ch",
    3: "s8j6971k",
    4: "aon9aux9",
    5: "zl7aqn4z",
    6: "1saij5m2",
    7: "6txg6jf6",
    8: "dw8vfmwd",
    9: "p7vnypho",
}
# ---------------------

sns.set_style("ticks")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "axes.titlesize": 34,
    "lines.linewidth": 3,
})

api = wandb.Api()

dof_list = sorted(list(dof_run_mapping.keys()))
ours_mean, ours_low, ours_high = [], [], []
ours_b_mean, ours_b_low, ours_b_high = [], [], []

print("Fetching data from Weights & Biases...")

for dof_val in dof_list:
    run_id = dof_run_mapping[dof_val]
    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        summary = run.summary

        # 1. Pull Random (Validation) Data
        r_mean = summary.get("Validation/F1 Score (Mean)", 0.0)
        r_low = summary.get("Validation/F1 Score (CI Lower)", r_mean)
        r_high = summary.get("Validation/F1 Score (CI Upper)", r_mean)

        ours_mean.append(r_mean)
        ours_low.append(r_low)
        ours_high.append(r_high)

        # 2. Pull Boundary Data
        b_mean_key = "Boundary/F1 Score (Mean)" if "Boundary/F1 Score (Mean)" in summary else "Boundary/Validation/F1 Score (Mean)"
        b_low_key = "Boundary/F1 Score (CI Lower)" if "Boundary/F1 Score (CI Lower)" in summary else "Boundary/Validation/F1 Score (CI Lower)"
        b_high_key = "Boundary/F1 Score (CI Upper)" if "Boundary/F1 Score (CI Upper)" in summary else "Boundary/Validation/F1 Score (CI Upper)"

        b_mean = summary.get(b_mean_key, 0.0)
        b_low = summary.get(b_low_key, b_mean)
        b_high = summary.get(b_high_key, b_mean)

        ours_b_mean.append(b_mean)
        ours_b_low.append(b_low)
        ours_b_high.append(b_high)

    except Exception as e:
        print(f"Warning: Failed to fetch DoF {dof_val}. Using zeros. Error: {e}")
        ours_mean.append(0.0); ours_low.append(0.0); ours_high.append(0.0)
        ours_b_mean.append(0.0); ours_b_low.append(0.0); ours_b_high.append(0.0)

# Convert arrays to numpy for straightforward horizontal line rendering
dof = np.array(dof_list)
r_mean = np.array(ours_mean)
r_low = np.array(ours_low)
r_high = np.array(ours_high)

b_mean = np.array(ours_b_mean)
b_low = np.array(ours_b_low)
b_high = np.array(ours_b_high)

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(30, 7))
width = 0.25

# --- RANDOM BARS ---
# 1. Solid base bar from 0 up to the Lower Confidence Bound
rects1_base = ax.bar(dof - width/2, r_low, width, color=get_plt_colour(0), zorder=2)
# 2. High alpha (translucent shadow) interval bar from Lower to Upper Bound
rects1_ci = ax.bar(dof - width/2, r_high - r_low, width, bottom=r_low, color=get_plt_colour(0), alpha=0.4, zorder=2)
# 3. Flat line highlighting the actual observed mean value inside the column structure
ax.hlines(y=r_mean, xmin=dof - width, xmax=dof, colors="black", linewidth=4, zorder=3)
# 4. Text labels positioned cleanly right above the highest reach of the CI block
ax.bar_label(rects1_ci, padding=4, labels=[int(round(m)) for m in r_mean], fontsize=34)


# --- BOUNDARY BARS ---
# 1. Solid base bar up to Lower bound
rects2_base = ax.bar(dof + width/2, b_low, width, color=get_plt_colour(1), hatch="//", zorder=2)
# 2. Translucent extension bar showing the performance variance interval
rects2_ci = ax.bar(dof + width/2, b_high - b_low, width, bottom=b_low, color=get_plt_colour(1), hatch="//", alpha=0.4, zorder=2)
# 3. Mean identifier strike line
ax.hlines(y=b_mean, xmin=dof, xmax=dof + width, colors="black", linewidth=4, zorder=3)
# 4. Text label representing the integer mean string
ax.bar_label(rects2_ci, padding=4, labels=[int(round(m)) for m in b_mean], fontsize=34)


# Distribution Spans
span_ood = ax.axvspan(0.5, 4.5, alpha=0.12, color=get_plt_colour(3), ymin=0, ymax=1)
span_id = ax.axvspan(4.5, 7.5, alpha=0.15, color=get_plt_colour(2), ymin=0, ymax=1)
ax.axvspan(7.5, 9.5, alpha=0.12, color=get_plt_colour(3), ymin=0, ymax=1)

ax.set_xlim(0.5, 9.5)
ax.set_xticks(dof)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 25))
ax.set_xlabel(r"Degrees of Freedom")
ax.set_ylabel(r"$F_1$-Score (\%)")
ax.grid(linewidth=1, alpha=0.5, zorder=0, axis="y")

ax.legend(
    [rects1_base, rects2_base, span_id, span_ood],
    [ r"Random", r"Boundary", r"In Distribution",r"Out of Distribution"],
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02)
)


plt.tight_layout()
plt.savefig("ood.pdf")
plt.show()