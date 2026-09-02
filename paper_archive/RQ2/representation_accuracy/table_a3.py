import pickle
from pathlib import Path

from paper_archive.utils import latex_mean_and_ci

import wandb

WANDB_ENTITY = "tim-walter-tum"
WANDB_PROJECT = "RAM"

ALL_DOF = r"$\left\{5,6,7\right\}$"

run_mapping = {
    "RAM": {
        ALL_DOF: "ept1avaf",
        "9": "vspsc5vv",
        "8": "d1i4miix",
        "7": "xsmaa5g2",
        "6": "ws8sdkir",
        "5": "qnl3flqf",
        "4": "j641xs0l",
        "3": "s28fgzrp",
        "2": "anvh3qhs",
        "1": "ud93z44u",
    }
}

GGIK_CACHE = Path(__file__).parent / "cache" / "ggik"
ggik_mapping = {
    ALL_DOF: "All",
    "7": "7",
    "6": "6",
    "5": "5",
}

METRIC_NAMES = ['True Positive Rate', 'False Negative Rate', 'False Positive Rate', 'True Negative Rate']

api = wandb.Api()


def fetch_metrics(run_id, prefix):
    """Fetches metrics for a given prefix path from the run summary."""
    if not run_id:
        return ["??"] * 4

    try:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        summary = run.summary
    except Exception as e:
        print(f"% Error fetching run {run_id}: {e}")
        return ["??"] * 4

    formatted_values = []

    for name in METRIC_NAMES:
        mean_key = f"{prefix}/{name} (Mean)"
        low_key = f"{prefix}/{name} (CI Lower)"
        high_key = f"{prefix}/{name} (CI Upper)"

        if mean_key not in summary and "Boundary" in prefix:
            mean_key = f"Boundary/Validation/{name} (Mean)"
            low_key = f"Boundary/Validation/{name} (CI Lower)"
            high_key = f"Boundary/Validation/{name} (CI Upper)"

        formatted_values.append(latex_mean_and_ci(summary[mean_key], summary.get(low_key), summary.get(high_key), decimals=0))

    return formatted_values


def load_ggik_results():
    """Loads the cached GGIK metrics written by ggik_validate.py, keyed by split and DoF."""
    results = {}
    for prefix, path in [("Validation", "test"), ("Boundary", "test_boundary")]:
        file = GGIK_CACHE / path / "results.pickle"
        try:
            with open(file, "rb") as handle:
                results[prefix] = pickle.load(handle)
        except FileNotFoundError:
            print(f"% Missing {file}, run ggik_validate.py")
            results[prefix] = {}
    return results


def fetch_ggik_metrics(results, dof, prefix):
    """Fetches metrics for a given DoF split from the cached GGIK results."""
    split = results.get(prefix, {}).get(ggik_mapping[dof])
    if split is None:
        return ["??"] * 4

    return [latex_mean_and_ci(split[f"{name} (Mean)"], split[f"{name} (CI Lower)"], split[f"{name} (CI Upper)"], decimals=0)
            for name in METRIC_NAMES]


latex_rows = []

for classifier in ["RAM"]:
    for idx, dof in enumerate([ALL_DOF, "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
        run_id = run_mapping[classifier].get(dof)

        cls_label = classifier if idx == 0 else ""

        random_metrics = fetch_metrics(run_id, prefix="Validation")
        boundary_metrics = fetch_metrics(run_id, prefix="Boundary")

        r_tp, r_fn, r_fp, r_tn = random_metrics
        b_tp, b_fn, b_fp, b_tn = boundary_metrics

        row = (
            f"        {cls_label:<10} & {dof:<4} "
            f"& {r_tp} & {r_fn} & {r_fp} & {r_tn} "
            f"& {b_tp} & {b_fn} & {b_fp} & {b_tn} \\\\"
        )
        latex_rows.append(row)

    if classifier == "RAM":
        latex_rows.append("        \\midrule")

ggik_results = load_ggik_results()
for idx, dof in enumerate([ALL_DOF, "7", "6", "5"]):
    cls_label = "GGIK" if idx == 0 else ""

    r_tp, r_fn, r_fp, r_tn = fetch_ggik_metrics(ggik_results, dof, prefix="Validation")
    b_tp, b_fn, b_fp, b_tn = fetch_ggik_metrics(ggik_results, dof, prefix="Boundary")

    latex_rows.append(
        f"        {cls_label:<10} & {dof:<4} "
        f"& {r_tp} & {r_fn} & {r_fp} & {r_tn} "
        f"& {b_tp} & {b_fn} & {b_fp} & {b_tn} \\\\"
    )

print(f"""\\begin{{table}}[H]
    \\centering
    \\caption{{Comparison of RAM and GGIK. We denote the binary confusion matrix with True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN).}}
    \\label{{tab:rq2_binary_confusion}}
    \\begin{{tblr}}{{
            colspec = {{l r r r r r r r r r}},
            row{{1,2}} = {{font=\\bfseries}}, 
            cell{{1}}{{1}} = {{r=2}}{{l}}, 
            cell{{1}}{{2}} = {{r=2}}{{r}}, 
            cell{{1}}{{3}} = {{c=4}}{{c}}, 
            cell{{1}}{{7}} = {{c=4}}{{c}}, 
        }}
        \\toprule
        Classifier & DoF & Random (\\%) & & & & Boundary (\\%) & & & \\\\
                \\cmidrule[lr]{{3-6}}
                \\cmidrule[lr]{{7-10}}
                & & TPR & FNR & FPR & TNR & TPR & FNR & FPR & TNR\\
        \\midrule""")
for r in latex_rows:
    print(r)
print("""        \\bottomrule
    \\end{tblr}
\\end{table}""")
