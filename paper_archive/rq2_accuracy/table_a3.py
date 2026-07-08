import wandb

# --- CONFIGURATION ---
WANDB_ENTITY = "tim-walter-tum"
WANDB_PROJECT = "RAM"

# Map your Run IDs / Hashes to the specific cells in the table
# Replace these placeholder hashes with your actual W&B run IDs
run_mapping = {
    "RAM": {
        r"$\left\{5,6,7\right\}$": "htzuql9e",
        "9": "p7vnypho",
        "8": "dw8vfmwd",
        "7": "6txg6jf6",
        "6": "1saij5m2",
        "5": "zl7aqn4z",
        "4": "aon9aux9",
        "3": "s8j6971k",
        "2": "zid8a8ch",
        "1": "dcpxw4zi",
    }
}

# ---------------------

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

    metric_names = ['True Positives', 'False Negatives', 'False Positives', 'True Negatives']
    formatted_values = []

    for name in metric_names:
        mean_key = f"{prefix}/{name} (Mean)"
        low_key = f"{prefix}/{name} (CI Lower)"
        high_key = f"{prefix}/{name} (CI Upper)"

        # Fallback check for nested tracking prefixes (e.g. Boundary/Validation/...)
        if mean_key not in summary and "Boundary" in prefix:
            mean_key = f"Boundary/Validation/{name} (Mean)"
            low_key = f"Boundary/Validation/{name} (CI Lower)"
            high_key = f"Boundary/Validation/{name} (CI Upper)"

        if mean_key in summary:
            mean = summary[mean_key]
            # Check if CI limits exist (macro-average branch)
            if low_key in summary and high_key in summary:
                low = summary[low_key]
                high = summary[high_key]
                # Matches your \num{mean(low:high)} LaTeX formatting
                formatted_values.append(f"\\num{{{mean:.0f}({high-mean:.0f}:{mean-low:.0f})}}")
            else:
                # Fallback if no CIs exist (unsegmented / micro-average metrics)
                formatted_values.append(f"\\num{{{mean:.0f}}}")
        else:
            formatted_values.append("??")

    return formatted_values


# Generate the dynamic LaTeX content
latex_rows = []

for classifier in ["RAM"]:
    for idx, dof in enumerate([r"$\left\{5,6,7\right\}$", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
        run_id = run_mapping[classifier].get(dof)

        # Omit classifier label on subsequent rows for clean tabularray rendering
        cls_label = classifier if idx == 0 else ""

        # Fetch Validation (Random) space and Boundary space metrics
        random_metrics = fetch_metrics(run_id, prefix="Validation")
        boundary_metrics = fetch_metrics(run_id, prefix="Boundary")

        # Unpack metrics
        r_tp, r_fn, r_fp, r_tn = random_metrics
        b_tp, b_fn, b_fp, b_tn = boundary_metrics

        row = (
            f"        {cls_label:<10} & {dof:<4} "
            f"& {r_tp} & {r_fn} & {r_fp} & {r_tn} "
            f"& {b_tp} & {b_fn} & {b_fp} & {b_tn} \\\\"
        )
        latex_rows.append(row)

    # Append midrule separator between classifiers
    if classifier == "RAM":
        latex_rows.append("        \\midrule")

# Print the complete populated table code block
print("\n=== GENERATED LATEX TABLE ===\n")
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
                & & TP & FN & FP & TN & TP & FN & FP & TN\\\\
        
        \\midrule""")
for r in latex_rows:
    print(r)
print("""        \\bottomrule
    \\end{tblr}
\\end{table}""")