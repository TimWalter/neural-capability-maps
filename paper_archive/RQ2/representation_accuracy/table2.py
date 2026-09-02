import pickle
from pathlib import Path

from ram.metrics import bootstrap_mean_ci
from paper_archive.utils import latex_mean_and_ci

import wandb
import torch

ggik = pickle.load(open(Path(__file__).parent / "cache" / "ggik" /"test" / "results.pickle", "rb"))["All"]["Balanced Accuracy (Mean)"]

inference_time = pickle.load(open(Path(__file__).parent / "runtime_ours.pkl", "rb"))

api = wandb.Api()
runs = api.runs(
    "tim-walter-tum/RAM",
    filters={"group": "RAM"}
)

training_time_list = []
bacc_list = []
for run in runs:
    bacc_list += [run.summary.get("Validation/Balanced Accuracy (Mean)")]
    training_time_list += [run.summary.get("_runtime")/3600]

training_time = sum(training_time_list) / len(training_time_list)

bacc = latex_mean_and_ci(*bootstrap_mean_ci(torch.tensor(bacc_list).unsqueeze(1)))

print(rf"""
\begin{{table}}[H]
    \begin{{talltblr}}[
        caption = {{Comparison of RAM and GGIK. Inference is from pose and morphology input to reachability output.}},
        label = {{tab:ram_accuracy}},
    ]{{
        colspec = {{l r r r}},
        row{{1}} = []{{font=\bfseries}}, 
    }}
        \toprule
        Classifier & {{Training  (h)}} & {{Inference (s)}} & \SetCell{{l}}{{Balanced (\%)\\ Accuracy }}  \\
        \midrule
        RAM  & \textbf{{{training_time}}} & $\bm{{{inference_time}\cdot10^ {{-6}}}}$ & {{\bfseries{bacc}}} \\
        GGIK & 756 & $2.2\cdot10^{{-2}}$ & {ggik} \\
        \bottomrule
    \end{{talltblr}}
\end{{table}}
""")