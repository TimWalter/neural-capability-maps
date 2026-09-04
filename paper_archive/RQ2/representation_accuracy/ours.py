import time
import pickle
from pathlib import Path

import torch

from ram.model import Model
from ram.validate import validate
from ram.dataset.loader import HomogeneousPoseSet

if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 1000

    cache = Path(__file__).parent / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    model_ids = list(range(2065, 2075))
    for model_id in model_ids:
        validate(model_id, batch_size, None, "test", "RAM")

    # Runtime
    validation_set = HomogeneousPoseSet(batch_size, False, "test", device)
    model = Model.from_id(model_ids[-1]).to(device)

    model.eval()
    runtime = []
    for batch_idx, (morph, pose, _, _) in enumerate(validation_set):
        start = time.perf_counter_ns()
        logit = model.predict(morph, pose)
        runtime += [time.perf_counter_ns() - start]

    runtime = sum(runtime) / len(runtime) / batch_size
    print(runtime)

    with open(cache / "runtime_ours.pkl", "wb") as file:
        pickle.dump(runtime, file)