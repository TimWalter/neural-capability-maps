import time
import pickle
from pathlib import Path

import torch

from tqdm import tqdm
from torch_geometric.data import Batch

import ram.dataset.se3 as se3
from ram.dataset.loader import HomogeneousPoseSet

from paper_archive.RQ2.representation_accuracy.ggik_adapter import GGIK

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 1000
    num_samples = 32

    path = "test"

    model = GGIK.from_pretrained("TimWalter/GGIK").to(device)

    directory = Path(__file__).parent / "cache" / "ggik" / path

    graph = pickle.load(open(directory / "graphs.pickle", "rb"))
    data = pickle.load(open(directory / "data.pickle", "rb"))
    eval_set = HomogeneousPoseSet(batch_size, False, path, device=device)

    full_inference = []
    for batch_idx, (morph, pose, label, morph_idx) in enumerate(tqdm(eval_set, desc=f"Validation")):
        pose = se3.from_vector(pose)
        start = time.perf_counter()
        predicted_pose = model.forward_pose(data[batch_idx*batch_size: (batch_idx+1)*batch_size], num_samples,
                                      morph, pose, morph_idx, graph)
        full_inference += [time.perf_counter() - start]

    data = pickle.load(open(directory / "data.pickle", "rb"))
    data = [model.preprocess(d) for d in data[batch_idx*batch_size: (batch_idx+1)*batch_size]]
    batch = Batch.from_data_list(data).to(device)
    total_nodes_in_batch = batch.num_nodes
    nodes_per_robot = data[0].num_nodes
    only_query = []
    for _ in range(100):
        start = time.perf_counter()
        output = model.forward_eval(
            x=batch.pos,
            h=torch.cat((batch.type, batch.goal_data_repeated_per_node), dim=-1),
            edge_attr=batch.edge_attr,
            edge_attr_partial=batch.edge_attr_partial,
            edge_index=batch.edge_index_full,
            partial_goal_mask=batch.partial_goal_mask,
            nodes_per_single_graph=total_nodes_in_batch,
            batch_size=1,  # Treated as 1 large graph for the sample expansion logic
            num_samples=num_samples
        )
        only_query += [time.perf_counter() - start]

    only_query = sum(only_query) / len(only_query) / batch_size * 10**6
    full_inference = sum(full_inference) / len(full_inference) / batch_size * 10**6

    print(only_query)
    print(full_inference)