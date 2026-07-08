import torch


import ram.dataset.se3 as se3
from tqdm import tqdm
import pickle
from ram.dataset.loader import ValidationSet
from pathlib import Path
from datetime import datetime
from torch_geometric.data import Batch

from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.model import Model
from paper_archive.rq2_accuracy.adapter import network_args, forward_pose

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch_size = 1000
    num_samples = 32

    path = "test_boundary"

    model = Model(network_args())
    model.load_state_dict(torch.load("/home/wtim/generative-graphik/saved_models/NRM/checkpoints/checkpoint.pth", map_location="cuda")["net"])
    model = model.to(device)

    only_query = []
    full_inference = []
    for _ in range(10):
        graph = pickle.load(open(Path(__file__).parent / "data" / path / "graphs.pickle", "rb"))
        data = pickle.load(open(Path(__file__).parent / "data" / path / "data.pickle", "rb"))
        eval_set = ValidationSet(batch_size, False, path)

        for batch_idx, (morph, pose, label) in enumerate(tqdm(eval_set, desc=f"Validation")):
            morph = morph.to(device, non_blocking=True)
            pose = se3.from_vector(pose.to(device, non_blocking=True))
            morph_idx = eval_set._get_batch(batch_idx)[:, 0].long()
            start = datetime.now()
            predicted_pose = forward_pose(model, data[batch_idx*batch_size: (batch_idx+1)*batch_size], num_samples,
                                          morph, pose, morph_idx, graph)
            full_inference += [datetime.now() - start]

        batch_size, dofp1 = morph.shape[:2]
        data = pickle.load(open(Path(__file__).parent / "data" / path / "data.pickle", "rb"))
        data = [model.preprocess(d) for d in data[batch_idx*batch_size: (batch_idx+1)*batch_size]]
        batch = Batch.from_data_list(data).to(morph.device)
        total_nodes_in_batch = batch.num_nodes
        nodes_per_robot = data[0].num_nodes
        start = datetime.now()
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
        only_query += [datetime.now() - start]

    only_query = torch.tensor([fk.seconds + fk.microseconds* 10**(-6) for fk in only_query])
    full_inference = torch.tensor([fk.seconds + fk.microseconds* 10**(-6) for fk in full_inference])

    print(only_query.mean() / batch_size)
    print(full_inference.mean() / batch_size)