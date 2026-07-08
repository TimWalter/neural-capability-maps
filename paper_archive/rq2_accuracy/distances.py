import torch


import ram.dataset.se3 as se3
from tqdm import tqdm
import pickle
from ram.dataset.loader import ValidationSet
from pathlib import Path

from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.model import Model
from paper_archive.rq2_accuracy.adapter import network_args, forward_pose

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda")
    batch_size = 1000
    num_samples = 32

    for path in ["test", "test_boundary"]:

        model = Model(network_args())
        model.load_state_dict(torch.load("/home/wtim/generative-graphik/saved_models/NRM/checkpoints/checkpoint.pth", map_location=device)["net"])
        model = model.to(device)

        directory = Path(__file__).parent / "data" / path
        graph = pickle.load(open(directory / "graphs.pickle", "rb"))
        data = pickle.load(open(directory / "data.pickle", "rb"))
        eval_set = ValidationSet(batch_size, False, path)

        se3_dist = []

        morph_indices = torch.load(directory / "morph_indices.pth")
        pose_buffer = torch.load(directory / "poses.pth")

        for start in tqdm(range(0, len(data), 500)):
            end = start + 500

            morph = eval_set.morphologies[morph_indices[start:end]]
            pose = pose_buffer[start:end]

            morph = morph.to(device, non_blocking=True)
            pose = se3.from_vector(pose.to(device, non_blocking=True))

            predicted_pose = forward_pose(model,
                                          data[start:end],
                                          num_samples,
                                          morph,
                                          pose,
                                          morph_indices[start:end],
                                          graph)

            se3_dist += [se3.distance(predicted_pose, pose).cpu()]

        torch.save(torch.cat(se3_dist, dim=0), Path(__file__).parent / "data" / path / "se3_dist.pth")
