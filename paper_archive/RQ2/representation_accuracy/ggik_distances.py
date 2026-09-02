import pickle
from pathlib import Path

import torch
from tqdm import tqdm

import ram.dataset.se3 as se3
from ram.dataset.loader import HomogeneousPoseSet

from paper_archive.RQ2.representation_accuracy.ggik_adapter import GGIK

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 1000
    num_samples = 32

    for path in ["test", "test_boundary"]:
        model = GGIK.from_pretrained("TimWalter/GGIK").to(device)

        directory = Path(__file__).parent / "cache" / "ggik" / path

        graph = pickle.load(open(directory / "graphs.pickle", "rb"))
        data = pickle.load(open(directory / "data.pickle", "rb"))
        eval_set = HomogeneousPoseSet(batch_size, False, path, device=device)

        se3_dist = []

        morph_indices = torch.load(directory / "morph_indices.pth")
        pose_buffer = torch.load(directory / "poses.pth")

        for start in tqdm(range(0, len(data), 500)):
            end = start + 500

            morph = eval_set.morphologies[morph_indices[start:end]]
            pose = pose_buffer[start:end]

            morph = morph.to(device, non_blocking=True)
            pose = se3.from_vector(pose.to(device, non_blocking=True))

            predicted_pose = model.forward_pose(data[start:end],
                                                num_samples,
                                                morph,
                                                pose,
                                                morph_indices[start:end],
                                                graph)

            se3_dist += [se3.distance(predicted_pose, pose).cpu()]

        torch.save(torch.cat(se3_dist, dim=0), directory / "distances.pth")
