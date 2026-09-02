import pickle
from pathlib import Path

import torch
from tqdm import tqdm


import ram.dataset.se3 as se3

from paper_archive.utils import display_slice
from paper_archive.RQ2.representation_accuracy.ggik_adapter import GGIK

from ram.validate import best_confidence_threshold


if __name__ == "__main__":
    save_dir = Path(__file__).parent / "cache" / "slice"

    morph = torch.load(save_dir / "morph.pth")
    pose = torch.load(save_dir / "pose.pth")
    label = torch.load(save_dir / "label.pth")
    graph = pickle.load(open(save_dir / "graph.pickle", "rb"))
    data = pickle.load(open(save_dir / "data.pickle", "rb"))

    device = torch.device("cuda")
    batch_size = 500
    num_samples = 32
    model = GGIK.from_pretrained("TimWalter/GGIK").to(device)
    se3_dist = []
    for batch_idx in tqdm(range(0, pose.shape[0], batch_size)):
        current_data = data[batch_idx:batch_idx + batch_size]
        current_batch_size = len(current_data)
        current_morph = morph.unsqueeze(0).expand(current_batch_size, -1, -1)
        current_pose = pose[batch_idx:batch_idx + current_batch_size]
        predicted_pose = model.forward_pose(current_data,
                                            num_samples,
                                            current_morph,
                                            current_pose,
                                            torch.zeros(current_batch_size).int(),
                                            [graph])
        se3_dist += [se3.distance(predicted_pose, current_pose).cpu()]

    se3_dist = torch.cat(se3_dist, dim=0)

    logit = -se3_dist

    threshold = best_confidence_threshold(logit[:, 0], label, torch.zeros_like(label).long())

    label_ggik = torch.sigmoid(logit) > threshold
    display_slice([label_ggik], [""], morph, save_dir / "slice_ggik.pdf")
