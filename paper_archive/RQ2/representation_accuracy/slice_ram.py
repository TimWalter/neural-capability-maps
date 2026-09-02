from pathlib import Path

import torch

import ram.dataset.se3 as se3
from ram.model import Model

from paper_archive.utils import display_slice

if __name__ == "__main__":
    save_dir = Path(__file__).parent / "cache" / "slice"
    morph = torch.load(save_dir / "morph.pth")
    pose = torch.load(save_dir / "pose.pth")
    label = torch.load(save_dir / "label.pth")

    model = Model.from_id(142).to("cuda")
    label_mlp = []
    for batch_idx in range(0, len(pose), 1000):
        current_pose = pose[batch_idx:batch_idx + 1000].to("cuda")
        bmorph = morph.to("cuda").unsqueeze(0).expand(current_pose.shape[0], -1, -1)

        label_mlp += [model.predict(bmorph, se3.to_vector(current_pose)).cpu()]

    label_mlp = torch.cat(label_mlp, dim=0)
    display_slice([torch.sigmoid(label_mlp) > 0.5], [""], morph, save_dir / "slice_ram.pdf")