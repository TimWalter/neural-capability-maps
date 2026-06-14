import torch
from pathlib import Path

from ram.dataset.morphology import sample_morph
import ram.dataset.se3 as se3

from paper_archive.rq3_design_optimisation.ours import ours


if __name__ == "__main__":
    torch.manual_seed(1)

    save_dir = Path(__file__).parent / "morphology_optimisation"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")


    task = se3.random_ball(5, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])).to(device)
    initial_morph = sample_morph(1, 6, False, device)[0]

    _, _, _, _, pose_error, self_collision, morph = ours(initial_morph, task, 100)

    print(pose_error)
    print(self_collision)
    torch.save(task, save_dir / "task.pt")
    torch.save(morph, save_dir / "morph.pt")