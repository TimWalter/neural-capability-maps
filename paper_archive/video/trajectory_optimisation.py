import torch
from pathlib import Path

from ram.dataset.morphology import sample_morph, get_joint_limits
from ram.dataset.workspace import sample_workspace, sample_poses_in_reach
from ram.dataset.kinematics import inverse_kinematics

import ram.dataset.se3 as se3
from paper_archive.rq3_trajectory_optimisation.ours import ours


if __name__ == "__main__":

    torch.manual_seed(1)

    save_dir = Path(__file__).parent / "trajectory_optimisation"
    save_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 10
    device = torch.device("cuda")

    morph = sample_morph(1, 6, False, device)[0]
    poses = sample_poses_in_reach(1000, morph)


    start = poses[0]
    end = poses[1]

    tangent = se3.log(start, end)
    t = torch.linspace(0, 1, num_samples, device=tangent.device).view(-1, 1)
    target_trajectory = se3.exp(start.repeat(num_samples, 1, 1), t * tangent)

    _, _, deviations, _, pose_error, self_collision, trajectory = ours(morph, target_trajectory, 100)

    print(pose_error[0], pose_error[50], pose_error[99])
    print(deviations[0]*2, deviations[50]*2, deviations[99]*2)
    print(self_collision[0], self_collision[50], self_collision[99])
    torch.save(morph, save_dir / "morph.pt")
    torch.save(trajectory, save_dir / "trajectory.pt")
