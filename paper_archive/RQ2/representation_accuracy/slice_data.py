import pickle
from pathlib import Path

import torch

from ram.dataset.morphology import sample_morph
from ram.dataset.kinematics import inverse_kinematics, transformation_matrix
from ram.dataset.workspace import sample_poses_in_reach, ball_approximation
from paper_archive.utils import display_slice


from liegroups.numpy import SE3
from graphik.robots import RobotRevolute
from graphik.graphs.graph_revolute import ProblemGraphRevolute
from paper_archive.RQ2.representation_accuracy.generative_graphik.generative_graphik.utils.dataset_generation import \
    generate_struct_data, generate_data_point_from_pose

if __name__ == "__main__":
    torch.manual_seed(1)

    save_dir = Path(__file__).parent / "cache" / "slice"
    save_dir.mkdir(parents=True, exist_ok=True)

    morph = sample_morph(1, 6, False)[0].to("cuda")
    poses = sample_poses_in_reach(1_000, morph)
    label = inverse_kinematics(morph.double(), poses.double())[1] != -1

    steps = 1000
    mat = transformation_matrix(morph[0, 0:1],
                                morph[0, 1:2],
                                morph[0, 2:3],
                                torch.zeros_like(morph[0, 0:1]))
    torus_axis = torch.nn.functional.normalize(mat[:3, 2], dim=0)
    centre, radius = ball_approximation(morph)
    fixed_axes = torch.argmax(torus_axis.abs())
    axes_mask = torch.ones(3, dtype=torch.bool, device=morph.device)
    axes_mask[fixed_axes] = False
    axes_range = torch.linspace(-radius, radius, steps).to(morph.device)
    anchor = poses[label][torch.median(poses[label][:, :3, 3].norm(dim=1), dim=0).indices]
    pose = anchor.unsqueeze(0).expand(steps ** 2, -1, -1).clone()
    pose[:, :3, 3][:, axes_mask] = centre[axes_mask]
    pose[:, :3, 3][:, axes_mask] += torch.stack(torch.meshgrid(axes_range, axes_range, indexing='ij'),
                                                dim=-1).reshape(-1, 2)

    label = inverse_kinematics(morph.cpu(), pose.cpu())[1] != -1

    torch.save(morph, save_dir / "morph.pth")
    torch.save(pose, save_dir / "pose.pth")
    torch.save(label, save_dir / "label.pth")

    display_slice([label.cpu()], [""], morph, save_dir / "slice_ground-truth.pdf")

    # For GGIK
    params = {
        "alpha": morph[:, 0].tolist(),
        "a": morph[:, 1].tolist(),
        "d": morph[:, 2].tolist(),
        "theta": [0] * morph.shape[0],
        "num_joints": morph.shape[0],
        "modified_dh": True,
    }

    graph = ProblemGraphRevolute(RobotRevolute(params))
    struct_data = generate_struct_data(graph)

    data = []
    for pose in pose:
        data += [generate_data_point_from_pose(graph, SE3.from_matrix(pose.cpu().numpy(), normalize=True), struct_data)]

    pickle.dump(graph, open(save_dir / "graph.pickle", "wb"))
    pickle.dump(data, open(save_dir / "data.pickle", "wb"))
