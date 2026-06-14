import pickle
import torch
import ram.dataset.se3 as se3

from ram.dataset.morphology import sample_morph
from ram.dataset.kinematics import inverse_kinematics, transformation_matrix
from ram.dataset.workspace import sample_poses_in_reach, fk_approximation, \
    ball_approximation
from ram.model import Model

from paper_archive.utils import display_slice
from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.model import Model as GGIKModel
from paper_archive.rq2_accuracy.adapter import network_args, forward_pose

from liegroups.numpy import SE3
from graphik.robots import RobotRevolute
from graphik.graphs.graph_revolute import ProblemGraphRevolute
from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.utils.dataset_generation import \
    generate_struct_data, generate_data_point_from_pose
from tqdm import tqdm
from pathlib import Path
from ram.logger import Logger

if __name__ == "__main__":
    torch.manual_seed(1)

    save_dir = Path(__file__).parent / "data" / "slice"
    save_dir.mkdir(parents=True, exist_ok=True)

    # morph = sample_morph(1, 6, False)[0].to("cuda")
    # poses = sample_poses_in_reach(1_000, morph)
    # label = inverse_kinematics(morph.double(), poses.double())[1] != -1
    #
    # steps = 1000
    # mat = transformation_matrix(morph[0, 0:1],
    #                             morph[0, 1:2],
    #                             morph[0, 2:3],
    #                             torch.zeros_like(morph[0, 0:1]))
    # torus_axis = torch.nn.functional.normalize(mat[:3, 2], dim=0)
    # centre, radius = ball_approximation(morph)
    # fixed_axes = torch.argmax(torus_axis.abs())
    # axes_mask = torch.ones(3, dtype=torch.bool, device=morph.device)
    # axes_mask[fixed_axes] = False
    # axes_range = torch.linspace(-radius, radius, steps).to(morph.device)
    # anchor = poses[label][torch.median(poses[label][:, :3, 3].norm(dim=1), dim=0).indices]
    # pose = anchor.unsqueeze(0).expand(steps ** 2, -1, -1).clone()
    # pose[:, :3, 3][:, axes_mask] = centre[axes_mask]
    # pose[:, :3, 3][:, axes_mask] += torch.stack(torch.meshgrid(axes_range, axes_range, indexing='ij'),
    #                                             dim=-1).reshape(-1, 2)
    #
    # label = inverse_kinematics(morph.cpu(), pose.cpu())[1] != -1
    #
    # torch.save(morph, save_dir / "morph.pth")
    # torch.save(pose, save_dir / "pose.pth")
    # torch.save(label, save_dir / "label.pth")
    morph = torch.load(save_dir / "morph.pth")
    pose = torch.load(save_dir / "pose.pth")
    label = torch.load(save_dir / "label.pth")

    # display_slice([label.cpu()], [""], morph, "slice_ground-truth.pdf")
    # #
    # model = Model.from_id(142).to("cuda")
    # label_mlp = []
    # for batch_idx in range(0, len(pose), 1000):
    #     current_pose = pose[batch_idx:batch_idx + 1000].to("cuda")
    #     bmorph = morph.to("cuda").unsqueeze(0).expand(current_pose.shape[0], -1, -1)
    #
    #     label_mlp += [model.predict(bmorph, se3.to_vector(current_pose)).cpu()]
    #
    # label_mlp = torch.cat(label_mlp, dim=0)
    # display_slice([torch.nn.Sigmoid()(label_mlp) > 0.5], [""], morph, "slice_ram.pdf")
    #
    # params = {
    #     "alpha": morph[:, 0].tolist(),
    #     "a": morph[:, 1].tolist(),
    #     "d": morph[:, 2].tolist(),
    #     "theta": [0] * morph.shape[0],
    #     "num_joints": morph.shape[0],
    #     "modified_dh": True,
    # }
    #
    # graph = ProblemGraphRevolute(RobotRevolute(params))
    # struct_data = generate_struct_data(graph)
    #
    # data = []
    # for pose in tqdm(pose):
    #     data += [generate_data_point_from_pose(graph, SE3.from_matrix(pose.cpu().numpy(), normalize=True), struct_data)]
    #
    # pickle.dump(graph, open(save_dir / "graph.pickle", "wb"))
    # pickle.dump(data, open(save_dir / "data.pickle", "wb"))
    #
    graph = pickle.load(open(save_dir / "graph.pickle", "rb"))
    data = pickle.load(open(save_dir / "data.pickle", "rb"))
    #
    device = torch.device("cuda")
    batch_size = 500
    num_samples = 32
    model = GGIKModel(network_args())
    model.load_state_dict(torch.load("/home/wtim/generative-graphik/saved_models/NRM/checkpoints/checkpoint.pth")["net"])
    model = model.to(device)
    se3_dist = []
    for batch_idx in tqdm(range(0, pose.shape[0], batch_size)):
        current_data = data[batch_idx:batch_idx + batch_size]
        current_batch_size = len(current_data)
        current_morph = morph.unsqueeze(0).expand(current_batch_size, -1, -1)
        current_pose = pose[batch_idx:batch_idx + current_batch_size]
        predicted_pose = forward_pose(model,
                                      current_data,
                                      num_samples,
                                      current_morph,
                                      current_pose,
                                      torch.zeros(current_batch_size).int(),
                                      [graph])
        se3_dist += [se3.distance(predicted_pose, current_pose).cpu()]

    se3_dist = torch.cat(se3_dist, dim=0)

    torch.save(se3_dist, save_dir / "se3_dist.pth")

    se3_dist = torch.load(save_dir / "se3_dist.pth")

    min_unreachable_distance = se3_dist[~label].min()
    max_reachable_distance = se3_dist[label].max()

    if max_reachable_distance > min_unreachable_distance:
        threshold = min_unreachable_distance
        best_f1 = 0.0
        for candidate in torch.linspace(min_unreachable_distance, max_reachable_distance, 100):
            logit = se3_dist < candidate
            f1 = Logger.compute_metrics(logit.float()[:,0], label)["F1 Score"]
            if f1 > best_f1:
                threshold = candidate
                best_f1 = f1
    else:
        threshold = (max_reachable_distance + min_unreachable_distance) / 2

    label_ggik = se3_dist < threshold
    display_slice([label_ggik], [""], morph, "slice_ggik.pdf")
