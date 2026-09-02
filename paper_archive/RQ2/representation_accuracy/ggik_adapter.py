import multiprocessing
import concurrent.futures
from collections import namedtuple

import torch
from torch_geometric.data import Batch
from huggingface_hub import PyTorchModelHubMixin

from liegroups.numpy import SE3
from graphik.utils.dgp import graph_from_pos

import ram.dataset.se3 as se3
from ram.dataset.kinematics import forward_kinematics
from ram.dataset.self_collision import collision_check
from paper_archive.RQ2.representation_accuracy.generative_graphik.generative_graphik.model import Model

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set elsewhere


class GGIK(Model,
           PyTorchModelHubMixin,
           repo_url="https://huggingface.co/TimWalter/GGIK",
           paper_url="https://huggingface.co/papers/2606.09108",
           license="mit"):
    def __init__(self):
        kwargs = {
            "n_epoch": 360,
            "n_scheduler_epoch": 60,
            "n_checkpoint_epoch": 16,
            "n_beta_scaling_epoch": 1,
            "n_joint_scaling_epoch": 1,
            "n_batch": 128,
            "n_worker": 0,
            "lr": 3e-4,
            "num_anchor_nodes": 4,
            "num_node_features_out": 3,
            "num_coordinates_in": 3,
            "num_features_in": 3,
            "num_edge_features_in": 1,
            "gnn_type": "egnn",
            "num_gnn_layers": 5,
            "num_graph_mlp_layers": 2,
            "num_egnn_mlp_layers": 2,
            "num_iterations": 1,
            "dim_latent": 64,
            "dim_goal": 6,
            "num_prior_mixture_components": 16,
            "num_likelihood_mixture_components": 1,
            "train_prior": True,
            "rec_gain": 10,
            "non_linearity": "silu",
            "dim_latent_node_out": 16,
            "graph_mlp_hidden_size": 128,
            "mlp_hidden_size": 128,
            "norm_layer": "LayerNorm"
        }
        Model.__init__(self, namedtuple("temp", kwargs.keys())(**kwargs))

    def forward_pose(self, data, num_samples, morph, pose, morph_idx, graph):
        batch_size = morph.shape[0]
        data = [self.preprocess(d) for d in data]
        batch = Batch.from_data_list(data).to(morph.device)
        total_nodes_in_batch = batch.num_nodes
        output = self.forward_eval(
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

        output_cpu = output.cpu()
        pose_cpu = pose.cpu().numpy()

        tasks = []
        idx = 0
        for b in range(batch_size):
            current_graph = graph[morph_idx[b]]
            current_pose = pose_cpu[b]
            dofp1 = (morph[b].abs().sum(dim=1) != 0).sum().item()

            output_slice = output_cpu[:, idx:idx + data[b].num_nodes]
            idx += data[b].num_nodes

            tasks.append((output_slice, current_graph, current_pose, dofp1, num_samples, data[b].num_nodes))

        # Multi-process execution over available CPU cores
        batch_joints = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(_process_single_graph, *t) for t in tasks]
            for f in futures:
                batch_joints.append(f.result())

        pose_list = []
        for b in range(batch_size):
            dofp1 = (morph[b].abs().sum(dim=1) != 0).sum().item()
            current_morph = morph[b][:dofp1].unsqueeze(0).expand(num_samples, -1, -1)
            current_pose = pose[b]

            joints = batch_joints[b].unsqueeze(-1)
            joints = torch.cat([joints[:, 1:], torch.zeros_like(joints[:, 0:1])], dim=1).float().to(morph.device)
            joints = joints.reshape(num_samples, dofp1, 1)

            predicted_pose = forward_kinematics(current_morph, joints)
            collision = collision_check(current_morph, predicted_pose)
            distance = se3.distance(predicted_pose[:, -1], current_pose.unsqueeze(0).expand(num_samples, -1, -1))
            distance[collision] = torch.inf
            min_idx = distance.argmin(dim=0)
            pose_list += predicted_pose[min_idx, -1]

        return torch.stack(pose_list)


def _process_single_graph(output_slice, current_graph, current_pose, dofp1, num_samples, num_nodes):
    """Worker function to process joints for a single robot graph on CPU."""
    T_final = {f"p{dofp1}": SE3.from_matrix(current_pose, normalize=True)}
    joint_list = []
    for s in range(num_samples):
        current_output = output_slice[s, :num_nodes]
        g_pos = graph_from_pos(current_output, current_graph.node_ids)
        joint_list += [torch.tensor(list(current_graph.joint_variables(g_pos, T_final=T_final).values()))]
    return torch.stack(joint_list, dim=0)
