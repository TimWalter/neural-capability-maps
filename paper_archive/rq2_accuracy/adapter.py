import torch
from torch_geometric.data import Batch

import argparse
from liegroups.numpy import SE3
from graphik.utils.dgp import graph_from_pos
from paper_archive.rq2_accuracy.generative_graphik.generative_graphik.args.utils import str2bool

import ram.dataset.se3 as se3
from ram.dataset.kinematics import forward_kinematics
from ram.dataset.self_collision import collision_check
import concurrent.futures

# Ensure child processes are safely spawned for PyTorch compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set elsewhere


def network_args():
    parser = argparse.ArgumentParser()

    # Training Settings
    parser.add_argument('--n_epoch', type=int, default=360, help='Number of epochs')
    parser.add_argument('--n_scheduler_epoch', type=int, default=60,
                        help='Number of epochs before fixed scheduler steps.')
    parser.add_argument('--n_checkpoint_epoch', type=int, default=16, help='Number of epochs for checkpointing')
    parser.add_argument('--n_beta_scaling_epoch', type=int, default=1,
                        help='Warm start KL divergence for this amount of epochs.')
    parser.add_argument('--n_joint_scaling_epoch', type=int, default=1,
                        help='Warm start joint loss for this amount of epochs.')
    parser.add_argument('--n_batch', type=int, default=128, help='Batch size')
    parser.add_argument('--n_worker', type=int, default=0, help='Amount of workers for dataloading.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

    # Network parameters
    parser.add_argument('--num_anchor_nodes', type=int, default=4, help='Number of anchor nodes')
    parser.add_argument('--num_node_features_out', type=int, default=3, help='Size of node features out')
    parser.add_argument('--num_coordinates_in', type=int, default=3, help='Size of node coordinates in')
    parser.add_argument('--num_features_in', type=int, default=3, help='Size of node features in')
    parser.add_argument('--num_edge_features_in', type=int, default=1, help='Size of edge features in')
    parser.add_argument('--gnn_type', type=str, default="egnn", help='GNN type used.')
    parser.add_argument('--num_gnn_layers', type=int, default=5, help='Number of GNN layers')
    parser.add_argument('--num_graph_mlp_layers', type=int, default=2,
                        help='Number of layers for the MLPs used in the graph')
    parser.add_argument('--num_egnn_mlp_layers', type=int, default=2,
                        help='Number of layers for the MLPs used in the EGNN layer itself')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of iterations to networks go through')
    parser.add_argument('--dim_latent', type=int, default=64, help='Size of latent node features in to encoder')
    parser.add_argument('--dim_goal', type=int, default=6, help='Size of goal representation (SE3-->6, SE2-->3)')
    parser.add_argument('--num_prior_mixture_components', type=int, default=16,
                        help='Number of mixture components for prior network')
    parser.add_argument('--num_likelihood_mixture_components', type=int, default=1,
                        help='Number of mixture components for likelihood network')
    parser.add_argument('--train_prior', type=str2bool, default=True,
                        help='Learn prior parameters conditionned on variables.')
    parser.add_argument('--rec_gain', type=int, default=10, help='Gain on non-anchor node reconstruction')
    parser.add_argument('--non_linearity', type=str, default="silu", help='Non-linearity used.')
    parser.add_argument('--dim_latent_node_out', type=int, default=16, help='Size of node feature dim in enc/dec')
    parser.add_argument('--graph_mlp_hidden_size', type=int, default=128,
                        help='Size of hiddden layers of MLP used in GNN')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Size of all other MLP hiddden layers')
    parser.add_argument('--norm_layer',
                        choices=['None', 'BatchNorm', 'LayerNorm', 'GroupNorm', 'InstanceNorm', 'GraphNorm'],
                        default='LayerNorm', help='Layer normalization method.')

    args = parser.parse_args(args=[])
    return args

def _process_single_graph(output_slice, current_graph, current_pose, dofp1, num_samples, num_nodes):
    """Worker function to process joints for a single robot graph on CPU."""
    T_final = {f"p{dofp1}": SE3.from_matrix(current_pose, normalize=True)}
    joint_list = []
    idx = 0
    for s in range(num_samples):
        current_output = output_slice[s, :num_nodes]
        g_pos = graph_from_pos(current_output, current_graph.node_ids)
        joint_list += [torch.tensor(list(current_graph.joint_variables(g_pos, T_final=T_final).values()))]
    return torch.stack(joint_list, dim=0)


def forward_pose(model, data, num_samples, morph, pose, morph_idx, graph):
    batch_size = morph.shape[0]
    data = [model.preprocess(d) for d in data]
    batch = Batch.from_data_list(data).to(morph.device)
    total_nodes_in_batch = batch.num_nodes
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
    )#.reshape(num_samples, batch_size, nodes_per_robot, 3)
    ###
    output_cpu = output.cpu()
    pose_cpu = pose.cpu().numpy()

    tasks = []
    idx = 0
    for b in range(batch_size):
        current_graph = graph[morph_idx[b]]
        current_pose = pose_cpu[b]
        dofp1 = (morph[b].abs().sum(dim=1) != 0).sum().item()

        # Slice output for current graph
        output_slice = output_cpu[:, idx:idx+data[b].num_nodes]
        idx += data[b].num_nodes

        tasks.append((output_slice, current_graph, current_pose, dofp1, num_samples, data[b].num_nodes))

    # Multi-process execution over available CPU cores
    batch_joints = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(_process_single_graph, *t) for t in tasks]
        for f in futures:
            batch_joints.append(f.result())

    # Fast Vectorized GPU operations for Kinematics & Collisions
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
    ###
    # pose_list = []
    # idx = 0
    # for b in range(batch_size):
    #     current_graph = graph[morph_idx[b]]
    #     current_pose = pose[b]
    #     dofp1 = (morph[b].abs().sum(dim=1) != 0).sum().item()
    #     current_morph = morph[b][:dofp1].unsqueeze(0).expand(num_samples, -1, -1)
    #     T_final = {f"p{dofp1}": SE3.from_matrix(current_pose.cpu().numpy(), normalize=True)}
    #     joint_list = []
    #     for s in range(num_samples):
    #         current_output = output[s, idx:idx+data[b].num_nodes]
    #         g_pos = graph_from_pos(current_output.cpu(), current_graph.node_ids)
    #         joint_list += [torch.tensor(list(current_graph.joint_variables(g_pos, T_final=T_final).values()))]
    #
    #     idx += data[b].num_nodes
    #     joints = torch.stack(joint_list, dim=0).unsqueeze(-1)
    #     joints = torch.cat([joints[:, 1:], torch.zeros_like(joints[:, 0:1])], dim=1).float().to(morph.device)
    #     joints = joints.reshape(num_samples, dofp1, 1)
    #     predicted_pose = forward_kinematics(current_morph, joints)
    #     collision = collision_check(current_morph, predicted_pose)
    #     distance = se3.distance(predicted_pose[:, -1], current_pose.unsqueeze(0).expand(num_samples, -1, -1))
    #     distance[collision] = torch.inf
    #     min_idx = distance.argmin(dim=0)
    #     pose_list += predicted_pose[min_idx, -1]

    return torch.stack(pose_list)
