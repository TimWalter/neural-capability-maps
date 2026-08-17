import torch

from torch import Tensor
from jaxtyping import Float, Bool, jaxtyped
from beartype import beartype

import ram.dataset.se3 as se3
from ram.dataset.morphology import get_joint_limits
from ram.dataset.kinematics import inverse_kinematics
from ram.dataset.workspace import sample_workspace, sample_poses_in_reach


@jaxtyped(typechecker=beartype)
def get_boundary_pairs(morph: Float[Tensor, "dofp1 3"], num_pairs: int, oversampling: int = 10) \
        -> tuple[
            Float[Tensor, "num_pairs 4 4"],
            Float[Tensor, "num_pairs 4 4"]
        ]:
    """
    Generate pairs of reachable and unreachable poses given a morphology.

    Args:
        morph: MDH parameters encoding the robot geometry.
        num_pairs: Number of pairs to generate.
        oversampling: Factor to oversample with.

    Returns:
        Reachable poses and unreachable poses.
    """

    joint_limits = get_joint_limits(morph)

    reachable_pose = sample_workspace(morph.unsqueeze(0).expand(oversampling * num_pairs, -1, -1),
                                      joint_limits.unsqueeze(0).expand(oversampling * num_pairs, -1, -1))[0][:num_pairs]
    for i in range(10):
        new = sample_workspace(morph.unsqueeze(0).expand(oversampling * (num_pairs - reachable_pose.shape[0]), -1, -1),
                               joint_limits.unsqueeze(0).expand(oversampling * (num_pairs - reachable_pose.shape[0]),
                                                                -1, -1)
                               )[0][:num_pairs - reachable_pose.shape[0]]
        reachable_pose = torch.cat([reachable_pose, new])
        if reachable_pose.shape[0] == num_pairs:
            break
    if reachable_pose.shape[0] != num_pairs:
        if reachable_pose.shape[0] > 0:
            repeats = (num_pairs + reachable_pose.shape[0] - 1) // reachable_pose.shape[0]
            reachable_pose = reachable_pose.repeat(repeats, 1, 1)[:num_pairs]
        else:
            raise RuntimeError(
                f"Failed to sample reachable poses. Found only {reachable_pose.shape[0]} instead of {num_pairs}")

    unreachable_pose = sample_poses_in_reach(oversampling * num_pairs, morph)
    unreachable_pose = unreachable_pose[inverse_kinematics(morph, unreachable_pose)[-1] == -1][:num_pairs]
    for i in range(10):
        new = sample_poses_in_reach(oversampling * (num_pairs - unreachable_pose.shape[0]), morph)
        new = new[inverse_kinematics(morph, new)[-1] == -1][:(num_pairs - unreachable_pose.shape[0])]
        unreachable_pose = torch.cat([unreachable_pose, new])
        if unreachable_pose.shape[0] == num_pairs:
            break
    if unreachable_pose.shape[0] != num_pairs:
        if unreachable_pose.shape[0] > 0:
            repeats = (num_pairs + unreachable_pose.shape[0] - 1) // unreachable_pose.shape[0]
            unreachable_pose = unreachable_pose.repeat(repeats, 1, 1)[:num_pairs]
        else:
            raise RuntimeError(
                f"Failed to sample unreachable poses. Found only {unreachable_pose.shape[0]} instead of {num_pairs}")

    return reachable_pose, unreachable_pose


@jaxtyped(typechecker=beartype)
def sample_boundary(morph: Float[Tensor, "dofp1 3"], num_geodesics: int, num_samples: int) \
        -> tuple[
            Float[Tensor, "{num_geodesics*num_samples} 4 4"],
            Bool[Tensor, "{num_geodesics*num_samples}"]
        ]:
    """
    Given a morphology, compute boundary points by following geodesics across the boundary.

    Args:
        morph: MDH parameters encoding the robot geometry.
        num_geodesics: Number of geodesics to sample.
        num_samples: Number of samples per geodesic.

    Returns:
        Poses, labels
    """
    reachable_pose, unreachable_pose = get_boundary_pairs(morph, num_geodesics)

    tangent = se3.log(reachable_pose, unreachable_pose).unsqueeze(0)
    t = torch.linspace(0, 1, num_samples).view(-1, 1, 1).to(tangent.device)

    poses = se3.exp(reachable_pose.repeat(num_samples, 1, 1, 1), t * tangent).view(-1, 4, 4)
    labels = inverse_kinematics(morph, poses)[1] != -1

    return poses, labels
