from ram.dataset.kinematics import forward_kinematics
from ram.dataset.morphology import sample_morph

import torch

import newton
import warp as wp
from ram.dataset.self_collision import LINK_RADIUS
from scipy.spatial.transform import Rotation

import numpy as np
from tqdm import tqdm

torch.manual_seed(0)
morph = sample_morph(100, 6, False,device=torch.device("cuda"))
poses = forward_kinematics(morph, torch.zeros(morph.shape[0], 7, 1, device=morph.device))

COLOR_JOINT = (0.2, 0.2, 0.2)  # Dark Grey for motors
COLOR_BASE = (0.1, 0.4, 0.8)  # Blue for base
COLOR_LINKS = (0.8, 0.8, 0.8)  # Light Grey

JOINT_RADIUS = 1.02 * LINK_RADIUS
SPACING = 1.1

N_robots = morph.shape[0]
grid_size = int(np.ceil(np.sqrt(N_robots)))

builder = newton.ModelBuilder()
builder.add_ground_plane(height=-1.0)

for i in tqdm(range(N_robots)):
    row, col = divmod(i, grid_size)
    grid_offset = torch.tensor([col * SPACING, row * SPACING, 0.0], device=morph.device)

    joints = []
    links = []
    for j in range(morph[i].shape[0]):
        current_color = COLOR_BASE if i == j else COLOR_LINKS

        # Link are currently placed at the end! (as we use pose[j]) we have to go backwards
        links += [builder.add_link(mass=0.0, inertia=None,
                                   xform=wp.transform(p=poses[i, j, :3, 3] + grid_offset,
                                                      q=Rotation.from_matrix(poses[i, j, :3, :3]).as_quat()))]
        # d capsule
        new_z_dir = torch.tensor([0.0, -torch.sin(morph[i,j, 0]), torch.cos(morph[i,j, 0])])
        builder.add_shape_capsule(body=links[-1], radius=LINK_RADIUS, half_height=morph[i,j, 2].abs().item() / 2,
                                  xform=wp.transform(
                                      p=torch.tensor([0.0, 0.0, -morph[i,j, 2].item() / 2]),
                                      q=Rotation.identity().as_quat()
                                  ),
                                  color=current_color)

        # a capsule, along x-axis
        builder.add_shape_capsule(body=links[-1], radius=LINK_RADIUS, half_height=morph[i,j, 1].abs().item() / 2,
                                  xform=wp.transform(
                                      p=torch.tensor([-morph[i,j, 1].item() / 2, 0.0, -morph[i,j, 2].item()]),
                                      q=Rotation.from_euler('y', 90, degrees=True).as_quat()
                                  ),
                                  color=current_color)

        builder.add_shape_sphere(
            body=links[-1],
            radius=JOINT_RADIUS,
            xform=wp.transform_identity(),
            color=COLOR_JOINT
        )

        if j != 0:
            pose_rel = torch.linalg.inv(poses[i, j - 1]) @ poses[i,j]
            joints += [builder.add_joint_fixed(parent=links[j - 1], child=links[j],
                                               parent_xform=wp.transform(p=pose_rel[:3, 3], q=Rotation.from_matrix(
                                                   pose_rel[:3, :3]).as_quat()),
                                               child_xform=wp.transform_identity()
                                               )]

    builder.add_articulation(joints, label=f"robot_{i}")

model = builder.finalize(device="cpu")
state = model.state()

viewer = newton.viewer.ViewerUSD(output_path="army.usd", fps=60, up_axis="Z")
viewer.set_model(model)
# at every frame:
viewer.begin_frame(0.0)
viewer.log_state(state)
viewer.end_frame()

viewer.close()