import os
import math
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, Gf, Vt, Sdf

import newton
import warp as wp

from ram.dataset.kinematics import forward_kinematics
from ram.dataset.morphology import sample_morph, get_joint_limits
from ram.dataset.self_collision import LINK_RADIUS, collision_check

torch.manual_seed(1)

# ============================== config ==============================
DEVICE = torch.device("cuda")
N_ROBOTS, DOF = 1, 6  # Single robot for dashboard split-screen focus
SPACING, BASE_Z = 4.0, 1.5

COLOR_JOINT = (0.2, 0.2, 0.2)
COLOR_BASE = (0.1, 0.4, 0.8)
COLOR_LINKS = (0.8, 0.8, 0.8)
JOINT_RADIUS = 1.02 * LINK_RADIUS

FPS = 60
TOTAL_TIME = 30.0
N_FRAMES = int(TOTAL_TIME * FPS)
N_WAYPOINTS = 15
DWELL = 0.4  # seconds the robot pauses at each waypoint (keep in sync with dataset_plotly.py)
OUTPUT = "dataset.usd"

# ============================== setup ==============================
morph = sample_morph(N_ROBOTS, DOF, False, device=DEVICE)
joint_limits = get_joint_limits(morph)
n_links = morph.shape[1]

# Setup offsets
grid_size = int(np.ceil(np.sqrt(N_ROBOTS)))
idx = torch.arange(N_ROBOTS, device=DEVICE)
rows, cols = torch.div(idx, grid_size, rounding_mode='floor'), idx % grid_size
grid_offsets = torch.stack([(cols * SPACING).float(),
                            (rows * SPACING).float(),
                            torch.full((N_ROBOTS,), BASE_Z, device=DEVICE)], dim=1)

poses_home = forward_kinematics(morph, torch.zeros(N_ROBOTS, n_links, 1, device=DEVICE))

# ============================== robot model ==============================
builder = newton.ModelBuilder()
builder.add_ground_plane(height=-1.0)

for i in tqdm(range(N_ROBOTS), desc="Building Kinematic Tree"):
    off = grid_offsets[i]
    joints, links = [], []

    for j in range(n_links):
        color = COLOR_BASE if j == 0 else COLOR_LINKS
        a_j, d_j = morph[i, j, 1].item(), morph[i, j, 2].item()

        links.append(builder.add_link(
            mass=0.0, inertia=None,
            xform=wp.transform(p=poses_home[i, j, :3, 3] + off,
                               q=Rotation.from_matrix(poses_home[i, j, :3, :3].cpu().numpy()).as_quat())))

        if j == 0:
            parent = -1
            parent_T = poses_home[i, 0].clone()
            parent_T[:3, 3] = parent_T[:3, 3] + off
        else:
            parent = links[j - 1]
            parent_T = torch.linalg.inv(poses_home[i, j - 1]) @ poses_home[i, j]

        builder.add_shape_capsule(
            body=links[-1], radius=LINK_RADIUS, half_height=abs(d_j) / 2,
            xform=wp.transform(p=torch.tensor([0.0, 0.0, -d_j / 2]), q=Rotation.identity().as_quat()),
            color=color)

        X_cap = torch.eye(4, device=DEVICE, dtype=poses_home.dtype)
        X_cap[:3, 3] = torch.tensor([-a_j / 2, 0.0, -d_j], device=DEVICE, dtype=poses_home.dtype)
        X_cap[:3, :3] = torch.tensor(Rotation.from_euler('y', 90, degrees=True).as_matrix(),
                                     device=DEVICE, dtype=poses_home.dtype)
        X_cap = parent_T @ X_cap

        builder.add_shape_capsule(
            body=parent, radius=LINK_RADIUS, half_height=abs(a_j) / 2,
            xform=wp.transform(p=X_cap[:3, 3], q=Rotation.from_matrix(X_cap[:3, :3].cpu().numpy()).as_quat()),
            color=color)

        builder.add_shape_sphere(body=links[-1], radius=JOINT_RADIUS, xform=wp.transform_identity(), color=COLOR_JOINT)

        joints.append(builder.add_joint_revolute(
            parent=parent, child=links[j], axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=parent_T[:3, 3], q=Rotation.from_matrix(parent_T[:3, :3].cpu().numpy()).as_quat()),
            child_xform=wp.transform_identity()))

    builder.add_articulation(joints, label=f"robot_{i}")

model = builder.finalize(device="cpu")
state = model.state()
assert state.body_q.shape[0] == N_ROBOTS * n_links

# ============================== trajectory ==============================
def sample_joint_waypoints(joint_limits, n_wp):
    shape = (n_wp, *joint_limits.shape[:-1], 1)
    return torch.rand(shape, device=joint_limits.device) * joint_limits[..., 0:1] + joint_limits[..., 1:2]

waypoints = sample_joint_waypoints(joint_limits, N_WAYPOINTS)

# --- Dwell schedule: hold at each waypoint, smoothstep between them ---
move_time = (TOTAL_TIME - N_WAYPOINTS * DWELL) / (N_WAYPOINTS - 1)
assert move_time > 0, "DWELL too large for TOTAL_TIME / N_WAYPOINTS"
wp_period = DWELL + move_time          # arrival-to-arrival spacing in seconds

# --- Per-waypoint EEF poses + self-collision flags + arrival frames (for dataset_plotly.py) ---
# The plotly dashboard ticks once per waypoint, so it needs the exact pose held at
# each waypoint, whether that configuration self-collides, and the frame the robot
# arrives there (so the two clips stay in sync given the dwell).
wp_eef, wp_coll = [], []
for w in range(N_WAYPOINTS):
    fk_w = forward_kinematics(morph, waypoints[w])      # [N_ROBOTS, n_links, 4, 4]
    wp_eef.append(fk_w[:, -1].cpu())                    # [N_ROBOTS, 4, 4]
    wp_coll.append(collision_check(morph, fk_w).cpu())  # [N_ROBOTS] (bool)
wp_arrival = torch.tensor([round(w * wp_period * FPS) for w in range(N_WAYPOINTS)],
                          dtype=torch.long)
torch.save(torch.stack(wp_eef), "eef_waypoints.pt")          # [N_WAYPOINTS, N_ROBOTS, 4, 4]
torch.save(torch.stack(wp_coll), "collision_waypoints.pt")   # [N_WAYPOINTS, N_ROBOTS]
torch.save(wp_arrival, "wp_arrival_frames.pt")               # [N_WAYPOINTS]

def q_at(t):
    t = min(t, TOTAL_TIME)
    w = min(int(t // wp_period), N_WAYPOINTS - 1)
    local = t - w * wp_period
    if w == N_WAYPOINTS - 1 or local <= DWELL:   # paused at the waypoint
        return waypoints[w]
    a = (local - DWELL) / move_time              # moving toward the next one
    a = a * a * (3 - 2 * a)                       # smoothstep
    return (1 - a) * waypoints[w] + a * waypoints[w + 1]

def joint_q_flat(theta):
    return theta[..., 0].reshape(-1).cpu().numpy().astype(np.float32)

# ============================== render & extract ==============================
viewer = newton.viewer.ViewerUSD(output_path=OUTPUT, fps=FPS, up_axis="Z")
viewer.set_model(model)

eef_trajectory = []
eef_collisions = []

for f in tqdm(range(N_FRAMES), desc="Simulating Frames"):
    t = f / FPS
    current_theta = q_at(t)

    # USD render loop
    state.joint_q.assign(joint_q_flat(current_theta))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    viewer.begin_frame(t)
    viewer.log_state(state)
    viewer.end_frame()

    # Extract exact EEF poses every 2nd frame (as requested)
    if t % 2 == 0:
        fk_matrices = forward_kinematics(morph, current_theta)
        current_eef_poses = fk_matrices[:, -1, :, :].cpu()
        eef_trajectory.append(current_eef_poses)
        collision = collision_check(morph, fk_matrices)
        eef_collisions.append(collision)

viewer.close()

# ============================== Lab Floor Modification ==============================
def add_lab_floor(stage, base, size=200.0):
    floor_path = base.AppendChild("Floor")
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    s = size
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-s, -s, 0.5),
        Gf.Vec3f( s, -s, 0.5),
        Gf.Vec3f( s,  s, 0.5),
        Gf.Vec3f(-s,  s, 0.5),
    ]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-s, -s, 0.5),
        Gf.Vec3f( s,  s, 0.5),
    ]))
    texcoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    texcoords.Set(Vt.Vec2fArray([
        Gf.Vec2f(0.0,   0.0),
        Gf.Vec2f(500.0, 0.0),
        Gf.Vec2f(500.0, 500.0),
        Gf.Vec2f(0.0,   500.0),
    ]))

stage = Usd.Stage.Open(OUTPUT)
default = stage.GetDefaultPrim()
assert default and default.IsValid(), f"'{OUTPUT}' has no default prim"
base = default.GetPath()
add_lab_floor(stage, base)
stage.Save()

# ============================== Data Export ==============================
eef_tensor = torch.stack(eef_trajectory, dim=0)
output_filename = "eef_poses_trajectory.pt"
torch.save(eef_tensor, output_filename)
eef_collision_tensor = torch.stack(eef_collisions, dim=0)
torch.save(eef_collision_tensor, "eef_collisions.pt")

print(f"\nSuccessfully generated '{OUTPUT}'.")
print(f"Saved end-effector poses to '{output_filename}'.")
print(f"Tensor shape: {list(eef_tensor.shape)} -> [Extracted Frames, Robots, 4, 4]")