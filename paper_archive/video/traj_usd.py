import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, Gf, Vt

import newton
import warp as wp

from ram.dataset.kinematics import forward_kinematics, inverse_kinematics
from ram.dataset.self_collision import LINK_RADIUS

torch.manual_seed(1)

# ============================== config ==============================
DEVICE = torch.device("cuda")

# Load the generated task and morphologies
LOAD_DIR = Path(__file__).parent / "trajectory_optimisation"
morph = torch.load(LOAD_DIR / "morph.pt", map_location=DEVICE)
trajectory = torch.load(LOAD_DIR / "trajectory.pt", map_location=DEVICE)

OUTPUT_DIR = Path(__file__).parent / "usd_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAJECTORIES = trajectory.shape[0]
n_links = morph.shape[0]
N_WAYPOINTS = trajectory.shape[1]
BASE_Z = 1.5

COLOR_JOINT = (0.2, 0.2, 0.2)
COLOR_BASE = (0.1, 0.4, 0.8)
COLOR_LINKS = (0.8, 0.8, 0.8)
JOINT_RADIUS = 1.02 * LINK_RADIUS

FPS = 60
TOTAL_TIME = 20.0
N_FRAMES = int(TOTAL_TIME * FPS)
DWELL = 0.4

move_time = (TOTAL_TIME - N_WAYPOINTS * DWELL) / (N_WAYPOINTS - 1)
wp_period = DWELL + move_time


# ============================== helper: rgb frames ==============================
def attach_rgb_frame(builder, body, length=0.05, radius=0.005):
    """Attaches a fat RGB coordinate system to a given body link."""
    qx = Rotation.from_euler('y', 90, degrees=True).as_quat()
    px = np.array([length / 2, 0.0, 0.0])
    builder.add_shape_capsule(body=body, radius=radius, half_height=length / 2,
                              xform=wp.transform(p=px, q=qx), color=(1.0, 0.0, 0.0))

    qy = Rotation.from_euler('x', -90, degrees=True).as_quat()
    py = np.array([0.0, length / 2, 0.0])
    builder.add_shape_capsule(body=body, radius=radius, half_height=length / 2,
                              xform=wp.transform(p=py, q=qy), color=(0.0, 1.0, 0.0))

    qz = Rotation.identity().as_quat()
    pz = np.array([0.0, 0.0, length / 2])
    builder.add_shape_capsule(body=body, radius=radius, half_height=length / 2,
                              xform=wp.transform(p=pz, q=qz), color=(0.0, 0.0, 1.0))


# ============================== helper: lab floor ==============================
def add_lab_floor(usd_path, size=200.0):
    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    base = default.GetPath()

    floor_path = base.AppendChild("Floor")
    mesh = UsdGeom.Mesh.Define(stage, floor_path)
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    s = size
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-s, -s, 0.5), Gf.Vec3f(s, -s, 0.5),
        Gf.Vec3f(s, s, 0.5), Gf.Vec3f(-s, s, 0.5),
    ]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-s, -s, 0.5), Gf.Vec3f(s, s, 0.5),
    ]))
    stage.Save()


# ============================== MAIN LOOP ==============================

for i in tqdm(range(N_TRAJECTORIES), desc="Generating individual USDs"):
    if i != 0 and i != N_TRAJECTORIES - 1 and i!=N_TRAJECTORIES//2:
        continue

    output_path = OUTPUT_DIR / f"trajectory_{i:03d}.usd"

    # 1. Compute Native IK for this specific robot
    waypoints, manips = inverse_kinematics(morph, trajectory[i])  # [N_WAYPOINTS, dofp1, 1]

    # 2. Build Newton Model for this specific robot
    builder = newton.ModelBuilder()

    # Centered offset (no grid)
    off = torch.tensor([0.0, 0.0, BASE_Z], device=DEVICE)

    # Draw Static Task Poses (RGB Frames)
    for w in range(N_WAYPOINTS):
        task_T = trajectory[i, w].cpu().numpy()
        pos = task_T[:3, 3] + off.cpu().numpy()
        quat = Rotation.from_matrix(task_T[:3, :3]).as_quat()

        task_link = builder.add_link(mass=0.0, inertia=None, xform=wp.transform(p=pos, q=quat))
        attach_rgb_frame(builder, task_link, length=0.15, radius=0.015)

    # Robot Kinematics Tree
    # Forward kinematics for home position (single robot batch)
    pose_home_single = forward_kinematics(morph.unsqueeze(0), torch.zeros(1, n_links, 1, device=DEVICE))[0]

    joints, links = [], []
    for j in range(n_links):
        color = COLOR_BASE if j == 0 else COLOR_LINKS
        a_j, d_j = morph[j, 1].item(), morph[j, 2].item()

        links.append(builder.add_link(
            mass=0.0, inertia=None,
            xform=wp.transform(p=pose_home_single[j, :3, 3] + off,
                               q=Rotation.from_matrix(pose_home_single[j, :3, :3].cpu().numpy()).as_quat())))

        if j == 0:
            parent = -1
            parent_T = pose_home_single[0].clone()
            parent_T[:3, 3] = parent_T[:3, 3] + off
        else:
            parent = links[j - 1]
            parent_T = torch.linalg.inv(pose_home_single[j - 1]) @ pose_home_single[j]

        # Link Geometry
        builder.add_shape_capsule(
            body=links[-1], radius=LINK_RADIUS, half_height=abs(d_j) / 2,
            xform=wp.transform(p=torch.tensor([0.0, 0.0, -d_j / 2]), q=Rotation.identity().as_quat()),
            color=color)

        X_cap = torch.eye(4, device=DEVICE, dtype=pose_home_single.dtype)
        X_cap[:3, 3] = torch.tensor([-a_j / 2, 0.0, -d_j], device=DEVICE, dtype=pose_home_single.dtype)
        X_cap[:3, :3] = torch.tensor(Rotation.from_euler('y', 90, degrees=True).as_matrix(),
                                     device=DEVICE, dtype=pose_home_single.dtype)
        X_cap = parent_T @ X_cap

        builder.add_shape_capsule(
            body=parent, radius=LINK_RADIUS, half_height=abs(a_j) / 2,
            xform=wp.transform(p=X_cap[:3, 3], q=Rotation.from_matrix(X_cap[:3, :3].cpu().numpy()).as_quat()),
            color=color)

        builder.add_shape_sphere(body=links[-1], radius=JOINT_RADIUS, xform=wp.transform_identity(), color=COLOR_JOINT)

        joints.append(builder.add_joint_revolute(
            parent=parent, child=links[j], axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=parent_T[:3, 3],
                                      q=Rotation.from_matrix(parent_T[:3, :3].cpu().numpy()).as_quat()),
            child_xform=wp.transform_identity()))

    # Attach RGB Frame to End Effector
    attach_rgb_frame(builder, links[-1], length=0.15, radius=0.015)
    builder.add_articulation(joints, label=f"robot_{i}")

    model = builder.finalize(device="cpu")
    state = model.state()


    # 3. Define Trajectory Interpolator for this robot
    def q_at(t):
        t = min(t, TOTAL_TIME)
        w = min(int(t // wp_period), N_WAYPOINTS - 1)
        local = t - w * wp_period

        if w == N_WAYPOINTS - 1 or local <= DWELL:
            return waypoints[w]

        # Smoothstep timing for soft start/stop
        a = (local - DWELL) / move_time
        a = a * a * (3 - 2 * a)

        q1 = waypoints[w]
        q2 = waypoints[w + 1]

        # Calculate the shortest angular path in configuration space [-pi, pi]
        delta_q = torch.remainder(q2 - q1 + torch.pi, 2 * torch.pi) - torch.pi

        # Interpolate along the shortest path
        q_interp = q1 + a * delta_q

        # Normalize the result back to [-pi, pi] to keep values safely bounded
        q_interp = torch.remainder(q_interp + torch.pi, 2 * torch.pi) - torch.pi

        return q_interp


    def joint_q_flat(theta):
        return theta[..., 0].reshape(-1).cpu().numpy().astype(np.float32)


    # 4. Render to USD
    viewer = newton.viewer.ViewerUSD(output_path=str(output_path), fps=FPS, up_axis="Z")
    viewer.set_model(model)

    for f in range(N_FRAMES):
        t = f / FPS
        current_theta = q_at(t)

        state.joint_q.assign(joint_q_flat(current_theta))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()

    viewer.close()

    # 5. Add Lab Floor
    add_lab_floor(output_path)

print(f"\nSuccessfully generated {N_TRAJECTORIES} individual USD files in '{OUTPUT_DIR}'.")
