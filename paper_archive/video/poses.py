import os
import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, Gf, Vt

import newton
import warp as wp

# Import your native kinematics functions
from ram.dataset.kinematics import forward_kinematics, inverse_kinematics, transformation_matrix
from ram.dataset.morphology import sample_morph
from ram.dataset.self_collision import LINK_RADIUS
import ram.dataset.se3 as se3
import ram.dataset.r3 as r3
from ram.dataset.workspace import ball_approximation

torch.manual_seed(1)

# ============================== config ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path(__file__).parent / "usd_exports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Sample Morphology and Poses
morph = sample_morph(1, 6, False, DEVICE)[0] # [6, 3]
n_links = morph.shape[0]
poses = se3.random(100).to(DEVICE)           # [100, 4, 4]

# Compute Filters
filter1 = poses[:, :3, 3].norm(dim=1) < 1.0

centre, radius = ball_approximation(morph[:-1])
radius = max(0.0, radius - r3.DISTANCE_BETWEEN_CELLS)

eef_transformation = transformation_matrix(
    morph[-1, 0:1], morph[-1, 1:2], morph[-1, 2:3], torch.zeros_like(morph[-1, 0:1])
).to(DEVICE)
inv_eef = torch.linalg.inv(eef_transformation)
poses_without_eef = poses @ inv_eef
filter2 = (poses[:, :3, 3] - centre).norm(dim=1) < radius

# Compute Upward Joint Angles for Animation
extended = torch.eye(4).unsqueeze(0).to(DEVICE)
extended[0, :3, 3] = torch.tensor([10.0, 0.0, 0.0], device=DEVICE)
extended_joints, _ = inverse_kinematics(morph, extended)

BASE_Z = 1.5
COLOR_JOINT = (0.2, 0.2, 0.2)
COLOR_BASE = (0.1, 0.4, 0.8)
COLOR_LINKS = (0.8, 0.8, 0.8)
JOINT_RADIUS = 1.02 * LINK_RADIUS

FPS = 60
TOTAL_TIME = 2.0
N_FRAMES = int(TOTAL_TIME * FPS)


# ============================== helpers ==============================
def attach_rgb_frame(builder, body, length=0.15, radius=0.015):
    """Attaches an RGB coordinate system to a given body link."""
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
        Gf.Vec3f(-s, -s, 0.0), Gf.Vec3f(s, -s, 0.0),
        Gf.Vec3f(s, s, 0.0), Gf.Vec3f(-s, s, 0.0),
    ]))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
        Gf.Vec3f(-s, -s, 0.0), Gf.Vec3f(s, s, 0.0),
    ]))
    stage.Save()


# ============================== wireframe injectors ==============================

def add_wireframe_cube(usd_path, center, side_length=1.0, width_frac=0.012):
    """Bakes a 3D wireframe box directly into the USD stage."""
    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    cube_path = default.GetPath().AppendChild("WireframeCube")

    curves = UsdGeom.BasisCurves.Define(stage, cube_path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)

    h = side_length / 2.0
    cx, cy, cz = center
    pts = [
        Gf.Vec3f(cx-h, cy-h, cz-h), Gf.Vec3f(cx+h, cy-h, cz-h),
        Gf.Vec3f(cx+h, cy+h, cz-h), Gf.Vec3f(cx-h, cy+h, cz-h),
        Gf.Vec3f(cx-h, cy-h, cz+h), Gf.Vec3f(cx+h, cy-h, cz+h),
        Gf.Vec3f(cx+h, cy+h, cz+h), Gf.Vec3f(cx-h, cy+h, cz+h),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),   # lower base
        (4,5),(5,6),(6,7),(7,4),   # upper base
        (0,4),(1,5),(2,6),(3,7),   # pillars
    ]
    curve_pts = []
    for e in edges:
        curve_pts.append(pts[e[0]])
        curve_pts.append(pts[e[1]])

    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray([2] * len(edges)))
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(curve_pts))

    # THE FIX: thin per-vertex widths -> lines, not fat default-width tubes.
    w = side_length * width_frac
    curves.CreateWidthsAttr().Set(Vt.FloatArray([w] * len(curve_pts)))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.6, 0.0)]))  # orange
    curves.CreateDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))                    # fully opaque
    stage.Save()


def add_wireframe_sphere(usd_path, center, radius=1.0, width_frac=0.012):
    """Generates a latitude/longitude wireframe grid for a sphere."""
    stage = Usd.Stage.Open(str(usd_path))
    default = stage.GetDefaultPrim()
    sphere_path = default.GetPath().AppendChild("WireframeSphere")

    curves = UsdGeom.BasisCurves.Define(stage, sphere_path)
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)

    cx, cy, cz = center
    curve_pts = []
    vertex_counts = []

    # Latitude rings
    num_lat = 12
    num_pts_per_ring = 32
    for i in range(1, num_lat):
        lat = -math.pi / 2.0 + (math.pi * i / num_lat)
        z_offset = radius * math.sin(lat)
        r_ring = radius * math.cos(lat)
        ring_pts = []
        for j in range(num_pts_per_ring + 1):
            lon = 2.0 * math.pi * j / num_pts_per_ring
            ring_pts.append(Gf.Vec3f(cx + r_ring * math.cos(lon),
                                     cy + r_ring * math.sin(lon),
                                     cz + z_offset))
        curve_pts.extend(ring_pts)
        vertex_counts.append(len(ring_pts))

    # Longitude arcs
    num_lon = 12
    num_pts_per_arc = 32
    for i in range(num_lon):
        lon = 2.0 * math.pi * i / num_lon  # full coverage
        arc_pts = []
        for j in range(num_pts_per_arc + 1):
            lat = -math.pi / 2.0 + (math.pi * j / num_pts_per_arc)
            arc_pts.append(Gf.Vec3f(cx + radius * math.cos(lat) * math.cos(lon),
                                    cy + radius * math.cos(lat) * math.sin(lon),
                                    cz + radius * math.sin(lat)))
        curve_pts.extend(arc_pts)
        vertex_counts.append(len(arc_pts))

    curves.CreateCurveVertexCountsAttr().Set(Vt.IntArray(vertex_counts))
    curves.CreatePointsAttr().Set(Vt.Vec3fArray(curve_pts))

    # Same fix applied here.
    w = radius * width_frac
    curves.CreateWidthsAttr().Set(Vt.FloatArray([w] * len(curve_pts)))
    curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

    curves.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.6, 0.0)]))  # orange
    curves.CreateDisplayOpacityAttr().Set(Vt.FloatArray([1.0]))
    stage.Save()


def joint_q_flat(theta):
    return theta[..., 0].reshape(-1).cpu().numpy().astype(np.float32)


# ============================== CORE USD GENERATOR ==============================
def generate_pose_scene(filename, active_poses, animate_robot=False):
    output_path = OUTPUT_DIR / filename
    builder = newton.ModelBuilder()
    off = torch.tensor([0.0, 0.0, BASE_Z], device=DEVICE)

    # 1. Add Filtered Static Task Poses
    for w in range(active_poses.shape[0]):
        task_T = active_poses[w].cpu().numpy()
        pos = task_T[:3, 3] + off.cpu().numpy()
        quat = Rotation.from_matrix(task_T[:3, :3]).as_quat()

        task_link = builder.add_link(mass=0.0, inertia=None, xform=wp.transform(p=pos, q=quat))
        attach_rgb_frame(builder, task_link, length=0.15, radius=0.015)

    # 2. Build Robot Tree
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

        # Link Geometry setup
        if abs(d_j) > 1e-4:
            builder.add_shape_capsule(
                body=links[-1], radius=LINK_RADIUS, half_height=abs(d_j) / 2,
                xform=wp.transform(p=torch.tensor([0.0, 0.0, -d_j / 2], device=DEVICE), q=Rotation.identity().as_quat()),
                color=color)

        if abs(a_j) > 1e-4:
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

    builder.add_articulation(joints, label="robot")
    model = builder.finalize(device="cpu")
    state = model.state()

    # 3. Save Frames to USD Stage
    viewer = newton.viewer.ViewerUSD(output_path=str(output_path), fps=FPS, up_axis="Z")
    viewer.set_model(model)

    frames_to_render = N_FRAMES if animate_robot else 1

    for f in range(frames_to_render):
        t = f / FPS

        if animate_robot:
            alpha = f / (N_FRAMES - 1)
            alpha = alpha * alpha * (3 - 2 * alpha)
            current_theta = (1.0 - alpha) * torch.zeros_like(extended_joints) + alpha * extended_joints
        else:
            current_theta = torch.zeros(1, n_links, 1, device=DEVICE)

        state.joint_q.assign(joint_q_flat(current_theta))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.end_frame()

    viewer.close()
    add_lab_floor(output_path)


# ============================== EXECUTE EXPORTS ==============================

# USD 1: Static - All Poses + Unit Cube Wireframe
generate_pose_scene("1_all_poses.usd", poses, animate_robot=False)
add_wireframe_cube(OUTPUT_DIR / "1_all_poses.usd", center=[0.0, 0.0, BASE_Z], side_length=2.0)
print(f"Generated: 1_all_poses.usd with Wireframe Cube")

# USD 2: Static - Filter 1 Poses + Unit Sphere Wireframe
generate_pose_scene("2_filter_1_poses.usd", poses[filter1], animate_robot=False)
add_wireframe_sphere(OUTPUT_DIR / "2_filter_1_poses.usd", center=[0.0, 0.0, BASE_Z], radius=1.0)
print(f"Generated: 2_filter_1_poses.usd with Wireframe Sphere")

# USD 3: Dynamic - Filter 2 Poses + Robot Extending Upward
generate_pose_scene("3_filter_2_animated.usd", poses[filter2], animate_robot=True)
print(f"Generated: 3_filter_2_animated.usd")