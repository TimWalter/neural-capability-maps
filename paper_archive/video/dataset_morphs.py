from ram.dataset.kinematics import forward_kinematics
from ram.dataset.morphology import sample_morph

import torch

import newton
import warp as wp
from ram.dataset.self_collision import LINK_RADIUS
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdShade, Gf, Vt, Sdf


DEVICE = torch.device("cuda")
torch.manual_seed(0)
morph = torch.cat([
    torch.cat([sample_morph(250, 5, False,device=torch.device("cuda")), torch.zeros(250, 2, 3, device="cuda")],dim=1),
    torch.cat([sample_morph(500, 6, False,device=torch.device("cuda")), torch.zeros(500, 1, 3, device="cuda")],dim=1),
    sample_morph(250, 7, False,device=torch.device("cuda"))
])
poses_home = forward_kinematics(morph, torch.zeros(len(morph), morph.shape[1], 1, device=DEVICE))   # [N, n_links, 4, 4]

COLOR_JOINT = (0.2, 0.2, 0.2)  # Dark Grey for motors
COLOR_BASE = (0.1, 0.4, 0.8)  # Blue for base
COLOR_LINKS = (0.8, 0.8, 0.8)  # Light Grey

JOINT_RADIUS = 1.02 * LINK_RADIUS
SPACING, BASE_Z = 2.0, 1.5

N_robots = morph.shape[0]

rows = torch.arange(100, device=DEVICE).repeat_interleave(10)
cols = torch.arange(10, device=DEVICE).repeat(100)

grid_offsets = torch.stack([(cols * SPACING).float(),
                            (rows * SPACING).float(),
                            torch.full((N_robots,), BASE_Z, device=DEVICE)], dim=1)   # [N, 3]


builder = newton.ModelBuilder()

builder = newton.ModelBuilder()
#builder.add_ground_plane(height=0.5)

for i in range(len(morph)):
    off = poses_home.new_zeros(3); off[:] = grid_offsets[i]
    joints, links = [], []

    for j in range(morph.shape[1]):
        color = COLOR_BASE if j == 0 else COLOR_LINKS
        a_j, d_j = morph[i, j, 1].item(), morph[i, j, 2].item()

        links.append(builder.add_link(
            mass=0.0, inertia=None,
            xform=wp.transform(p=poses_home[i, j, :3, 3] + off,
                               q=Rotation.from_matrix(poses_home[i, j, :3, :3]).as_quat())))

        if j == 0:
            parent = -1
            parent_T = poses_home[i, 0].clone()
            parent_T[:3, 3] = parent_T[:3, 3] + off
        else:
            parent = links[j - 1]
            parent_T = torch.linalg.inv(poses_home[i, j - 1]) @ poses_home[i, j]

        builder.add_shape_capsule(
            body=links[-1], radius=LINK_RADIUS, half_height=abs(d_j) / 2,
            xform=wp.transform(p=torch.tensor([0.0, 0.0, -d_j / 2]),
                               q=Rotation.identity().as_quat()),
            color=color)

        X_cap = torch.eye(4, device=DEVICE, dtype=poses_home.dtype)
        X_cap[:3, 3]  = torch.tensor([-a_j / 2, 0.0, -d_j], device=DEVICE, dtype=poses_home.dtype)
        X_cap[:3, :3] = torch.tensor(Rotation.from_euler('y', 90, degrees=True).as_matrix(),
                                     device=DEVICE, dtype=poses_home.dtype)
        X_cap = parent_T @ X_cap
        builder.add_shape_capsule(
            body=parent, radius=LINK_RADIUS, half_height=abs(a_j) / 2,
            xform=wp.transform(p=X_cap[:3, 3], q=Rotation.from_matrix(X_cap[:3, :3]).as_quat()),
            color=color)

        builder.add_shape_sphere(body=links[-1], radius=JOINT_RADIUS,
                                 xform=wp.transform_identity(), color=COLOR_JOINT)

        joints.append(builder.add_joint_revolute(
            parent=parent, child=links[j], axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=parent_T[:3, 3],
                                      q=Rotation.from_matrix(parent_T[:3, :3]).as_quat()),
            child_xform=wp.transform_identity()))

    builder.add_articulation(joints, label=f"robot_{i}")

model = builder.finalize(device="cpu")
state = model.state()

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

viewer = newton.viewer.ViewerUSD(output_path="morphs.usd", fps=60, up_axis="Z")
viewer.set_model(model)
# at every frame:
viewer.begin_frame(0.0)
viewer.log_state(state)
viewer.end_frame()

viewer.close()

stage   = Usd.Stage.Open("morphs.usd")
default = stage.GetDefaultPrim()
assert default and default.IsValid(), f"{"morphs.usd"} has no default prim"
base = default.GetPath()
add_lab_floor(stage, base)
stage.Save()