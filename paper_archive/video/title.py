import os
os.environ.setdefault("SCIPY_ARRAY_API", "1")          # required by ram.dataset.so3 at import

import math
import numpy as np                                      # only at Newton / USD(Vt) boundaries
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import isoext
from pxr import Usd, UsdGeom, UsdShade, Gf, Vt, Sdf

import newton
import warp as wp

from ram.dataset.kinematics import forward_kinematics
from ram.dataset.morphology import sample_morph, get_joint_limits
from ram.dataset.self_collision import LINK_RADIUS
import ram.dataset.so3 as so3
from ram.model import Model

# ============================== config ==============================
torch.manual_seed(0)
DEVICE = torch.device("cuda")

# scene
N_ROBOTS, DOF = 100, 6
SPACING, BASE_Z = 4.0, 1.5
COLOR_JOINT, COLOR_BASE, COLOR_LINKS = (0.2, 0.2, 0.2), (0.1, 0.4, 0.8), (0.8, 0.8, 0.8)
JOINT_RADIUS = 1.02 * LINK_RADIUS

# trajectory
FPS, TOTAL_TIME, N_WAYPOINTS = 60, 30.0, 15
N_FRAMES = int(TOTAL_TIME * FPS)

# reachability surface
RAM_MODEL_ID = 142
VOX_DIV   = 30
VOX_R     = 1.0 / VOX_DIV
ISO_LEVEL = 0.5
SURF_COLOR = (0.54, 0.94, 0.60)

OUTPUT = "army.usd"

# ============================== setup ==============================
morph        = sample_morph(N_ROBOTS, DOF, False, device=DEVICE)          # [N, DOF+1, 3]
joint_limits = get_joint_limits(morph)
N_robots, n_links = morph.shape[0], morph.shape[1]
ram_model    = Model.from_id(RAM_MODEL_ID).to(DEVICE).eval()

grid_size = math.ceil(math.sqrt(N_robots))
idx       = torch.arange(N_robots, device=DEVICE)
rows, cols = idx // grid_size, idx % grid_size
grid_offsets = torch.stack([(cols * SPACING).float(),
                            (rows * SPACING).float(),
                            torch.full((N_robots,), BASE_Z, device=DEVICE)], dim=1)   # [N, 3]

poses_home = forward_kinematics(morph, torch.zeros(N_robots, n_links, 1, device=DEVICE))   # [N, n_links, 4, 4]

# ---- one ball grid, reused for the query (sparse cells) and the volume scatter (dense mask) ----
def make_ball_grid():
    lin  = torch.linspace(-1 + VOX_R, 1 - VOX_R, VOX_DIV, device=DEVICE)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), dim=-1).reshape(-1, 3)
    inside = grid.norm(dim=-1) <= 1.0
    return grid[inside], inside

cells, inside_mask = make_ball_grid()
N_cells         = cells.shape[0]
SPACING_VOX     = 2.0 * VOX_R            # marching-cubes step (== linspace step)
ORIGIN_VOX      = -1.0 + VOX_R           # shifts voxel-index coords back into the ball

# ============================== robot model ==============================
def build_army():
    builder = newton.ModelBuilder()
    #builder.add_ground_plane(height=0.5)

    for i in range(N_robots):
        off = poses_home.new_zeros(3); off[:] = grid_offsets[i]
        joints, links = [], []

        for j in range(n_links):
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

    return builder.finalize(device="cpu")

# ============================== trajectory ==============================
def sample_joint_waypoints(joint_limits, n_wp):
    shape = (n_wp, *joint_limits.shape[:-1], 1)
    return torch.rand(shape, device=joint_limits.device) * joint_limits[..., 0:1] + joint_limits[..., 1:2]

waypoints = sample_joint_waypoints(joint_limits, N_WAYPOINTS)

def q_at_batch(ts):
    s   = ts / TOTAL_TIME * (N_WAYPOINTS - 1)
    seg = s.floor().clamp(0, N_WAYPOINTS - 2).long()
    a   = (s - seg).clamp(0.0, 1.0)
    a   = (a * a * (3 - 2 * a))[:, None, None, None]
    return (1 - a) * waypoints[seg] + a * waypoints[seg + 1]

def joint_q_flat(theta):
    return theta[..., 0].reshape(-1).cpu().numpy().astype(np.float32)

ts        = torch.arange(N_FRAMES, device=DEVICE) / FPS
theta_all = q_at_batch(ts)

with torch.inference_mode():
    latents = ram_model.encoder(morph.float())[1][0][-1]
    D = latents.shape[1]

    morph_b   = morph[None].expand(N_FRAMES, N_robots, n_links, 3)
    R_all     = forward_kinematics(morph_b, theta_all)[:, :, -1, :3, :3]
    rot6d_all = so3.to_vector(R_all)

# ============================== reachability query ==============================
@torch.inference_mode()
def compute_reachability():
    # Base spatial view
    pos_view = cells[None, None, :, :]  # [1, 1, N_cells, 3]

    # Allocate on CPU to save precious VRAM if N_robots * N_FRAMES * N_cells is massive
    prob_frames = torch.empty(N_FRAMES, N_robots, N_cells, device="cpu")

    BATCH_FRAMES = 1  # Can increase this now that robot dimension is chunked
    BATCH_ROBOTS = 50   # Tune this chunk size down if you still hit OOMs

    for start_f in tqdm(range(0, N_FRAMES, BATCH_FRAMES), desc="reachability frames"):
        end_f = min(start_f + BATCH_FRAMES, N_FRAMES)
        b_frames = end_f - start_f

        # Slice temporal chunk orientations -> [b_frames, N_robots, 6]
        rot6d_frame_chunk = rot6d_all[start_f:end_f]

        for start_r in range(0, N_robots, BATCH_ROBOTS):
            end_r = min(start_r + BATCH_ROBOTS, N_robots)
            b_robots = end_r - start_r

            # Slice latent vectors and orientations for the current robot sub-batch
            latent_chunk = latents[start_r:end_r][None, :, None, :]      # [1, b_robots, 1, D]
            rot6d_chunk  = rot6d_frame_chunk[:, start_r:end_r, None, :]  # [b_frames, b_robots, 1, 6]

            # Broadcast natively via virtual memory views
            pos_expanded    = pos_view.expand(b_frames, b_robots, N_cells, 3)
            rot6d_expanded  = rot6d_chunk.expand(b_frames, b_robots, N_cells, 6)
            latent_expanded = latent_chunk.expand(b_frames, b_robots, N_cells, D)

            # Materialize the sub-batch chunk into VRAM for the MLP pass
            inp_flat = torch.cat([pos_expanded, rot6d_expanded, latent_expanded], dim=-1).reshape(-1, 9 + D)

            # Global vectorized evaluation inside VRAM context
            logits = ram_model.decoder(inp_flat).squeeze(-1)
            probs  = torch.sigmoid(logits).reshape(b_frames, b_robots, N_cells)

            # Push the evaluated chunk back out to the CPU storage array
            prob_frames[start_f:end_f, start_r:end_r] = probs.cpu()

    return prob_frames

# ============================== drive + record ==============================
def record(model, state):
    viewer = newton.viewer.ViewerUSD(output_path=OUTPUT, fps=FPS, up_axis="Z")
    viewer.set_model(model)
    for f in tqdm(range(N_FRAMES), desc="simulate"):
        state.joint_q.assign(joint_q_flat(theta_all[f]))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(f / FPS); viewer.log_state(state); viewer.end_frame()
    viewer.close()

# ============================== material + authoring ==============================
def make_surface_material(stage, mat_path, color):
    mat    = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    prim   = shader.GetPrim()
    prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    prim.CreateAttribute("info:mdl:sourceAsset",      Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniGlass.mdl"))
    prim.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("OmniGlass")

    shader.CreateInput("glass_color",         Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("glass_ior",           Sdf.ValueTypeNames.Float).Set(1.0)    # no refraction
    shader.CreateInput("depth",               Sdf.ValueTypeNames.Float).Set(0.0)    # no absorption depth tint
    shader.CreateInput("thin_walled",         Sdf.ValueTypeNames.Bool).Set(True)    # treat as thin shell, no refraction offset
    shader.CreateInput("reflection_color",    Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))

    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    out.GetAttr().SetMetadata("renderType", "material")
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat
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


@torch.inference_mode()
def author_surfaces(usd_path, prob_frames, level, color):
    print("Allocating 3D dense grids in VRAM...")
    vol_gpu = torch.zeros(
        N_FRAMES, N_robots, VOX_DIV, VOX_DIV, VOX_DIV,
        device=DEVICE, dtype=torch.float32
    )
    vol_gpu.view(N_FRAMES, N_robots, -1)[..., inside_mask] = prob_frames.to(DEVICE)

    print("Applying temporal stabilization...")
    vol_smoothed = vol_gpu.clone()
    for f in range(1, N_FRAMES - 1):
        # 3-frame Gaussian-style moving average
        vol_smoothed[f] = 0.25 * vol_gpu[f-1] + 0.5 * vol_gpu[f] + 0.25 * vol_gpu[f+1]
    vol_gpu = vol_smoothed

    print("Opening USD Stage...")
    stage   = Usd.Stage.Open(usd_path)
    default = stage.GetDefaultPrim()
    assert default and default.IsValid(), f"{usd_path} has no default prim"
    base = default.GetPath()
    add_lab_floor(stage, base)
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(N_FRAMES - 1)

    looks = base.AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks)
    mat = make_surface_material(stage, looks.AppendChild("ReachabilityMaterial"), color)

    # Instantiate one UniformGrid on the GPU and reuse it safely
    grid = isoext.UniformGrid([VOX_DIV, VOX_DIV, VOX_DIV])

    for i in tqdm(range(N_robots), desc="Authoring USD Meshes"):
        mesh = UsdGeom.Mesh.Define(
            stage, base.AppendChild("Reachability").AppendChild(f"robot_{i}").AppendChild("surface")
        )
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*grid_offsets[i].tolist()))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

        pts  = mesh.CreatePointsAttr()
        cnts = mesh.CreateFaceVertexCountsAttr()
        idxs = mesh.CreateFaceVertexIndicesAttr()

        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        for f in range(N_FRAMES):
            grid.set_values(vol_gpu[f, i] - level)
            verts_gpu, faces_gpu = isoext.marching_cubes(grid)

            if verts_gpu.shape[0] == 0 or faces_gpu.shape[0] == 0:
                continue

            tc = Usd.TimeCode(f)
            pts.Set(Vt.Vec3fArray.FromNumpy(verts_gpu.cpu().numpy()), tc)
            cnts.Set(Vt.IntArray.FromNumpy(np.full(len(faces_gpu), 3, dtype=np.int32)), tc)
            idxs.Set(Vt.IntArray.FromNumpy(faces_gpu.reshape(-1).cpu().numpy()), tc)

    stage.Save()

# ============================== run ==============================
model = build_army()
state = model.state()
assert state.body_q.shape[0] == N_robots * n_links

prob_frames = compute_reachability()
record(model, state)
author_surfaces(OUTPUT, prob_frames, ISO_LEVEL, SURF_COLOR)

print("Pipeline finished successfully!")