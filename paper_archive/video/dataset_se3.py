import os
import math
import subprocess
import torch
import numpy as np
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()
import pyvista as pv
from scipy.spatial import SphericalVoronoi
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)
pv.OFF_SCREEN = True
pv.global_theme.transparent_background = True

# ======================= Configuration =======================
COLOR_REACH   = "#de8f05"   # orange  (reached cell)
COLOR_UNREACH = "#0173b2"   # blue    (flood-filled / assumed reachable)
COLOR_ACTIVE  = "#F52E0B"   # red     (voxel whose sphere is currently shown)
COLOR_EMPTY   = "#d8dbe0"   # light grey (unreached sphere cells)
COLOR_BLACK   = "#111111"
COLOR_GRID    = "#9aa0a6"   # light grey wireframe
SPHERE_BODY   = "#ffffff"   # occluder colour for hidden-line removal

WORKSPACE_BOUNDS = 1.2

FPS          = 60
TOTAL_TIME   = 30.0
WP_ARRIVALS  = int(TOTAL_TIME * FPS)

N_WAYPOINTS  = 15
DWELL        = 0.4

FLOOD             = True
FLOOD_HOLD_FRAMES = 120

N_FRAMES     = WP_ARRIVALS + FLOOD_HOLD_FRAMES

GRID_OPACITY   = 1.0
ACTIVE_OPACITY = 0.45
REACH_FILL_OP  = 0.50
SPHERE_OCCLUDE = False
VIEW_SPIN      = False
VIEW_AZIMUTH   = math.radians(45)
SPIN_TURNS     = 0.5
CAM_ELEV       = 0.42

PREVIEW = False
if PREVIEW:
    OUT_W, OUT_H, FRAME_STRIDE = 900, 460, 30
else:
    OUT_W, OUT_H, FRAME_STRIDE = 1920, 1080, 1

CAP_RADIUS = 0.04
CAP_SEG    = 12
DEVICE = torch.device("cpu")

def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

RGB_REACH = hex_rgb(COLOR_REACH)
RGB_EMPTY = hex_rgb(COLOR_EMPTY)

# ======================= Geometry: R3 (Cube) =======================
VPS = 4
VS = 1.0 / VPS
N_CUBE_CELLS = VPS ** 3

def init_r3_geometry():
    idx = torch.arange(VPS)
    gi, gj, gk = torch.meshgrid(idx, idx, idx, indexing='ij')
    origins = torch.stack([gi.flatten(), gj.flatten(), gk.flatten()], dim=1).float() * VS
    offsets = torch.tensor([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ], dtype=torch.float32) * VS
    all_vertices = origins[:, None, :] + offsets[None, :, :]
    base_faces = torch.tensor([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]
    ])
    verts_flat = all_vertices.reshape(-1, 3)
    face_offsets = torch.arange(N_CUBE_CELLS)[:, None, None] * 8
    faces_flat = (base_faces[None].expand(N_CUBE_CELLS, -1, -1) + face_offsets).reshape(-1, 3)
    return verts_flat, faces_flat

def get_cube_idx(pos):
    ijk = torch.clamp(torch.floor(pos / VS), 0, VPS - 1).long()
    return int(ijk[0] * VPS * VPS + ijk[1] * VPS + ijk[2])

R3_VERTS, R3_FACES = init_r3_geometry()
CELL_EDGE_PAIRS = torch.tensor([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

# ======================= Geometry: SO3 (Sphere) =======================
def subdivide_triangle_on_sphere(a, b, c, edge_splits):
    local_vertices, vertex_index = [], {}
    for i in range(edge_splits + 1):
        for j in range(edge_splits + 1 - i):
            k = edge_splits - i - j
            p = (i * a + j * b + k * c) / edge_splits
            p /= torch.linalg.norm(p)
            vertex_index[(i, j)] = len(local_vertices)
            local_vertices.append(p)
    local_tris = []
    for i in range(edge_splits):
        for j in range(edge_splits - i):
            v0 = vertex_index[(i,j)]; v1 = vertex_index[(i+1,j)]; v2 = vertex_index[(i,j+1)]
            local_tris.append((v0, v1, v2))
            if j < edge_splits - i - 1:
                v3 = vertex_index[(i+1,j+1)]
                local_tris.append((v1, v3, v2))
    return torch.stack(local_vertices), local_tris

def build_spherical_cap(center, radius_angle=CAP_RADIUS, segments=CAP_SEG):
    n = center / torch.linalg.norm(center)
    helper = torch.tensor([0.0, 0.0, 1.0], dtype=n.dtype)
    if torch.abs(torch.dot(n, helper)) > 0.9:
        helper = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype)
    u = torch.linalg.cross(n, helper); u = u / torch.linalg.norm(u)
    v = torch.linalg.cross(n, u)
    ra = torch.as_tensor(radius_angle, dtype=n.dtype)
    thetas = torch.arange(segments + 1, dtype=n.dtype) * (2.0 * math.pi / segments)
    ring = (torch.cos(ra) * n[None, :]
            + torch.sin(ra) * (torch.cos(thetas)[:, None] * u[None, :]
                               + torch.sin(thetas)[:, None] * v[None, :]))
    ring = ring / torch.linalg.norm(ring, dim=1, keepdim=True)
    verts = torch.cat([n[None, :], ring], dim=0)
    s = torch.arange(segments, dtype=torch.long)
    return verts, torch.zeros(segments, dtype=torch.long), 1 + s, 2 + s

def init_so3_geometry(subdivisions=2):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = torch.tensor([
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
        [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]
    ], dtype=torch.float64)
    vertices /= torch.linalg.norm(vertices[0])
    faces = [
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],
        [5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],
        [3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ]
    for _ in range(subdivisions):
        new_faces, vertex_map = [], {}
        def get_midpoint(p1, p2, verts, vmap):
            key = tuple(sorted((p1, p2)))
            if key not in vmap:
                mid = (verts[p1] + verts[p2]) / 2.0
                mid = mid / torch.linalg.norm(mid)
                verts = torch.cat([verts, mid.unsqueeze(0)], dim=0)
                vmap[key] = len(verts) - 1
            return vmap[key], verts
        for face in faces:
            m1, vertices = get_midpoint(face[0], face[1], vertices, vertex_map)
            m2, vertices = get_midpoint(face[1], face[2], vertices, vertex_map)
            m3, vertices = get_midpoint(face[2], face[0], vertices, vertex_map)
            new_faces.extend([[face[0],m1,m3],[face[1],m2,m1],[face[2],m3,m2],[m1,m2,m3]])
        faces = new_faces
    points = vertices
    sv = SphericalVoronoi(points.numpy(), 1, np.zeros(3))
    sv.sort_vertices_of_regions()
    final_vertices, tri_i, tri_j, tri_k, tri_to_region = [], [], [], [], []
    edge_segs, edge_region = [], []
    for region_idx, region in enumerate(sv.regions):
        poly = torch.tensor(sv.vertices[region], dtype=torch.float64)
        for j in range(len(poly) - 2):
            lv, lt = subdivide_triangle_on_sphere(poly[0], poly[j+1], poly[j+2], 2)
            off = len(final_vertices)
            final_vertices.extend(lv.tolist())
            for tri in lt:
                tri_i.append(off+tri[0]); tri_j.append(off+tri[1]); tri_k.append(off+tri[2])
                tri_to_region.append(region_idx)
        for j in range(len(region)):
            edge_segs.append([poly[j].tolist(), poly[(j+1) % len(region)].tolist()])
            edge_region.append(region_idx)
    return (torch.tensor(final_vertices), tri_i, tri_j, tri_k, tri_to_region,
            np.array(edge_segs, dtype=float), np.array(edge_region), points)

(SO3_VERTS, SO3_I, SO3_J, SO3_K, TRI_TO_REGION,
 SO3_EDGE_SEGS, SO3_EDGE_REGION, SO3_SITES) = init_so3_geometry()
N_SPHERE_CELLS = SO3_SITES.shape[0]
TRI_TO_REGION  = np.asarray(TRI_TO_REGION)
SO3_TRIS       = np.column_stack([np.asarray(SO3_I), np.asarray(SO3_J), np.asarray(SO3_K)])

def get_sphere_idx(axis):
    return int(torch.argmax(SO3_SITES @ axis))

def axes_from_rots(R):
    tr = R[:,0,0] + R[:,1,1] + R[:,2,2]
    angle = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0))
    ax = torch.stack([R[:,2,1]-R[:,1,2], R[:,0,2]-R[:,2,0], R[:,1,0]-R[:,0,1]], dim=1)
    sin_a = torch.sin(angle).unsqueeze(1)
    safe = sin_a.abs() > 1e-5
    default = torch.tensor([0.0, 0.0, 1.0], dtype=R.dtype)
    ax = torch.where(safe, ax / (2.0 * sin_a).clamp_min(1e-9), default)
    return ax / torch.linalg.norm(ax, dim=1, keepdim=True).clamp_min(1e-9)

# ======================= PyVista mesh builders =======================
def segments_polydata(segs):
    segs = np.asarray(segs, dtype=float)
    if segs.size == 0:
        return None
    m = segs.shape[0]
    pts = segs.reshape(-1, 3)
    lines = np.hstack([np.full((m,1), 2), np.arange(2*m).reshape(m,2)]).astype(np.int64).ravel()
    pd = pv.PolyData(pts)
    pd.lines = lines
    return pd

def faces_polydata(points, tris):
    if len(tris) == 0:
        return None
    faces_pv = np.hstack([np.full((tris.shape[0],1), 3), tris]).astype(np.int64).ravel()
    return pv.PolyData(points, faces_pv)

def cells_polydata(cell_ids):
    cell_ids = list(cell_ids)
    if not cell_ids:
        return None
    rows = torch.cat([torch.arange(c*12, c*12+12) for c in cell_ids])
    return faces_polydata(R3_VERTS.numpy(), R3_FACES[rows].numpy())

def cell_edges_segments(v):
    pts = R3_VERTS[v*8:(v+1)*8]
    return np.array([[pts[a].tolist(), pts[b].tolist()] for a,b in CELL_EDGE_PAIRS.tolist()], dtype=float)

def cells_edges_segments(cell_ids):
    cell_ids = list(cell_ids)
    if not cell_ids:
        return None
    return np.concatenate([cell_edges_segments(v) for v in cell_ids], axis=0)

def sphere_region_fill(regions):
    regions = set(regions)
    if not regions:
        return None
    mask = np.isin(TRI_TO_REGION, list(regions))
    return faces_polydata(SO3_VERTS.numpy(), SO3_TRIS[mask])

def sphere_region_edges(regions):
    regions = set(regions)
    if not regions:
        return None
    mask = np.isin(SO3_EDGE_REGION, list(regions))
    return segments_polydata(SO3_EDGE_SEGS[mask] * 1.004)

def caps_polydata(vox_wps, cap_nv):
    if not vox_wps:
        return None
    vs = torch.cat([wp_caps[w][0] for w in vox_wps], dim=0).numpy() * 1.006
    f_list = []
    for p, w in enumerate(vox_wps):
        off = p * cap_nv
        ii, jj, kk = wp_caps[w][1]+off, wp_caps[w][2]+off, wp_caps[w][3]+off
        f_list.append(np.column_stack([ii.numpy(), jj.numpy(), kk.numpy()]))
    return faces_polydata(vs, np.concatenate(f_list, axis=0))

# Static geometry
GRID_PD = segments_polydata(np.concatenate([cell_edges_segments(v) for v in range(N_CUBE_CELLS)], axis=0))
_box = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
_box_pairs = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
BOX_PD = segments_polydata(np.array([[_box[a], _box[b]] for a,b in _box_pairs]))
SO3_CAGE_PD = segments_polydata(SO3_EDGE_SEGS * 1.003)
OCC_PD = faces_polydata(SO3_VERTS.numpy() * 0.99, SO3_TRIS)

# ======================= Waypoint poses =======================
ROBOT = 0
try:
    wp_poses = torch.load("eef_waypoints.pt").double()[:, ROBOT]
    wp_coll  = torch.load("collision_waypoints.pt").bool()[:, ROBOT]
    N_WAYPOINTS = wp_poses.shape[0]
    print(f"Loaded {N_WAYPOINTS} waypoints ({int((~wp_coll).sum())} collision-free).")
except FileNotFoundError:
    print("WARNING: waypoint files not found. Generating a dummy trajectory for testing.")
    g = torch.Generator().manual_seed(0)
    pos = (torch.rand(N_WAYPOINTS, 3, generator=g) * 2 - 1) * (0.8 * WORKSPACE_BOUNDS)
    ax = torch.nn.functional.normalize(torch.randn(N_WAYPOINTS, 3, generator=g), dim=1)
    ang = torch.rand(N_WAYPOINTS, generator=g) * math.pi
    K = torch.zeros(N_WAYPOINTS, 3, 3)
    K[:,0,1], K[:,0,2] = -ax[:,2], ax[:,1]
    K[:,1,0], K[:,1,2] =  ax[:,2],-ax[:,0]
    K[:,2,0], K[:,2,1] = -ax[:,1], ax[:,0]
    eye = torch.eye(3).expand(N_WAYPOINTS, 3, 3)
    Rm = (eye + torch.sin(ang)[:,None,None]*K + (1-torch.cos(ang))[:,None,None]*(K@K)).double()
    wp_poses = torch.eye(4).double().expand(N_WAYPOINTS, 4, 4).clone()
    wp_poses[:,:3,:3] = Rm; wp_poses[:,:3,3] = pos.double()
    wp_coll = torch.zeros(N_WAYPOINTS, dtype=torch.bool); wp_coll[3] = True

positions = wp_poses[:, :3, 3]
rotations = wp_poses[:, :3, :3]
pos_norm = torch.clamp(0.5 + 0.5*(positions/WORKSPACE_BOUNDS), 0.0, 1.0-1e-5)
axes = axes_from_rots(rotations)

wp_cube_idx   = [get_cube_idx(pos_norm[w]) for w in range(N_WAYPOINTS)]
wp_sphere_idx = [get_sphere_idx(axes[w]) for w in range(N_WAYPOINTS)]
wp_caps       = [build_spherical_cap(axes[w]) for w in range(N_WAYPOINTS)]
CAP_NV        = wp_caps[0][0].shape[0]

move_time = (TOTAL_TIME - N_WAYPOINTS * DWELL) / (N_WAYPOINTS - 1)
wp_period = DWELL + move_time
try:
    wp_frame = torch.load("wp_arrival_frames.pt").tolist()
except FileNotFoundError:
    wp_frame = [int(round(w * wp_period * FPS)) for w in range(N_WAYPOINTS)]
flood_start_f = max(wp_frame)
def active_waypoints(f):
    return [w for w in range(N_WAYPOINTS) if wp_frame[w] <= f and not bool(wp_coll[w])]

# ======================= Render Loop =======================
os.makedirs("frames_pv", exist_ok=True)

def main():
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(OUT_W, OUT_H), border=False)
    pl.enable_anti_aliasing('fxaa')

    # Static actors
    pl.subplot(0, 0)
    pl.add_mesh(GRID_PD, color=COLOR_GRID, opacity=GRID_OPACITY, line_width=2.0, name='grid')
    pl.add_mesh(BOX_PD, color='black', line_width=5.0, name='box')
    pl.enable_depth_peeling(8)

    pl.subplot(0, 1)
    if SPHERE_OCCLUDE:
        pl.add_mesh(OCC_PD, color=SPHERE_BODY, name='occluder',
                    lighting=True, ambient=0.95, diffuse=0.05, specular=0.0)
    pl.add_mesh(SO3_CAGE_PD, color=COLOR_GRID, line_width=1.6, name='cage')
    pl.enable_depth_peeling(8)

    def set_cameras(f):
        az = VIEW_AZIMUTH + (2.0*math.pi*SPIN_TURNS*f/N_FRAMES if VIEW_SPIN else 0.0)
        d = (math.cos(az), math.sin(az), CAM_ELEV)
        pl.subplot(0, 0)
        fp = (0.5, 0.5, 0.5)
        pl.camera_position = [(fp[0]+d[0], fp[1]+d[1], fp[2]+d[2]), fp, (0,0,1)]
        pl.reset_camera(); pl.camera.zoom(1.0)
        pl.subplot(0, 1)
        pl.camera_position = [(d[0], d[1], d[2]), (0,0,0), (0,0,1)]
        pl.reset_camera(); pl.camera.zoom(1.35)

    def safe_remove(name):
        try:
            pl.remove_actor(name, render=False)
        except Exception:
            pass

    frames_to_render = range(0, N_FRAMES, FRAME_STRIDE)
    print(f"Rendering {len(frames_to_render)} PyVista frames at {OUT_W}x{OUT_H}...")

    for idx, f in enumerate(tqdm(frames_to_render)):
        if idx < 1800:
            continue
        active = active_waypoints(f)
        reached_cells = sorted({wp_cube_idx[w] for w in active})
        active_voxel  = wp_cube_idx[active[-1]] if active else None

        if active_voxel is not None:
            vox_wps       = [w for w in active if wp_cube_idx[w] == active_voxel]
            reached_regions = {wp_sphere_idx[w] for w in vox_wps}
        else:
            vox_wps, reached_regions = [], set()
        blue_cells = [c for c in reached_cells if c != active_voxel]

        # ---- Cube panel ----
        pl.subplot(0, 0)
        blue_pd = cells_polydata(blue_cells)
        if blue_pd is not None:
            pl.add_mesh(blue_pd, color=COLOR_REACH, opacity=0.12,
                        show_edges=False, smooth_shading=False, name='blue')
            pl.add_mesh(segments_polydata(cells_edges_segments(blue_cells)),
                        color=COLOR_REACH, line_width=3.0, name='blue_edges')
        else:
            safe_remove('blue'); safe_remove('blue_edges')

        if active_voxel is not None:
            pl.add_mesh(cells_polydata([active_voxel]), color=COLOR_ACTIVE, opacity=ACTIVE_OPACITY,
                        show_edges=False, smooth_shading=False, name='active')
            pl.add_mesh(segments_polydata(cell_edges_segments(active_voxel)),
                        color=COLOR_ACTIVE, line_width=4.5, name='active_edges')
        else:
            safe_remove('active'); safe_remove('active_edges')

        if active:
            pl.add_mesh(pv.PolyData(pos_norm[active].numpy()), color=COLOR_BLACK,
                        point_size=25, render_points_as_spheres=True, name='dots')
        else:
            safe_remove('dots')

        # ---- Sphere panel ----
        pl.subplot(0, 1)
        fill_pd = sphere_region_fill(reached_regions)
        if fill_pd is not None:
            pl.add_mesh(fill_pd, color=COLOR_REACH, opacity=REACH_FILL_OP,
                        show_edges=False, smooth_shading=False, name='sph_fill')
            pl.add_mesh(sphere_region_edges(reached_regions),
                        color=COLOR_REACH, line_width=3.0, name='sph_edges')
        else:
            safe_remove('sph_fill'); safe_remove('sph_edges')

        caps_pd = caps_polydata(vox_wps, CAP_NV)
        if caps_pd is not None:
            pl.add_mesh(caps_pd, color=COLOR_BLACK, name='caps')
        else:
            safe_remove('caps')

        # ---- Flood fill (closed-world over-approximation) ----
        # Unvisited cube cells → COLOR_UNREACH (blue); same for sphere regions.
        # Uses distinct actor names so it never collides with orange reached actors.
        flooding = FLOOD and f >= flood_start_f
        if flooding:
            unreached_cube = sorted(set(range(N_CUBE_CELLS)) - set(reached_cells))
            pl.subplot(0, 0)
            unreach_cube_pd = cells_polydata(unreached_cube)
            if unreach_cube_pd is not None:
                pl.add_mesh(unreach_cube_pd, color=COLOR_UNREACH, opacity=0.12,
                            show_edges=False, smooth_shading=False, name='flood_cube')
                pl.add_mesh(segments_polydata(cells_edges_segments(unreached_cube)),
                            color=COLOR_UNREACH, line_width=3.0, name='flood_cube_edges')
            else:
                safe_remove('flood_cube'); safe_remove('flood_cube_edges')

            # Sphere: unreached regions of the N_SPHERE_CELLS geodesic cells
            unreached_regions = set(range(N_SPHERE_CELLS)) - reached_regions
            pl.subplot(0, 1)
            unreach_sph_pd = sphere_region_fill(unreached_regions)
            if unreach_sph_pd is not None:
                pl.add_mesh(unreach_sph_pd, color=COLOR_UNREACH, opacity=REACH_FILL_OP,
                            show_edges=False, smooth_shading=False, name='flood_sph_fill')
                pl.add_mesh(sphere_region_edges(unreached_regions),
                            color=COLOR_UNREACH, line_width=3.0, name='flood_sph_edges')
            else:
                safe_remove('flood_sph_fill'); safe_remove('flood_sph_edges')
        else:
            pl.subplot(0, 0)
            safe_remove('flood_cube'); safe_remove('flood_cube_edges')
            pl.subplot(0, 1)
            safe_remove('flood_sph_fill'); safe_remove('flood_sph_edges')

        set_cameras(f)
        pl.screenshot(f"frames_pv/f_{idx:04d}.png", transparent_background=True)

    pl.close()

    print("Compiling DNxHR .mov via FFmpeg...")
    output_video = "dataset.mov"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", "frames_pv/f_%04d.png",
        "-c:v", "dnxhd", "-profile:v", "dnxhr_hq", "-pix_fmt", "yuv422p",
        "-r", str(FPS), output_video
    ], check=True)
    print(f"Done -> {output_video}")


if __name__ == "__main__":
    main()