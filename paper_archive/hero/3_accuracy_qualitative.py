import ram.dataset.se3 as se3
from ram.dataset.workspace import synthesise_data
from ram.dataset.kinematics import forward_kinematics, inverse_kinematics
from ram.dataset.self_collision import get_capsules, LINK_RADIUS
from ram.model import Model
from ram.dataset.morphology import sample_morph

import torch

import seaborn as sns

from plotly.subplots import make_subplots
from paper_archive.utils import get_cylinder_mesh, get_sphere_mesh
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio



def plot_hero_accuracy(morph, name):
    poses, labels = synthesise_data(morph, 100_000, return_poses=True, use_ik=True)
    predictions = model.predict(morph.unsqueeze(0).expand(poses.shape[0], -1, -1), se3.to_vector(poses)) > 0

    print((labels == predictions).sum()/labels.shape[0])

    s_all, e_all = get_capsules(morph, forward_kinematics(morph, inverse_kinematics(morph, poses[labels][0:1])[0][0]))

    palette = sns.color_palette("colorblind", 4)
    color_true = palette[2]
    color_false = palette[3]

    COLOR_JOINT =  f"rgb{tuple((255*np.array((0.2, 0.2, 0.2))).astype(int).tolist())}"  # Dark Grey for motors
    COLOR_BASE =  f"rgb{tuple((255*np.array((0.1, 0.4, 0.8))).astype(int).tolist())}"  # Blue for base
    COLOR_LINKS =  f"rgb{tuple((255*np.array((0.8, 0.8, 0.8))).astype(int).tolist())}"  # Light Grey

    COLOR_T =  f"rgb{tuple((255*np.array(color_true)).astype(int).tolist())}"  # Dark Grey for motors
    COLOR_F =  f"rgb{tuple((255*np.array(color_false)).astype(int).tolist())}"  # Blue for base

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scene"}]],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )

    a = morph[..., -1, 1:2]
    d = morph[..., -1, 2:3]

    idx = -1
    while torch.any(mask := (a[..., 0] == 0) & (d[..., 0] == 0)):
        idx -= 1
        if len(morph.shape) > 2:
            a[mask] = morph[mask, idx, 1:2]
            d[mask] = morph[mask, idx, 2:3]
        else:
            a = morph[idx, 1:2]
            d = morph[idx, 2:3]
    a = a.abs()
    d = d.abs()

    for i, p in enumerate([poses[labels == predictions], poses[labels != predictions]]):
        origins = p[:, :3, 3]
        z_axes = p[:, :3, 2]
        x_axes = p[:, :3, 0]
        # Calculate start points
        z_ends = origins - (z_axes * 0.025 * (d / (a + d)))
        x_ends = z_ends - (x_axes * 0.025 * (a / (a + d)))

        # Build line segments: [start, origin, NaN]
        l_shapes = torch.stack([z_ends, origins, z_ends, x_ends], dim=1)
        nans = torch.full((l_shapes.shape[0], 1, 3), float('nan'))
        with_nans = torch.cat([l_shapes, nans], dim=1)

        plot_data = with_nans.reshape(-1, 3).numpy()

        fig.add_trace(
            go.Scatter3d(
                x=plot_data[:, 0],
                y=plot_data[:, 1],
                z=plot_data[:, 2],
                mode='lines',
                opacity=0.25,
                line=dict(color=COLOR_T if i == 0 else COLOR_F, width=1),
                hoverinfo='skip'
            ), row=1, col=1)
        fig.add_trace(
            go.Scatter3d(
                x=origins[:, 0],
                y=origins[:, 1],
                z=origins[:, 2],
                mode='markers',
                marker=dict(size=1, color=COLOR_T if i == 0 else COLOR_F, opacity=0.25),
                hoverinfo='skip'
            ), row=1, col=1)


    base = 0
    for i in range(len(s_all)):
        if torch.norm(s_all[i] - e_all[i]) < 1e-6:
            continue

        cx, cy, cz = get_cylinder_mesh(s_all[i].cpu(), e_all[i].cpu(), radius=LINK_RADIUS, resolution=15)
        fig.add_trace(
            go.Surface(
                x=cx, y=cy, z=cz,
                showscale=False,
                surfacecolor=torch.zeros_like(cx),
                colorscale=[[0, COLOR_LINKS], [1, COLOR_LINKS]],
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
                hoverinfo='skip'
            ), row=1, col=1)
        sx, sy, sz = get_sphere_mesh(s_all[i], radius=LINK_RADIUS*1.1, resolution=15)
        fig.add_trace(
            go.Surface(
                x=sx, y=sy, z=sz,
                showscale=False,
                surfacecolor=torch.zeros_like(sx),
                colorscale=[[0, COLOR_JOINT if base != 0 else COLOR_BASE],
                            [1, COLOR_JOINT if base != 0 else COLOR_BASE]],
                lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
                hoverinfo='skip'
            ), row=1, col=1)
        base += 1
    sx, sy, sz = get_sphere_mesh(e_all[-1], radius=LINK_RADIUS*1.1, resolution=15)
    fig.add_trace(
        go.Surface(
            x=sx, y=sy, z=sz,
            showscale=False,
            surfacecolor=torch.zeros_like(sx),
            colorscale=[[0, COLOR_LINKS],
                        [1, COLOR_LINKS]],
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2, roughness=0.5),
            hoverinfo='skip'
        ), row=1, col=1)




    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        template="plotly_white",
        # CRITICAL: Keep it transparent
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        scene=dict(
            camera=dict(
                eye=dict(x=0.75, y=0.75, z=0.75)
            ),
            domain=dict(x=[0, 1], y=[0, 1]),  # Preserved max domain
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode='data',
            bgcolor='rgba(0,0,0,0)'  # Transparent scene
        ),
        scene_aspectratio=dict(x=1, y=1, z=1)
    )
    pio.write_image(fig, f"{name}.png", width=1100, height=1100, scale=1)

    fig.show()



model = Model.from_id(142)

torch.manual_seed(1)
morph5 = sample_morph(1, 5, False)[0]
morph6 = sample_morph(1, 6, False)[0]
morph7 = sample_morph(1, 7, False)[0]
plot_hero_accuracy(morph5, "5dof")
plot_hero_accuracy(morph6, "6dof")
plot_hero_accuracy(morph7, "7dof")