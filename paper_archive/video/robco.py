import torch

from ram.dataset.self_collision import LINK_RADIUS

og_robco = torch.tensor([
    [0, 0, 0.2707],
    [torch.pi/2, 0, 0.0702],
    [torch.pi, 0.5640, 0.0512],
    [torch.pi/2, 0.0, 0.5192],
    [torch.pi/2, 0.0, 0.1192],
    [torch.pi/2, 0.0, 0.0552]
])

robco = torch.tensor([
    [0,           0,       0.2707],
    [torch.pi/2,  0,       0.0702],
    [0,           0.5640, -0.0512],  # alpha: pi -> 0,  d: d -> -d
    [-torch.pi/2, 0.0,     0.5192],  # alpha: pi/2 -> -pi/2
    [torch.pi/2,  0.0,     0.1192],
    [torch.pi/2,  0.0,     0.0552]
])

# TODO flip joint variable [2]!

def normalise(morph):
    l2_norm = torch.hypot(morph[:, 0:1], morph[:, 1:2])
    norm = l2_norm.sum(dim=0, keepdim=True)
    return morph/norm

robco = normalise(robco)
mask = (robco.abs() >= 2 * LINK_RADIUS).float()
robco *= mask
robco = normalise(robco)


# Get joint configurations in video
# Calculate the orientation at that time
# Sample the positions at that orientation
# Calculate reachability
# Render