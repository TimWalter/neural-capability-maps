import re
import argparse
from pathlib import Path

import torch
import zarr
import fasteners
from tqdm import tqdm

import ram.dataset.se3 as se3
from ram.dataset.morphology import sample_morph
from ram.dataset.workspace import synthesise_data
from ram.dataset.boundaries import sample_boundary

CHUNK_SIZE = 100_000  # ~4.4MB
SHARD_SIZE = CHUNK_SIZE * 1000  # ~4.4GB

parser = argparse.ArgumentParser()
parser.add_argument("--set", type=str, default="val", choices=["val", "test"], help="Set type")
parser.add_argument("--dof", type=int, default=6, help="Degrees of freedom")
parser.add_argument("--num_robots", type=int, default=2, help="Number of robots to generate")
parser.add_argument("--num_samples", type=int, default=100_000, help="Number of samples per robot")
parser.add_argument("--num_geodesic", type=int, default=1000, help="Number of geodesics per robot")
parser.add_argument("--num_geodesic_samples", type=int, default=100, help="Number of samples per geodesics")
args = parser.parse_args()

assert args.num_samples * args.num_robots % CHUNK_SIZE == 0, \
    f"Only full chunks are supported (chunk size {CHUNK_SIZE})"
assert args.num_robots * args.num_geodesic * args.num_geodesic_samples % CHUNK_SIZE == 0, \
    f"Only full chunks are supported (chunk size {CHUNK_SIZE})"

assert SHARD_SIZE / args.num_samples == SHARD_SIZE // args.num_samples, \
    f"One robot must belong to one shard (shard size {SHARD_SIZE})"
assert (SHARD_SIZE / (args.num_geodesic * args.num_geodesic_samples) ==
        SHARD_SIZE // (args.num_geodesic * args.num_geodesic_samples)), \
    f"One robot must belong to one shard (shard size {SHARD_SIZE})"

SAFE_FOLDER = Path(__file__).parent.parent.parent / "data" / args.set
BOUNDARY_FOLDER = safe_folder = Path(__file__).parent.parent.parent / 'data' / (args.set + "_boundary")

lock = fasteners.InterProcessLock(SAFE_FOLDER.parent / f'{args.set}_lock.file')
compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)

with lock:
    SAFE_FOLDER.mkdir(parents=True, exist_ok=True)
    BOUNDARY_FOLDER.mkdir(parents=True, exist_ok=True)

root = zarr.open(SAFE_FOLDER, mode="a")
boundary_root = zarr.open(BOUNDARY_FOLDER, mode="a")
with lock:
    file_indices = [
        int(match.group(1))
        for k in root.array_keys()
        if (match := re.search(r'^(\d+)_samples$', k))
    ]
    file_idx = (max(file_indices) + 1) if file_indices else 0
    morph_offset = sum([root[f"{idx}_morphologies"].shape[0] for idx in file_indices])

    morph_filename = str(file_idx) + "_morphologies"
    sample_filename = str(file_idx) + "_samples"

    root.create_array(
        morph_filename,
        shape=(args.num_robots, args.dof + 1, 3),
        dtype="float32",
        chunks=(args.num_robots, args.dof + 1, 3),
        compressors=compressor,
        overwrite=False,
    )
    root.create_array(sample_filename,
                      shape=(args.num_robots * args.num_samples, 11),
                      dtype="float32",
                      chunks=(CHUNK_SIZE, 11),
                      shards=(SHARD_SIZE, 11),
                      compressors=compressor)

    boundary_root.create_array(
        morph_filename,
        shape=(args.num_robots, args.dof + 1, 3),
        dtype="float32",
        chunks=(args.num_robots, args.dof + 1, 3),
        compressors=compressor,
        overwrite=False,
    )
    boundary_root.create_array(sample_filename,
                               shape=(args.num_robots * args.num_geodesic * args.num_geodesic_samples, 11),
                      dtype="float32",
                      chunks=(CHUNK_SIZE, 11),
                      shards=(SHARD_SIZE, 11),
                      compressors=compressor)


morphs = sample_morph(args.num_robots, args.dof, False, torch.device("cuda"))
root[morph_filename][0:] = morphs.cpu().numpy()
boundary_root[morph_filename][0:] = morphs.cpu().numpy()

file = [root[sample_filename], boundary_root[sample_filename]]
file_offset = [0,0]
buffer = [
    torch.zeros(min(args.num_robots * args.num_samples, SHARD_SIZE), 11, dtype=torch.float32),
    torch.zeros(min(args.num_robots * args.num_geodesic * args.num_geodesic_samples, SHARD_SIZE), 11, dtype=torch.float32)
]
buffer_offset = [0,0]

desc = f"Eval set file {file_idx} with morph offset {morph_offset} for {args.dof} DoF robots"
for idx, morph in enumerate(tqdm(morphs, desc=desc)):
    random_poses, random_labels = synthesise_data(morph, args.num_samples, return_poses=True, use_ik=True)
    boundary_poses, boundary_labels = sample_boundary(morph, args.num_geodesic, args.num_geodesic_samples)

    for i, (poses, labels) in enumerate(zip([random_poses, boundary_poses], [random_labels, boundary_labels])):
        poses = se3.to_vector(poses)
        labels = labels.float().unsqueeze(1)
        morph_ids = torch.full_like(labels, idx + morph_offset)

        samples = torch.cat([morph_ids, poses, labels], dim=1)

        buffer[i][buffer_offset[i]: buffer_offset[i] + samples.shape[0]] = samples
        buffer_offset[i] += samples.shape[0]

        if buffer_offset[i] == buffer[i].shape[0] or idx == morphs.shape[0] - 1:
            active_data = buffer[i][:buffer_offset[i]]
            active_data = active_data[torch.randperm(active_data.shape[0])]

            file[i][file_offset[i]: file_offset[i] + active_data.shape[0]] = active_data.cpu().numpy()
            file_offset[i] += active_data.shape[0]

            buffer_offset[i] = 0
