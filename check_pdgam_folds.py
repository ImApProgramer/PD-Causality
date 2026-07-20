#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone CARE-PD skeleton visualizer.

This file does NOT import the training script and will never start training.
It can inspect:

1. Raw CARE-PD H36M-17 sequences from the NPZ file, side by side with the
   converted NTU-25 skeleton.
2. Packed CTR-GCN clips from a generated PKL file, shape (3, T, 25, 1).

Examples
--------
Raw NPZ, inspect 3 random samples and save PNGs:

python visualize_carepd_mapping.py \
    --npz care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz \
    --num-samples 3

Also make GIFs:

python visualize_carepd_mapping.py \
    --npz care-pd-dataset/h36m_3d_world_floorXZZplus_30f_or_longer.npz \
    --num-samples 2 \
    --make-gif

Inspect packed PKL clips:

python visualize_carepd_mapping.py \
    --pkl care-pd-dataset/ctrgcn_processing/PD_center_True/PD_train_1.pkl \
    --num-samples 3
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


H36M_NAMES = [
    "Pelvis", "Right Hip", "Right Knee", "Right Ankle",
    "Left Hip", "Left Knee", "Left Ankle", "Spine",
    "Thorax", "Neck", "Head", "Left Shoulder",
    "Left Elbow", "Left Wrist", "Right Shoulder",
    "Right Elbow", "Right Wrist",
]

H36M_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

NTU25_NAMES = [
    "Spine Base", "Spine Mid", "Neck", "Head",
    "Left Shoulder", "Left Elbow", "Left Wrist", "Left Hand",
    "Right Shoulder", "Right Elbow", "Right Wrist", "Right Hand",
    "Left Hip", "Left Knee", "Left Ankle", "Left Foot",
    "Right Hip", "Right Knee", "Right Ankle", "Right Foot",
    "Spine Shoulder", "Left Hand Tip", "Left Thumb",
    "Right Hand Tip", "Right Thumb",
]

NTU25_INWARD_ORI_INDEX = [
    (1, 2), (2, 21), (3, 21), (4, 3),
    (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15),
    (17, 1), (18, 17), (19, 18), (20, 19),
    (22, 23), (23, 8), (24, 25), (25, 12),
]

NTU25_EDGES = [
    (start - 1, end - 1)
    for start, end in NTU25_INWARD_ORI_INDEX
]


def center_poses(sequence: np.ndarray) -> np.ndarray:
    return sequence - sequence[:, 0:1, :]


def convert_h36m_to_ntu25(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence)

    if sequence.ndim != 3 or sequence.shape[1:] != (17, 3):
        raise ValueError(
            f"Expected H36M input shape (T, 17, 3), got {sequence.shape}"
        )

    output = np.zeros(
        (sequence.shape[0], 25, 3),
        dtype=sequence.dtype,
    )

    output[:, 0] = sequence[:, 0]
    output[:, 1] = sequence[:, 7]
    output[:, 20] = sequence[:, 8]
    output[:, 2] = sequence[:, 9]
    output[:, 3] = sequence[:, 10]

    output[:, 4] = sequence[:, 11]
    output[:, 5] = sequence[:, 12]
    output[:, 6] = sequence[:, 13]
    output[:, 7] = sequence[:, 13]
    output[:, 21] = sequence[:, 13]
    output[:, 22] = sequence[:, 13]

    output[:, 8] = sequence[:, 14]
    output[:, 9] = sequence[:, 15]
    output[:, 10] = sequence[:, 16]
    output[:, 11] = sequence[:, 16]
    output[:, 23] = sequence[:, 16]
    output[:, 24] = sequence[:, 16]

    output[:, 12] = sequence[:, 4]
    output[:, 13] = sequence[:, 5]
    output[:, 14] = sequence[:, 6]
    output[:, 15] = sequence[:, 6]

    output[:, 16] = sequence[:, 1]
    output[:, 17] = sequence[:, 2]
    output[:, 18] = sequence[:, 3]
    output[:, 19] = sequence[:, 3]

    return output


EXPECTED_MAPPING = {
    0: 0, 1: 7, 20: 8, 2: 9, 3: 10,
    4: 11, 5: 12, 6: 13, 7: 13, 21: 13, 22: 13,
    8: 14, 9: 15, 10: 16, 11: 16, 23: 16, 24: 16,
    12: 4, 13: 5, 14: 6, 15: 6,
    16: 1, 17: 2, 18: 3, 19: 3,
}


def check_mapping_exact(
    h36m_sequence: np.ndarray,
    ntu25_sequence: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    max_error = 0.0

    for ntu_idx, h36m_idx in EXPECTED_MAPPING.items():
        error = float(
            np.max(
                np.abs(
                    ntu25_sequence[:, ntu_idx]
                    - h36m_sequence[:, h36m_idx]
                )
            )
        )
        max_error = max(max_error, error)

    passed = max_error <= tolerance
    status = "PASS" if passed else "FAIL"
    print(
        f"[{status}] Mapping-copy check | "
        f"maximum absolute error = {max_error:.8g}"
    )
    return passed


def print_basic_checks(
    h36m_sequence: np.ndarray,
    ntu25_sequence: np.ndarray,
    centered: bool,
) -> None:
    print(f"  H36M shape: {h36m_sequence.shape}")
    print(f"  NTU25 shape: {ntu25_sequence.shape}")
    print(f"  H36M finite: {np.isfinite(h36m_sequence).all()}")
    print(f"  NTU25 finite: {np.isfinite(ntu25_sequence).all()}")

    if centered:
        pelvis_error = float(
            np.max(np.abs(h36m_sequence[:, 0, :]))
        )
        print(
            f"  Centered pelvis max |coordinate|: "
            f"{pelvis_error:.8g}"
        )

    check_mapping_exact(h36m_sequence, ntu25_sequence)

    median_lengths = []
    edge_rows = []

    for start, end in NTU25_EDGES:
        lengths = np.linalg.norm(
            ntu25_sequence[:, start] - ntu25_sequence[:, end],
            axis=-1,
        )
        median_length = float(np.median(lengths))
        median_lengths.append(median_length)
        edge_rows.append((start, end, median_length))

    nonzero = [value for value in median_lengths if value > 1e-8]
    typical = float(np.median(nonzero)) if nonzero else 0.0

    suspicious = [
        (start, end, length)
        for start, end, length in edge_rows
        if typical > 0 and length > typical * 3.0
    ]

    print(
        f"  Typical non-zero NTU graph edge length: "
        f"{typical:.6g}"
    )

    if suspicious:
        print("  WARNING: unusually long graph edges:")
        for start, end, length in suspicious:
            print(
                f"    {start:2d} {NTU25_NAMES[start]:16s} -> "
                f"{end:2d} {NTU25_NAMES[end]:16s}: "
                f"{length:.6g}"
            )
    else:
        print("  No unusually long NTU graph edges found.")


def safe_filename(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:160] or "sample"


def get_global_limits(
    sequences: Iterable[np.ndarray],
    padding: float = 0.08,
) -> tuple[np.ndarray, float]:
    points = []

    for sequence in sequences:
        array = np.asarray(sequence)
        points.append(array.reshape(-1, 3))

    all_points = np.concatenate(points, axis=0)
    all_points = all_points[np.isfinite(all_points).all(axis=1)]

    if len(all_points) == 0:
        raise ValueError("No finite 3D points available for plotting.")

    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = float(np.max(maximum - minimum) / 2.0)

    if radius < 1e-8:
        radius = 1.0

    radius *= 1.0 + padding
    return center, radius


def apply_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def draw_skeleton(
    ax,
    points: np.ndarray,
    edges: Sequence[tuple[int, int]],
    title: str,
    annotate: bool,
    elev: float,
    azim: float,
) -> None:
    points = np.asarray(points)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=18,
    )

    for start, end in edges:
        ax.plot(
            [points[start, 0], points[end, 0]],
            [points[start, 1], points[end, 1]],
            [points[start, 2], points[end, 2]],
            linewidth=1.3,
        )

    if annotate:
        for joint_index, point in enumerate(points):
            ax.text(
                point[0],
                point[1],
                point[2],
                str(joint_index),
                fontsize=7,
            )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)


def save_raw_comparison_png(
    h36m_sequence: np.ndarray,
    ntu25_sequence: np.ndarray,
    output_path: Path,
    num_frames: int,
    annotate: bool,
    elev: float,
    azim: float,
) -> None:
    total_frames = len(h36m_sequence)
    count = min(max(num_frames, 1), total_frames)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        count,
        dtype=int,
    )

    center, radius = get_global_limits(
        [h36m_sequence, ntu25_sequence]
    )

    fig = plt.figure(figsize=(12, 4 * count))

    for row, frame_index in enumerate(frame_indices):
        ax_h36m = fig.add_subplot(
            count,
            2,
            row * 2 + 1,
            projection="3d",
        )
        ax_ntu = fig.add_subplot(
            count,
            2,
            row * 2 + 2,
            projection="3d",
        )

        draw_skeleton(
            ax_h36m,
            h36m_sequence[frame_index],
            H36M_EDGES,
            f"H36M-17 | frame {frame_index}",
            annotate,
            elev,
            azim,
        )
        draw_skeleton(
            ax_ntu,
            ntu25_sequence[frame_index],
            NTU25_EDGES,
            f"Converted NTU-25 | frame {frame_index}",
            annotate,
            elev,
            azim,
        )

        apply_limits(ax_h36m, center, radius)
        apply_limits(ax_ntu, center, radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_raw_comparison_gif(
    h36m_sequence: np.ndarray,
    ntu25_sequence: np.ndarray,
    output_path: Path,
    frame_step: int,
    fps: int,
    elev: float,
    azim: float,
) -> None:
    frame_indices = list(
        range(0, len(h36m_sequence), max(frame_step, 1))
    )

    center, radius = get_global_limits(
        [h36m_sequence, ntu25_sequence]
    )

    fig = plt.figure(figsize=(12, 6))
    ax_h36m = fig.add_subplot(1, 2, 1, projection="3d")
    ax_ntu = fig.add_subplot(1, 2, 2, projection="3d")

    def update(frame_index: int):
        ax_h36m.clear()
        ax_ntu.clear()

        draw_skeleton(
            ax_h36m,
            h36m_sequence[frame_index],
            H36M_EDGES,
            f"H36M-17 | frame {frame_index}",
            False,
            elev,
            azim,
        )
        draw_skeleton(
            ax_ntu,
            ntu25_sequence[frame_index],
            NTU25_EDGES,
            f"Converted NTU-25 | frame {frame_index}",
            False,
            elev,
            azim,
        )

        apply_limits(ax_h36m, center, radius)
        apply_limits(ax_ntu, center, radius)
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    animation.save(
        output_path,
        writer=PillowWriter(fps=max(fps, 1)),
    )
    plt.close(fig)


def save_packed_clip_png(
    clip: np.ndarray,
    output_path: Path,
    num_frames: int,
    annotate: bool,
    elev: float,
    azim: float,
) -> None:
    clip = np.asarray(clip)

    if clip.ndim != 4 or clip.shape[0] != 3 or clip.shape[2] != 25:
        raise ValueError(
            "Packed clip must have shape (3, T, 25, M); "
            f"received {clip.shape}"
        )

    if clip.shape[3] < 1:
        raise ValueError("Packed clip has no person dimension.")

    sequence = clip[:, :, :, 0].transpose(1, 2, 0)

    total_frames = len(sequence)
    count = min(max(num_frames, 1), total_frames)
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        count,
        dtype=int,
    )

    center, radius = get_global_limits([sequence])
    fig = plt.figure(figsize=(7, 4 * count))

    for row, frame_index in enumerate(frame_indices):
        ax = fig.add_subplot(
            count,
            1,
            row + 1,
            projection="3d",
        )

        draw_skeleton(
            ax,
            sequence[frame_index],
            NTU25_EDGES,
            f"Packed CTR-GCN clip | frame {frame_index}",
            annotate,
            elev,
            azim,
        )
        apply_limits(ax, center, radius)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def select_indices(
    total_count: int,
    num_samples: int,
    sample_index: int | None,
    seed: int,
) -> list[int]:
    if total_count <= 0:
        raise ValueError("Dataset is empty.")

    if sample_index is not None:
        if sample_index < 0 or sample_index >= total_count:
            raise IndexError(
                f"sample-index {sample_index} is outside "
                f"[0, {total_count - 1}]"
            )
        return [sample_index]

    rng = np.random.default_rng(seed)
    count = min(max(num_samples, 1), total_count)

    return sorted(
        rng.choice(
            total_count,
            size=count,
            replace=False,
        ).tolist()
    )


def inspect_npz(args: argparse.Namespace, output_dir: Path) -> None:
    raw_data = np.load(args.npz, allow_pickle=True)
    names = list(raw_data.keys())

    indices = select_indices(
        len(names),
        args.num_samples,
        args.sample_index,
        args.seed,
    )

    print(f"\nLoaded raw NPZ: {args.npz}")
    print(f"Sequence count: {len(names)}")
    print(f"Selected indices: {indices}")

    for position, dataset_index in enumerate(indices):
        name = names[dataset_index]
        raw_sequence = np.asarray(
            raw_data[name],
            dtype=np.float32,
        )

        if raw_sequence.ndim != 3 or raw_sequence.shape[1:] != (17, 3):
            print(
                f"\n[SKIP] {name}: unexpected shape "
                f"{raw_sequence.shape}"
            )
            continue

        processed = (
            center_poses(raw_sequence)
            if args.center
            else raw_sequence.copy()
        )

        converted = convert_h36m_to_ntu25(processed)

        print("\n" + "=" * 78)
        print(
            f"Raw sample {position + 1}/{len(indices)} | "
            f"dataset index={dataset_index} | name={name}"
        )

        print_basic_checks(
            processed,
            converted,
            centered=args.center,
        )

        stem = (
            f"raw_{position:02d}_idx_{dataset_index}_"
            f"{safe_filename(name)}"
        )

        png_path = output_dir / f"{stem}.png"

        save_raw_comparison_png(
            processed,
            converted,
            png_path,
            num_frames=args.frames,
            annotate=args.annotate,
            elev=args.elev,
            azim=args.azim,
        )
        print(f"  Saved PNG: {png_path}")

        if args.make_gif:
            gif_path = output_dir / f"{stem}.gif"
            save_raw_comparison_gif(
                processed,
                converted,
                gif_path,
                frame_step=args.frame_step,
                fps=args.fps,
                elev=args.elev,
                azim=args.azim,
            )
            print(f"  Saved GIF: {gif_path}")


def inspect_pkl(args: argparse.Namespace, output_dir: Path) -> None:
    with open(args.pkl, "rb") as file:
        packed = pickle.load(file)

    required_keys = {"pose", "label", "video_name"}
    missing = required_keys.difference(packed)

    if missing:
        raise KeyError(
            f"Packed PKL is missing required keys: {sorted(missing)}"
        )

    poses = packed["pose"]
    labels = packed["label"]
    names = packed["video_name"]

    if not (len(poses) == len(labels) == len(names)):
        raise ValueError(
            "Packed PKL arrays have inconsistent lengths: "
            f"pose={len(poses)}, label={len(labels)}, "
            f"video_name={len(names)}"
        )

    indices = select_indices(
        len(poses),
        args.num_samples,
        args.sample_index,
        args.seed,
    )

    print(f"\nLoaded packed PKL: {args.pkl}")
    print(f"Clip count: {len(poses)}")
    print(f"Selected indices: {indices}")

    for position, dataset_index in enumerate(indices):
        clip = np.asarray(poses[dataset_index])
        label = labels[dataset_index]
        name = names[dataset_index]

        print("\n" + "=" * 78)
        print(
            f"Packed sample {position + 1}/{len(indices)} | "
            f"dataset index={dataset_index} | "
            f"name={name} | label={label}"
        )
        print(f"  Clip shape: {clip.shape}")
        print(f"  Finite: {np.isfinite(clip).all()}")

        stem = (
            f"packed_{position:02d}_idx_{dataset_index}_"
            f"label_{label}_{safe_filename(name)}"
        )
        png_path = output_dir / f"{stem}.png"

        save_packed_clip_png(
            clip,
            png_path,
            num_frames=args.frames,
            annotate=args.annotate,
            elev=args.elev,
            azim=args.azim,
        )
        print(f"  Saved PNG: {png_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone CARE-PD H36M-to-NTU25 visual checker. "
            "It never imports or runs the training script."
        )
    )

    source_group = parser.add_mutually_exclusive_group(required=True)

    source_group.add_argument(
        "--npz",
        type=str,
        help="Path to raw CARE-PD H36M NPZ.",
    )

    source_group.add_argument(
        "--pkl",
        type=str,
        help="Path to a packed CTR-GCN PKL.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="skeleton_checks",
        help="Directory used for saved PNG/GIF files.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of random samples to inspect.",
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Inspect exactly this dataset index.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting samples.",
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of static frames shown in each PNG.",
    )

    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For raw NPZ input, subtract the pelvis from every frame.",
    )

    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Draw joint indices next to skeleton points.",
    )

    parser.add_argument(
        "--make-gif",
        action="store_true",
        help="Also save side-by-side raw/conversion GIFs.",
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Use every Nth frame when generating GIFs.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="GIF playback frame rate.",
    )

    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="Matplotlib 3D elevation angle.",
    )

    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Matplotlib 3D azimuth angle.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.npz:
        if not os.path.isfile(args.npz):
            raise FileNotFoundError(f"NPZ file not found: {args.npz}")
        inspect_npz(args, output_dir)

    elif args.pkl:
        if not os.path.isfile(args.pkl):
            raise FileNotFoundError(f"PKL file not found: {args.pkl}")
        inspect_pkl(args, output_dir)

    print("\nDone. No model was created and no training code was executed.")


if __name__ == "__main__":
    main()