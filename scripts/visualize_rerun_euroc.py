#!/usr/bin/env python3
import sys
import os
import pathlib
import time
import argparse
import csv
import numpy as np
import cv2

# Set path for rerun binary
venv_bin = str(pathlib.Path(sys.executable).parent)
if venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

import rerun as rr

def parse_args():
    parser = argparse.ArgumentParser(description="Live 3D Trajectory & Ground Truth Visualizer for NeuroVIO in Rerun.io.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/media/disk1/euroc_mav_dataset/machine_hall/MH_01_easy",
        help="Path to EuRoC MAV sequence folder"
    )
    parser.add_argument(
        "--trajectory-path",
        type=str,
        default="trajectory_estimate.tum",
        help="Path to estimated TUM trajectory file"
    )
    parser.add_argument("--fps-limit", type=float, default=30.0, help="Playback speed FPS limit (0 for unlimited)")
    return parser.parse_args()

def load_ground_truth(dataset_path):
    gt_csv = pathlib.Path(dataset_path) / "mav0" / "state_groundtruth_estimate0" / "data.csv"
    if not gt_csv.exists():
        gt_csv = pathlib.Path(dataset_path) / "state_groundtruth_estimate0" / "data.csv"
    if not gt_csv.exists():
        return np.array([]), np.array([]), np.array([])

    timestamps = []
    positions = []
    quaternions = []

    with open(gt_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts_sec = float(row[0]) * 1e-9
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            timestamps.append(ts_sec)
            positions.append([px, py, pz])
            quaternions.append([qw, qx, qy, qz])

    return np.array(timestamps), np.array(positions), np.array(quaternions)

def load_tum_trajectory(tum_path):
    if not pathlib.Path(tum_path).exists():
        return np.array([]), np.array([]), np.array([])

    timestamps = []
    positions = []
    quaternions = []

    with open(tum_path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                ts = float(parts[0])
                p = [float(parts[1]), float(parts[2]), float(parts[3])]
                q = [float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])]
                timestamps.append(ts)
                positions.append(p)
                quaternions.append(q)

    return np.array(timestamps), np.array(positions), np.array(quaternions)

def umeyama_alignment(model, data, with_scale=True):
    """
    Standard Umeyama SE(3)/Sim(3) 6-DoF closed-form least squares alignment.
    Aligns estimated trajectory 'data' to Ground Truth 'model'.
    """
    n = min(len(model), len(data))
    if n < 3:
        return data

    m = model[:n]
    d = data[:n]

    mu_m = m.mean(axis=0)
    mu_d = d.mean(axis=0)

    cov = (d - mu_d).T @ (m - mu_m) / n
    var_d = np.var(d, axis=0).sum()

    u, singular, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1

    R = vt.T @ s @ u.T
    scale = (1.0 / var_d * np.trace(np.diag(singular) @ s)) if with_scale else 1.0
    t = mu_m - scale * (R @ mu_d)

    aligned_data = scale * (data @ R.T) + t
    return aligned_data

def main():
    args = parse_args()

    print("================================================================")
    print("  NeuroVIO: 3D Trajectory & Ground Truth Visualizer (Rerun.io)  ")
    print("================================================================")
    print(f"Dataset Path:       {args.dataset_path}")
    print(f"TUM Trajectory:     {args.trajectory_path}\n")

    gt_ts, gt_pos, gt_quat = load_ground_truth(args.dataset_path)
    est_ts, est_pos, est_quat = load_tum_trajectory(args.trajectory_path)

    print(f"Loaded {len(gt_pos)} Ground Truth poses.")
    print(f"Loaded {len(est_pos)} Estimated VIO poses.\n")

    if len(est_pos) > 0 and len(gt_pos) > 0:
        # Interpolate ground truth to match estimate timestamps
        gt_interp_pos = np.zeros_like(est_pos)
        for i, ts in enumerate(est_ts):
            idx = np.searchsorted(gt_ts, ts)
            if idx == 0:
                gt_interp_pos[i] = gt_pos[0]
            elif idx >= len(gt_ts):
                gt_interp_pos[i] = gt_pos[-1]
            else:
                alpha = (ts - gt_ts[idx - 1]) / max(gt_ts[idx] - gt_ts[idx - 1], 1e-6)
                gt_interp_pos[i] = (1 - alpha) * gt_pos[idx - 1] + alpha * gt_pos[idx]

        est_pos_aligned = umeyama_alignment(gt_interp_pos, est_pos, with_scale=True)
    else:
        est_pos_aligned = est_pos

    rr.init("NeuroVIO_3D_Trajectory_Viewer", spawn=True)

    # Log complete Ground Truth 3D path
    if len(gt_pos) > 0:
        rr.log(
            "world/ground_truth/full_path",
            rr.LineStrips3D([gt_pos], colors=[[0, 255, 120]], radii=0.03),
            static=True
        )

    # Log complete NeuroVIO estimated 3D path
    if len(est_pos_aligned) > 0:
        rr.log(
            "world/neurovio/full_path",
            rr.LineStrips3D([est_pos_aligned], colors=[[0, 180, 255]], radii=0.035),
            static=True
        )

    cam_dir = pathlib.Path(args.dataset_path) / "mav0" / "cam0"
    if not cam_dir.exists():
        cam_dir = pathlib.Path(args.dataset_path) / "cam0"

    cam_csv = cam_dir / "data.csv"
    cam_data = []
    if cam_csv.exists():
        with open(cam_csv, "r") as f:
            for row in csv.reader(f):
                if row and not row[0].startswith("#"):
                    cam_data.append((float(row[0]) * 1e-9, cam_dir / "data" / row[1].strip()))

    print(f"Streaming {len(cam_data)} camera frames with live 3D pose tracking to Rerun...")

    est_idx = 0
    gt_idx = 0
    accumulated_est_path = []

    for f_idx, (ts_sec, img_path) in enumerate(cam_data):
        rr.set_time("timestamp", timestamp=ts_sec)
        rr.set_time("frame_idx", sequence=f_idx)

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rr.log("camera/image", rr.Image(img_rgb))

        while gt_idx + 1 < len(gt_ts) and gt_ts[gt_idx + 1] <= ts_sec:
            gt_idx += 1

        if len(gt_pos) > gt_idx:
            curr_gt_p = gt_pos[gt_idx]
            rr.log("world/ground_truth/drone_pose", rr.Points3D([curr_gt_p], colors=[[0, 255, 120]], radii=0.15))

        while est_idx + 1 < len(est_ts) and est_ts[est_idx + 1] <= ts_sec:
            est_idx += 1

        if len(est_pos_aligned) > est_idx:
            curr_est_p = est_pos_aligned[est_idx]
            accumulated_est_path.append(curr_est_p)
            rr.log("world/neurovio/drone_pose", rr.Points3D([curr_est_p], colors=[[0, 180, 255]], radii=0.18))

            if len(gt_pos) > gt_idx:
                err = np.linalg.norm(curr_gt_p - curr_est_p)
                rr.log("metrics/ate_position_error_m", rr.Scalars(err))

        if f_idx % 200 == 0 or f_idx == len(cam_data) - 1:
            print(f"[{f_idx}/{len(cam_data)}] Timestamp: {ts_sec:.3f} s | Logged to Rerun.io")

        if args.fps_limit > 0:
            time.sleep(1.0 / args.fps_limit)

    print("\nRerun 3D trajectory playback complete!")

if __name__ == "__main__":
    main()
