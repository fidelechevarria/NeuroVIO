#!/usr/bin/env python3
"""
NeuroVIO Automated Evaluation Script using EVO
Computes Absolute Trajectory Error (ATE RMSE) against Ground Truth.
"""

import argparse
import os
import subprocess
import sys
import numpy as np


def convert_euroc_gt_to_tum(euroc_gt_csv: str, output_tum: str):
    """Converts EuRoC state_groundtruth_estimate0/data.csv to standard TUM format."""
    print(f"[NeuroVIO Evaluator] Converting EuRoC GT {euroc_gt_csv} -> {output_tum}")
    data = []
    with open(euroc_gt_csv, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            t_sec = float(parts[0]) * 1e-9
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            data.append(f"{t_sec:.6f} {px:.6f} {py:.6f} {pz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    with open(output_tum, "w") as f:
        f.writelines(data)
    print(f"[NeuroVIO Evaluator] Wrote {len(data)} GT poses to {output_tum}")


def evaluate_trajectory(estimate_path: str, gt_path: str, plot: bool = False):
    if not os.path.exists(estimate_path):
        print(f"[Error] Estimate trajectory file not found: {estimate_path}", file=sys.stderr)
        return 1

    temp_gt_tum = None
    if gt_path.endswith(".csv"):
        temp_gt_tum = gt_path.replace(".csv", "_tum.txt")
        convert_euroc_gt_to_tum(gt_path, temp_gt_tum)
        gt_tum_path = temp_gt_tum
    else:
        gt_tum_path = gt_path

    cmd = [
        "evo_ape", "tum",
        gt_tum_path,
        estimate_path,
        "-r", "full",
        "-a",  # 6-DOF SE(3) Umeyama alignment
        "-v"
    ]
    if plot:
        cmd.extend(["-p", "--plot_mode", "xyz"])

    print(f"[NeuroVIO Evaluator] Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return 0
    except FileNotFoundError:
        print("[Warning] 'evo_ape' not found. Please install evo via 'pip install evo'.", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"[Error] evo evaluation failed:\n{e.stderr}", file=sys.stderr)
        return e.returncode
    finally:
        if temp_gt_tum and os.path.exists(temp_gt_tum):
            try:
                os.remove(temp_gt_tum)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="NeuroVIO Trajectory Evaluator")
    parser.add_argument("--estimate", required=True, help="Path to estimated trajectory (TUM format)")
    parser.add_argument("--groundtruth", required=True, help="Path to groundtruth (TUM or EuRoC CSV format)")
    parser.add_argument("--plot", action="store_true", help="Display matplotlib 3D trajectory comparison plot")
    args = parser.parse_args()

    sys.exit(evaluate_trajectory(args.estimate, args.groundtruth, args.plot))


if __name__ == "__main__":
    main()
