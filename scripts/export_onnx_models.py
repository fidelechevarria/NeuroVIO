#!/usr/bin/env python3
"""
NeuroVIO: Robust ONNX Exporter for SuperPoint and LightGlue.
Uses legacy torch.onnx.export (dynamo=False) for maximum compatibility with ONNX Runtime C++.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointBackbone(nn.Module):
    """Clean self-contained SuperPoint PyTorch model for robust ONNX export."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        # Shared Encoder
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Detector
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # [B, 65, H/8, W/8]

        # Softmax & reshape to full resolution score map
        prob = F.softmax(semi, dim=1)[:, :-1, :, :]  # remove dustbin -> [B, 64, H/8, W/8]
        prob = F.pixel_shuffle(prob, 8)               # [B, 1, H, W]

        # Descriptors
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)                       # [B, 256, H/8, W/8]
        desc = F.normalize(desc, p=2, dim=1)          # L2 normalize

        return prob, desc


def export_superpoint(output_path: str, height: int = 480, width: int = 752):
    print(f"[NeuroVIO Exporter] Exporting SuperPoint to {output_path}...")
    from lightglue import SuperPoint

    orig_sp = SuperPoint(max_num_keypoints=512).eval()
    model = SuperPointBackbone().eval()

    # Load weights from LightGlue's checkpoint
    state_dict = orig_sp.state_dict()
    backbone_state = {}
    for k, v in state_dict.items():
        backbone_state[k] = v
    model.load_state_dict(backbone_state, strict=False)

    dummy_input = torch.randn(1, 1, height, width, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["scores", "descriptors"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "scores": {0: "batch", 2: "height", 3: "width"},
            "descriptors": {0: "batch", 2: "desc_h", 3: "desc_w"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"[NeuroVIO Exporter] Exported SuperPoint -> {output_path}")


def export_lightglue(output_path: str, extractor_type: str = "superpoint"):
    print(f"[NeuroVIO Exporter] Exporting LightGlue to {output_path}...")
    from lightglue import LightGlue

    # Static execution settings for clean ONNX export
    matcher = LightGlue(
        features=extractor_type,
        depth_confidence=-1,
        width_confidence=-1,
        flash=False,
    ).eval()

    class SimpleLightGlue(nn.Module):
        def __init__(self, lg):
            super().__init__()
            self.lg = lg

        def forward(self, kpts0, desc0, kpts1, desc1):
            data = {
                "image0": {"keypoints": kpts0, "descriptors": desc0},
                "image1": {"keypoints": kpts1, "descriptors": desc1},
            }
            res = self.lg(data)
            return res["matches0"], res["matching_scores0"]

    model = SimpleLightGlue(matcher).eval()

    N0, N1, D = 256, 256, 256
    dummy_kpts0 = torch.randn(1, N0, 2)
    dummy_desc0 = torch.randn(1, N0, D)
    dummy_kpts1 = torch.randn(1, N1, 2)
    dummy_desc1 = torch.randn(1, N1, D)

    dim_n0 = torch.export.Dim("num_kpts0", min=1, max=4096)
    dim_n1 = torch.export.Dim("num_kpts1", min=1, max=4096)

    dynamic_shapes = {
        "kpts0": {1: dim_n0},
        "desc0": {1: dim_n0},
        "kpts1": {1: dim_n1},
        "desc1": {1: dim_n1},
    }

    torch.onnx.export(
        model,
        (dummy_kpts0, dummy_desc0, dummy_kpts1, dummy_desc1),
        output_path,
        input_names=["kpts0", "desc0", "kpts1", "desc1"],
        output_names=["matches0", "mscores0"],
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        dynamo=True,
    )
    print(f"[NeuroVIO Exporter] Exported LightGlue -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NeuroVIO ONNX Exporter")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=752)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sp_path = os.path.join(args.output_dir, "superpoint.onnx")
    lg_path = os.path.join(args.output_dir, "lightglue_superpoint.onnx")

    export_superpoint(sp_path, height=args.height, width=args.width)
    export_lightglue(lg_path)
    print("All models exported successfully to ONNX!")


if __name__ == "__main__":
    main()
