"""GPU worker for deterministic dense RoMa nasal correspondences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--coarse-resolution", type=int, default=560)
    parser.add_argument("--upsample-resolution", type=int, default=560)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def run_worker(args: argparse.Namespace) -> None:
    os.environ["TORCH_HOME"] = str(args.torch_home.resolve())
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from romatch import roma_outdoor

    if not torch.cuda.is_available():
        raise RuntimeError("RoMa nasal worker requires CUDA")
    if int(args.stride) < 1:
        raise ValueError("stride must be positive")
    front_path = args.front.resolve()
    side_path = args.side.resolve()
    output_path = args.output.resolve()
    front_image = Image.open(front_path).convert("RGB")
    side_image = Image.open(side_path).convert("RGB")
    front_width, front_height = front_image.size
    side_width, side_height = side_image.size

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    matcher = roma_outdoor(
        device="cuda",
        coarse_res=int(args.coarse_resolution),
        upsample_res=int(args.upsample_resolution),
    )
    warp, certainty = matcher.match(
        str(front_path),
        str(side_path),
        device="cuda",
    )
    warp = warp[0]
    certainty = certainty[0]
    half_width = warp.shape[1] // 2
    if half_width * 2 != warp.shape[1]:
        raise RuntimeError("unexpected non-symmetric RoMa warp")

    forward = warp[:, :half_width]
    reverse = warp[:, half_width:]
    forward_certainty = certainty[:, :half_width]
    reverse_certainty = certainty[:, half_width:]
    front_coords = forward[..., :2]
    side_coords = forward[..., 2:]
    reverse_front = functional.grid_sample(
        reverse[..., :2].permute(2, 0, 1)[None],
        side_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0].permute(1, 2, 0)
    reverse_confidence = functional.grid_sample(
        reverse_certainty[None, None],
        side_coords[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    combined_certainty = torch.sqrt(
        torch.clamp(forward_certainty * reverse_confidence, min=0.0, max=1.0)
    )
    cycle_delta = reverse_front - front_coords
    cycle_error = torch.sqrt(
        (cycle_delta[..., 0] * (front_width / 2.0)) ** 2
        + (cycle_delta[..., 1] * (front_height / 2.0)) ** 2
    )

    stride = int(args.stride)
    front_coords = front_coords[::stride, ::stride]
    side_coords = side_coords[::stride, ::stride]
    combined_certainty = combined_certainty[::stride, ::stride]
    cycle_error = cycle_error[::stride, ::stride]
    front_pixels = torch.stack(
        (
            front_width / 2.0 * (front_coords[..., 0] + 1.0),
            front_height / 2.0 * (front_coords[..., 1] + 1.0),
        ),
        dim=-1,
    )
    side_pixels = torch.stack(
        (
            side_width / 2.0 * (side_coords[..., 0] + 1.0),
            side_height / 2.0 * (side_coords[..., 1] + 1.0),
        ),
        dim=-1,
    )
    valid = (
        torch.isfinite(front_pixels).all(dim=-1)
        & torch.isfinite(side_pixels).all(dim=-1)
        & torch.isfinite(combined_certainty)
        & torch.isfinite(cycle_error)
        & (combined_certainty > 0.0)
    )
    metadata = {
        "matcher": "roma_outdoor",
        "coarse_resolution": int(args.coarse_resolution),
        "upsample_resolution": int(args.upsample_resolution),
        "stride": stride,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1048576.0),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        front_pixels=front_pixels[valid].detach().cpu().numpy().astype(np.float32),
        side_pixels=side_pixels[valid].detach().cpu().numpy().astype(np.float32),
        certainty=combined_certainty[valid].detach().cpu().numpy().astype(np.float32),
        cycle_error_px=cycle_error[valid].detach().cpu().numpy().astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    run_worker(_parse_args())


if __name__ == "__main__":
    main()
