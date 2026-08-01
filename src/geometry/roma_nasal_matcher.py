"""Isolated RoMa adapter for dense cross-view nasal correspondences."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _readonly(value: Any, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


@dataclass(frozen=True)
class NasalCropTransform:
    """Exact mapping between a work image and a rectangular nasal crop."""

    work_size: tuple[int, int]
    bounds_xyxy: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        width, height = (int(value) for value in self.work_size)
        x0, y0, x1, y1 = (int(value) for value in self.bounds_xyxy)
        if width <= 0 or height <= 0:
            raise ValueError("work_size must contain positive dimensions")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("crop bounds must lie inside the work image")
        object.__setattr__(self, "work_size", (width, height))
        object.__setattr__(self, "bounds_xyxy", (x0, y0, x1, y1))

    @property
    def crop_size(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bounds_xyxy
        return x1 - x0, y1 - y0

    def crop_to_work(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError("crop points must end in shape (2,)")
        result = np.array(values, copy=True)
        result[..., 0] += self.bounds_xyxy[0]
        result[..., 1] += self.bounds_xyxy[1]
        return result

    def work_to_crop(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError("work points must end in shape (2,)")
        result = np.array(values, copy=True)
        result[..., 0] -= self.bounds_xyxy[0]
        result[..., 1] -= self.bounds_xyxy[1]
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_size": list(self.work_size),
            "bounds_xyxy": list(self.bounds_xyxy),
            "crop_size": list(self.crop_size),
        }


@dataclass(frozen=True)
class RoMaNasalMatchBatch:
    front_pixels: np.ndarray
    side_pixels: np.ndarray
    certainty: np.ndarray
    cycle_error_px: np.ndarray
    front_crop: NasalCropTransform
    side_crop: NasalCropTransform
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        front = np.asarray(self.front_pixels, dtype=np.float64)
        side = np.asarray(self.side_pixels, dtype=np.float64)
        certainty = np.asarray(self.certainty, dtype=np.float64).reshape(-1)
        cycle = np.asarray(self.cycle_error_px, dtype=np.float64).reshape(-1)
        if front.ndim != 2 or front.shape[1] != 2:
            raise ValueError("front_pixels must have shape (N, 2)")
        if side.shape != front.shape:
            raise ValueError("side_pixels must match front_pixels")
        if certainty.shape != (len(front),) or cycle.shape != (len(front),):
            raise ValueError("certainty and cycle_error_px must have shape (N,)")
        if not (
            np.isfinite(front).all()
            and np.isfinite(side).all()
            and np.isfinite(certainty).all()
            and np.isfinite(cycle).all()
        ):
            raise ValueError("RoMa match batch contains non-finite values")
        if np.any((certainty < 0.0) | (certainty > 1.0)):
            raise ValueError("RoMa certainty must lie in [0, 1]")
        if np.any(cycle < 0.0):
            raise ValueError("cycle errors must be non-negative")
        object.__setattr__(self, "front_pixels", _readonly(front, np.float64))
        object.__setattr__(self, "side_pixels", _readonly(side, np.float64))
        object.__setattr__(self, "certainty", _readonly(certainty, np.float64))
        object.__setattr__(self, "cycle_error_px", _readonly(cycle, np.float64))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RoMaNasalSelectionConfig:
    min_learned_certainty: float = 0.12
    max_cycle_error_px: float = 2.5
    max_epipolar_error_px: float = 2.0
    max_front_seed_distance_px: float = 8.0
    max_side_prior_distance_px: float = 48.0
    min_combined_confidence: float = 0.10
    duplicate_radius_px: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "min_learned_certainty",
            "max_cycle_error_px",
            "max_epipolar_error_px",
            "max_front_seed_distance_px",
            "max_side_prior_distance_px",
            "min_combined_confidence",
            "duplicate_radius_px",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.min_learned_certainty) > 1.0:
            raise ValueError("min_learned_certainty must not exceed one")
        if float(self.min_combined_confidence) > 1.0:
            raise ValueError("min_combined_confidence must not exceed one")


def _sample_map(values: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    image = np.asarray(values)
    points = np.asarray(pixels, dtype=np.float64)
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
    return np.asarray(image[y, x])


def select_roma_nasal_matches(
    batch: RoMaNasalMatchBatch,
    *,
    side_view: str,
    seeds: Sequence[Any],
    front_confidence: Any,
    side_confidence: Any,
    provenance_by_view: Mapping[str, Any],
    fundamental: np.ndarray,
    config: RoMaNasalSelectionConfig | None = None,
) -> Any:
    """Attach dense RoMa proposals to semantic seeds under fixed-rig geometry."""
    from src.geometry.nasal_texture_observations import (
        NasalEpipolarMatchResult,
        NasalEpipolarSeed,
        NasalPairMatch,
        RejectedNasalMatch,
    )

    cfg = config or RoMaNasalSelectionConfig()
    seed_values = tuple(seeds)
    if not all(isinstance(value, NasalEpipolarSeed) for value in seed_values):
        raise ValueError("seeds contain an invalid record")
    matrix = np.asarray(fundamental, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("fundamental must have finite shape (3, 3)")
    if set(provenance_by_view) != {"front", side_view}:
        raise ValueError("provenance must contain front and side_view")

    front_semantic = _sample_map(
        front_confidence.semantic_support,
        batch.front_pixels,
    ).astype(bool)
    side_semantic = _sample_map(
        side_confidence.semantic_support,
        batch.side_pixels,
    ).astype(bool)
    texture = np.sqrt(
        np.clip(
            _sample_map(front_confidence.texture_strength, batch.front_pixels),
            0.0,
            1.0,
        )
        * np.clip(
            _sample_map(side_confidence.texture_strength, batch.side_pixels),
            0.0,
            1.0,
        )
    )
    difficult = (
        _sample_map(front_confidence.specular_reject, batch.front_pixels).astype(bool)
        | _sample_map(side_confidence.specular_reject, batch.side_pixels).astype(bool)
        | _sample_map(front_confidence.shadow_reject, batch.front_pixels).astype(bool)
        | _sample_map(side_confidence.shadow_reject, batch.side_pixels).astype(bool)
    )
    image_reliability = (0.30 + 0.70 * texture) * np.where(difficult, 0.25, 1.0)
    cycle_factor = np.exp(
        -0.5
        * (
            batch.cycle_error_px
            / max(float(cfg.max_cycle_error_px), 1e-6)
        )
        ** 2
    )
    combined = np.clip(batch.certainty * image_reliability * cycle_factor, 0.0, 1.0)
    globally_valid = (
        front_semantic
        & side_semantic
        & (batch.certainty >= float(cfg.min_learned_certainty))
        & (batch.cycle_error_px <= float(cfg.max_cycle_error_px))
    )

    matches = []
    rejected = []
    used_indices: list[int] = []
    for seed in seed_values:
        front_distance = np.linalg.norm(batch.front_pixels - seed.front_pixel, axis=1)
        side_distance = np.linalg.norm(
            batch.side_pixels - seed.predicted_side_pixel,
            axis=1,
        )
        line = matrix @ np.append(seed.front_pixel, 1.0)
        line_norm = max(float(np.linalg.norm(line[:2])), 1e-12)
        epipolar_error = np.abs(
            np.column_stack((batch.side_pixels, np.ones(len(batch.side_pixels))))
            @ line
        ) / line_norm
        eligible = np.flatnonzero(
            globally_valid
            & (front_distance <= float(cfg.max_front_seed_distance_px))
            & (side_distance <= float(cfg.max_side_prior_distance_px))
            & (epipolar_error <= float(cfg.max_epipolar_error_px))
            & (combined >= float(cfg.min_combined_confidence))
        )
        if len(eligible):
            ranking = combined[eligible]
            ranking *= np.exp(
                -0.5
                * (front_distance[eligible] / float(cfg.max_front_seed_distance_px)) ** 2
            )
            ranking *= np.exp(
                -0.5
                * (epipolar_error[eligible] / float(cfg.max_epipolar_error_px)) ** 2
            )
            order = eligible[np.argsort(-ranking, kind="stable")]
        else:
            order = np.empty(0, dtype=np.int64)
        selected = None
        for index in order:
            if any(
                np.linalg.norm(batch.front_pixels[index] - batch.front_pixels[used])
                < float(cfg.duplicate_radius_px)
                or np.linalg.norm(batch.side_pixels[index] - batch.side_pixels[used])
                < float(cfg.duplicate_radius_px)
                for used in used_indices
            ):
                continue
            selected = int(index)
            break
        if selected is None:
            rejected.append(
                RejectedNasalMatch(
                    seed,
                    "roma_no_geometrically_consistent_match",
                    {
                        "near_front_count": float(
                            np.count_nonzero(
                                globally_valid
                                & (
                                    front_distance
                                    <= float(cfg.max_front_seed_distance_px)
                                )
                            )
                        ),
                        "eligible_count": float(len(eligible)),
                    },
                )
            )
            continue
        used_indices.append(selected)
        matches.append(
            NasalPairMatch(
                side_view=side_view,
                front_pixel=batch.front_pixels[selected],
                side_pixel=batch.side_pixels[selected],
                confidence=float(combined[selected]),
                semantic_region=seed.semantic_region,
                source_matcher="roma_dense_fixed_rig",
                provenance_by_view=provenance_by_view,
                diagnostics={
                    "roma_certainty": float(batch.certainty[selected]),
                    "cycle_error_px": float(batch.cycle_error_px[selected]),
                    "epipolar_error_px": float(epipolar_error[selected]),
                    "front_seed_distance_px": float(front_distance[selected]),
                    "side_prior_distance_px": float(side_distance[selected]),
                    "texture_reliability": float(image_reliability[selected]),
                    "baseline_vertex_index": float(seed.baseline_vertex_index),
                },
            )
        )
    return NasalEpipolarMatchResult(matches=tuple(matches), rejected=tuple(rejected))


def build_square_nasal_crop(
    image_rgb: np.ndarray,
    support_mask: np.ndarray,
    *,
    margin_px: int = 28,
    minimum_size_px: int = 128,
) -> tuple[np.ndarray, NasalCropTransform]:
    image = np.asarray(image_rgb)
    support = np.asarray(support_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 RGB shape (H, W, 3)")
    if support.shape != image.shape[:2] or not np.any(support):
        raise ValueError("support_mask must be non-empty and match the image")
    if int(margin_px) < 0 or int(minimum_size_px) < 16:
        raise ValueError("crop margin or minimum size is invalid")

    ys, xs = np.nonzero(support)
    center_x = 0.5 * (float(xs.min()) + float(xs.max()) + 1.0)
    center_y = 0.5 * (float(ys.min()) + float(ys.max()) + 1.0)
    side = max(
        int(xs.max() - xs.min() + 1) + 2 * int(margin_px),
        int(ys.max() - ys.min() + 1) + 2 * int(margin_px),
        int(minimum_size_px),
    )
    side = min(side, image.shape[1], image.shape[0])
    x0 = int(round(center_x - 0.5 * side))
    y0 = int(round(center_y - 0.5 * side))
    x0 = int(np.clip(x0, 0, image.shape[1] - side))
    y0 = int(np.clip(y0, 0, image.shape[0] - side))
    x1 = x0 + side
    y1 = y0 + side
    transform = NasalCropTransform(
        work_size=(image.shape[1], image.shape[0]),
        bounds_xyxy=(x0, y0, x1, y1),
    )
    return np.ascontiguousarray(image[y0:y1, x0:x1]), transform


def _load_worker_output(
    path: Path,
    front_crop: NasalCropTransform,
    side_crop: NasalCropTransform,
) -> RoMaNasalMatchBatch:
    with np.load(path, allow_pickle=False) as values:
        required = {"front_pixels", "side_pixels", "certainty", "cycle_error_px"}
        if not required.issubset(values.files):
            raise RuntimeError("RoMa worker output is incomplete")
        front = front_crop.crop_to_work(values["front_pixels"])
        side = side_crop.crop_to_work(values["side_pixels"])
        certainty = np.asarray(values["certainty"], dtype=np.float64)
        cycle = np.asarray(values["cycle_error_px"], dtype=np.float64)
        metadata_text = str(values["metadata_json"].item()) if "metadata_json" in values else "{}"
    metadata = json.loads(metadata_text)
    width, height = front_crop.work_size
    side_width, side_height = side_crop.work_size
    valid = (
        (front[:, 0] >= 0.0)
        & (front[:, 0] < width)
        & (front[:, 1] >= 0.0)
        & (front[:, 1] < height)
        & (side[:, 0] >= 0.0)
        & (side[:, 0] < side_width)
        & (side[:, 1] >= 0.0)
        & (side[:, 1] < side_height)
    )
    return RoMaNasalMatchBatch(
        front_pixels=front[valid],
        side_pixels=side[valid],
        certainty=certainty[valid],
        cycle_error_px=cycle[valid],
        front_crop=front_crop,
        side_crop=side_crop,
        metadata=metadata,
    )


def run_roma_nasal_pair(
    front_image_rgb: np.ndarray,
    side_image_rgb: np.ndarray,
    front_support_mask: np.ndarray,
    side_support_mask: np.ndarray,
    *,
    python_executable: str | Path,
    worker_script: str | Path,
    torch_home: str | Path,
    coarse_resolution: int = 560,
    upsample_resolution: int = 560,
    stride: int = 2,
    timeout_seconds: int = 300,
) -> RoMaNasalMatchBatch:
    """Run one isolated RoMa pair and return work-image coordinates."""
    front_crop_image, front_crop = build_square_nasal_crop(
        front_image_rgb,
        front_support_mask,
    )
    side_crop_image, side_crop = build_square_nasal_crop(
        side_image_rgb,
        side_support_mask,
    )
    executable = Path(python_executable).resolve()
    worker = Path(worker_script).resolve()
    cache = Path(torch_home).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"RoMa Python executable is missing: {executable}")
    if not worker.is_file():
        raise FileNotFoundError(f"RoMa worker script is missing: {worker}")
    for value, name in (
        (coarse_resolution, "coarse_resolution"),
        (upsample_resolution, "upsample_resolution"),
    ):
        if int(value) <= 0 or int(value) % 14 != 0:
            raise ValueError(f"{name} must be a positive multiple of 14")
    if int(stride) < 1:
        raise ValueError("stride must be positive")

    with tempfile.TemporaryDirectory(prefix="roma-nasal-") as temporary:
        root = Path(temporary)
        front_path = root / "front.png"
        side_path = root / "side.png"
        result_path = root / "matches.npz"
        if not cv2.imwrite(
            str(front_path),
            cv2.cvtColor(front_crop_image, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("failed to write the temporary front nasal crop")
        if not cv2.imwrite(
            str(side_path),
            cv2.cvtColor(side_crop_image, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError("failed to write the temporary side nasal crop")
        command: Sequence[str] = (
            str(executable),
            str(worker),
            "--front",
            str(front_path),
            "--side",
            str(side_path),
            "--output",
            str(result_path),
            "--torch-home",
            str(cache),
            "--coarse-resolution",
            str(int(coarse_resolution)),
            "--upsample-resolution",
            str(int(upsample_resolution)),
            "--stride",
            str(int(stride)),
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"RoMa worker failed: {detail}")
        if not result_path.is_file():
            raise RuntimeError("RoMa worker completed without match output")
        return _load_worker_output(result_path, front_crop, side_crop)


__all__ = [
    "NasalCropTransform",
    "RoMaNasalSelectionConfig",
    "RoMaNasalMatchBatch",
    "build_square_nasal_crop",
    "run_roma_nasal_pair",
    "select_roma_nasal_matches",
]
