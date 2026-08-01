"""Convert trusted RoMa surface observations into texture-warp controls."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def work_pixels_to_letterbox_canvas(
    pixels: np.ndarray,
    *,
    work_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("pixels must have finite shape (N, 2)")
    work_width, work_height = map(float, work_size)
    canvas_height, canvas_width = map(float, canvas_shape)
    if min(work_width, work_height, canvas_width, canvas_height) <= 0.0:
        raise ValueError("work and canvas dimensions must be positive")
    scale = min(canvas_width / work_width, canvas_height / work_height)
    offset = np.asarray(
        [
            0.5 * (canvas_width - work_width * scale),
            0.5 * (canvas_height - work_height * scale),
        ],
        dtype=np.float64,
    )
    return points * scale + offset


def _project_vertex(vertices: np.ndarray, index: int, camera: Mapping[str, Any]) -> np.ndarray:
    point = np.asarray(vertices[index], dtype=np.float64)
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    intrinsics = np.asarray(camera["K"], dtype=np.float64)
    camera_point = rotation @ point + translation
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1e-8:
        return np.asarray([np.nan, np.nan], dtype=np.float64)
    homogeneous = intrinsics @ camera_point
    return homogeneous[:2] / homogeneous[2]


@dataclass(frozen=True)
class RoMaTextureControlSet:
    model_points: np.ndarray
    observed_points: np.ndarray
    groups: tuple[str, ...]
    weights: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        model = np.asarray(self.model_points, dtype=np.float32)
        observed = np.asarray(self.observed_points, dtype=np.float32)
        weights = np.asarray(self.weights, dtype=np.float32).reshape(-1)
        if (
            model.ndim != 2
            or model.shape[1] != 2
            or observed.shape != model.shape
            or weights.shape != (len(model),)
            or len(self.groups) != len(model)
            or not np.isfinite(model).all()
            or not np.isfinite(observed).all()
            or not np.isfinite(weights).all()
        ):
            raise ValueError("RoMa texture controls are inconsistent")
        object.__setattr__(self, "model_points", model)
        object.__setattr__(self, "observed_points", observed)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "groups", tuple(map(str, self.groups)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def build_roma_texture_controls(
    metrics: Mapping[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    cameras: Mapping[str, Mapping[str, Any]],
    *,
    work_size: tuple[int, int] = (640, 480),
    canvas_shape: tuple[int, int] = (1024, 1024),
    max_controls_per_view: int = 14,
    minimum_spacing_px: float = 10.0,
    maximum_displacement_px: float = 72.0,
) -> dict[str, RoMaTextureControlSet]:
    """Build sparse, spatially distributed model-to-image controls."""
    mesh_vertices = np.asarray(vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    trusted = tuple(dict(metrics.get("observations", {})).get("trusted", ()))
    view_candidates: dict[str, list[dict[str, Any]]] = {
        "front": [],
        "left": [],
        "right": [],
    }
    semantic_to_texture = {
        "subject-left": "left",
        "subject-right": "right",
    }
    from src.geometry.cross_view_surface_observations import (
        sparse_zbuffer_attachments,
    )

    front_pixels_work = np.asarray(
        [
            dict(observation.get("pair_match", {})).get(
                "front_pixel",
                [np.nan, np.nan],
            )
            for observation in trusted
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    front_pixels_canvas = work_pixels_to_letterbox_canvas(
        front_pixels_work,
        work_size=work_size,
        canvas_shape=canvas_shape,
    )
    front_camera = cameras["front"]
    face_indices, barycentric, _depth = sparse_zbuffer_attachments(
        mesh_vertices,
        topology,
        np.asarray(front_camera["R"], dtype=np.float64),
        np.asarray(front_camera["t"], dtype=np.float64),
        np.asarray(front_camera["K"], dtype=np.float64),
        front_pixels_canvas,
    )
    rejected = 0
    for observation_index, observation in enumerate(trusted):
        pair = dict(observation.get("pair_match", {}))
        try:
            side_semantic = str(pair["side_view"])
            side_view = semantic_to_texture[side_semantic]
        except (KeyError, TypeError, ValueError):
            rejected += 1
            continue
        face_index = int(face_indices[observation_index])
        if face_index < 0:
            rejected += 1
            continue
        triangle = topology[face_index]
        surface_point = np.sum(
            mesh_vertices[triangle]
            * np.asarray(barycentric[observation_index], dtype=np.float64)[:, None],
            axis=0,
        )
        anchor_key = (
            face_index,
            tuple(np.rint(np.asarray(barycentric[observation_index]) * 10000.0).astype(int)),
        )
        weight = float(observation.get("weight", pair.get("confidence", 0.0)))
        region = str(pair.get("semantic_region", "nasal"))
        for view, raw_pixel in (
            ("front", pair.get("front_pixel")),
            (side_view, pair.get("side_pixel")),
        ):
            if view not in cameras or raw_pixel is None:
                rejected += 1
                continue
            observed = work_pixels_to_letterbox_canvas(
                np.asarray(raw_pixel, dtype=np.float64).reshape(1, 2),
                work_size=work_size,
                canvas_shape=canvas_shape,
            )[0]
            model = _project_vertex(surface_point.reshape(1, 3), 0, cameras[view])
            displacement = float(np.linalg.norm(observed - model))
            if (
                not np.isfinite(model).all()
                or not np.isfinite(observed).all()
                or not np.isfinite(weight)
                or weight <= 0.0
                or displacement > float(maximum_displacement_px)
            ):
                rejected += 1
                continue
            view_candidates[view].append(
                {
                    "anchor_key": anchor_key,
                    "model": model,
                    "observed": observed,
                    "weight": weight,
                    "group": f"roma_{region}",
                    "displacement_px": displacement,
                }
            )

    result: dict[str, RoMaTextureControlSet] = {}
    for view, candidates in view_candidates.items():
        by_anchor: dict[tuple[Any, ...], dict[str, Any]] = {}
        for candidate in candidates:
            anchor_key = tuple(candidate["anchor_key"])
            previous = by_anchor.get(anchor_key)
            if previous is None or float(candidate["weight"]) > float(previous["weight"]):
                by_anchor[anchor_key] = candidate
        ordered = sorted(
            by_anchor.values(),
            key=lambda value: (-float(value["weight"]), float(value["displacement_px"])),
        )
        selected: list[dict[str, Any]] = []
        for candidate in ordered:
            point = np.asarray(candidate["model"], dtype=np.float64)
            if selected and min(
                np.linalg.norm(point - np.asarray(value["model"], dtype=np.float64))
                for value in selected
            ) < float(minimum_spacing_px):
                continue
            selected.append(candidate)
            if len(selected) >= int(max_controls_per_view):
                break
        if len(selected) < 4:
            selected_keys = {tuple(value["anchor_key"]) for value in selected}
            for candidate in ordered:
                anchor_key = tuple(candidate["anchor_key"])
                if anchor_key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(anchor_key)
                if len(selected) >= min(4, int(max_controls_per_view)):
                    break
        if not selected:
            continue
        result[view] = RoMaTextureControlSet(
            model_points=np.asarray([value["model"] for value in selected]),
            observed_points=np.asarray([value["observed"] for value in selected]),
            groups=tuple(str(value["group"]) for value in selected),
            weights=np.asarray([value["weight"] for value in selected]),
            metadata={
                "source": "roma_dense_fixed_rig",
                "candidate_count": int(len(candidates)),
                "unique_surface_anchor_count": int(len(by_anchor)),
                "selected_count": int(len(selected)),
                "maximum_selected_displacement_px": float(
                    max(float(value["displacement_px"]) for value in selected)
                ),
                "rejected_count_global": int(rejected),
                "median_translation_px": np.median(
                    np.asarray([value["observed"] - value["model"] for value in selected]),
                    axis=0,
                ).astype(float).tolist(),
                "work_size": list(map(int, work_size)),
                "canvas_shape": list(map(int, canvas_shape)),
            },
        )
    return result


__all__ = [
    "RoMaTextureControlSet",
    "build_roma_texture_controls",
    "work_pixels_to_letterbox_canvas",
]
