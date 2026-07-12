"""Auditable cross-view surface observations for the fixed three-camera rig."""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    relative_camera_transform,
    triangulate_correspondences,
    write_ply,
)
from src.geometry.shared_rig_pose import reference_camera_to_model


@dataclass(frozen=True)
class ObservationFilterConfig:
    min_confidence: float = 0.5
    reciprocal_tolerance_px: float = 2.0
    front_merge_tolerance_px: float = 2.0
    max_reprojection_px: float = 2.5
    min_ray_angle_deg: float = 3.0
    max_three_view_delta_m: float = 0.003
    min_depth_m: float = 0.12
    max_depth_m: float = 1.50
    mask_erode_px: int = 8
    min_skin_luma: int = 25
    max_surface_distance_m: float = 0.035
    min_total_observations: int = 60
    min_observations_per_side: int = 20
    min_required_regions: int = 3


@dataclass(frozen=True)
class PairMatches:
    side_view: str
    front_points: np.ndarray
    side_points: np.ndarray
    confidence: np.ndarray
    semantic_regions: Tuple[str, ...]


@dataclass(frozen=True)
class CrossViewTrack:
    pixels_by_view: Mapping[str, np.ndarray]
    confidence_by_pair: Mapping[str, float]
    semantic_region: str


@dataclass(frozen=True)
class TriangulatedObservation:
    track: CrossViewTrack
    target_reference_point: np.ndarray
    reprojection_errors_px: Mapping[str, float]
    ray_angles_deg: Mapping[str, float]
    weight: float


@dataclass(frozen=True)
class SurfaceObservation:
    target_model_point: np.ndarray
    pixels_by_view: Mapping[str, np.ndarray]
    face_index: int
    bary_coords: np.ndarray
    surface_point: np.ndarray
    surface_distance_m: float
    reprojection_errors_px: Mapping[str, float]
    ray_angles_deg: Mapping[str, float]
    semantic_region: str
    weight: float


def _sample_map(values: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_np = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xy = np.rint(points_np).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < values.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < values.shape[0])
    )
    sampled = np.zeros(len(points_np), dtype=values.dtype)
    sampled[inside] = values[xy[inside, 1], xy[inside, 0]]
    return sampled, inside


def build_trusted_semantic_map(
    preprocessed_view: Mapping[str, object],
    config: Optional[ObservationFilterConfig] = None,
) -> np.ndarray:
    """Return 0=reject, 1=skin, 2=nose for cross-view matching."""
    cfg = config or ObservationFilterConfig()
    image = np.asarray(preprocessed_view["image"])
    labels = preprocessed_view.get("parser_labels")
    if labels is not None:
        labels_np = np.asarray(labels)
        semantic = np.zeros(labels_np.shape, dtype=np.uint8)
        semantic[labels_np == 1] = 1
        semantic[labels_np == 10] = 2
    else:
        face_mask = preprocessed_view.get("face_mask")
        if face_mask is None:
            semantic = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            semantic = (np.asarray(face_mask) > 0).astype(np.uint8)

    face_mask = preprocessed_view.get("face_mask")
    if face_mask is not None:
        semantic[np.asarray(face_mask) <= 0] = 0
    erosion = max(int(cfg.mask_erode_px), 0)
    if erosion:
        kernel_size = erosion * 2 + 1
        trusted = cv2.erode(
            (semantic > 0).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        )
        semantic[trusted == 0] = 0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    semantic[gray < int(cfg.min_skin_luma)] = 0
    return semantic


def build_trusted_skin_mask(
    preprocessed_view: Mapping[str, object],
    config: Optional[ObservationFilterConfig] = None,
) -> np.ndarray:
    return (build_trusted_semantic_map(preprocessed_view, config) > 0).astype(
        np.uint8
    ) * 255


def _face_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return 0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def classify_front_regions(
    points: np.ndarray,
    semantic_map: np.ndarray,
) -> Tuple[str, ...]:
    """Assign coarse subject-relative regions from front-view pixels."""
    sampled, _inside = _sample_map(semantic_map, points)
    x0, y0, x1, y1 = _face_bbox(semantic_map)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    result: List[str] = []
    for point, semantic in zip(np.asarray(points), sampled):
        nx = (float(point[0]) - x0) / width
        ny = (float(point[1]) - y0) / height
        if int(semantic) == 2:
            result.append("nose")
        elif ny >= 0.72:
            result.append("chin_or_jaw")
        elif 0.34 <= ny <= 0.76:
            # A person's left side appears on the right of a front image.
            result.append("subject_left_cheek" if nx >= 0.5 else "subject_right_cheek")
        else:
            result.append("upper_face")
    return tuple(result)


def _nearest_indices(query: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    query_np = np.asarray(query, dtype=np.float64).reshape(-1, 2)
    reference_np = np.asarray(reference, dtype=np.float64).reshape(-1, 2)
    if not len(query_np) or not len(reference_np):
        return (
            np.full(len(query_np), -1, dtype=np.int64),
            np.full(len(query_np), np.inf, dtype=np.float64),
        )
    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference_np).query(query_np, k=1)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float64)
    except Exception:
        delta = query_np[:, None, :] - reference_np[None, :, :]
        distances_all = np.linalg.norm(delta, axis=2)
        indices = np.argmin(distances_all, axis=1)
        return indices.astype(np.int64), distances_all[np.arange(len(query_np)), indices]


def filter_reciprocal_pair_matches(
    side_view: str,
    forward_front_points: np.ndarray,
    forward_side_points: np.ndarray,
    forward_confidence: np.ndarray,
    reverse_side_points: np.ndarray,
    reverse_front_points: np.ndarray,
    reverse_confidence: np.ndarray,
    front_semantic_map: np.ndarray,
    side_semantic_map: np.ndarray,
    config: Optional[ObservationFilterConfig] = None,
) -> Tuple[PairMatches, List[dict]]:
    """Apply confidence, skin semantics and bidirectional consistency gates."""
    cfg = config or ObservationFilterConfig()
    front = np.asarray(forward_front_points, dtype=np.float64).reshape(-1, 2)
    side = np.asarray(forward_side_points, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(forward_confidence, dtype=np.float64).reshape(-1)
    reverse_side = np.asarray(reverse_side_points, dtype=np.float64).reshape(-1, 2)
    reverse_front = np.asarray(reverse_front_points, dtype=np.float64).reshape(-1, 2)
    reverse_conf = np.asarray(reverse_confidence, dtype=np.float64).reshape(-1)
    if not (len(front) == len(side) == len(confidence)):
        raise ValueError("forward match arrays have inconsistent lengths")
    if not (len(reverse_side) == len(reverse_front) == len(reverse_conf)):
        raise ValueError("reverse match arrays have inconsistent lengths")

    front_semantic, front_inside = _sample_map(front_semantic_map, front)
    side_semantic, side_inside = _sample_map(side_semantic_map, side)
    reverse_indices, reverse_side_distance = _nearest_indices(side, reverse_side)
    reverse_front_distance = np.full(len(front), np.inf, dtype=np.float64)
    reverse_match_confidence = np.zeros(len(front), dtype=np.float64)
    has_reverse = reverse_indices >= 0
    if np.any(has_reverse):
        indices = reverse_indices[has_reverse]
        reverse_front_distance[has_reverse] = np.linalg.norm(
            reverse_front[indices] - front[has_reverse], axis=1
        )
        reverse_match_confidence[has_reverse] = reverse_conf[indices]

    accepted_indices: List[int] = []
    audit: List[dict] = []
    regions = classify_front_regions(front, front_semantic_map)
    for index in range(len(front)):
        reason = "accepted"
        if confidence[index] < cfg.min_confidence:
            reason = "low_confidence"
        elif not (front_inside[index] and side_inside[index]):
            reason = "outside_image"
        elif front_semantic[index] == 0 or side_semantic[index] == 0:
            reason = "outside_trusted_skin"
        elif front_semantic[index] != side_semantic[index]:
            reason = "semantic_mismatch"
        elif (
            reverse_side_distance[index] > cfg.reciprocal_tolerance_px
            or reverse_front_distance[index] > cfg.reciprocal_tolerance_px
        ):
            reason = "not_bidirectionally_consistent"
        elif reverse_match_confidence[index] < cfg.min_confidence:
            reason = "reverse_low_confidence"
        else:
            accepted_indices.append(index)
        audit.append(
            {
                "stage": "pair_filter",
                "side_view": side_view,
                "front_pixel": front[index].tolist(),
                "side_pixel": side[index].tolist(),
                "confidence": float(confidence[index]),
                "reverse_confidence": float(reverse_match_confidence[index]),
                "reverse_side_distance_px": float(reverse_side_distance[index]),
                "reverse_front_distance_px": float(reverse_front_distance[index]),
                "semantic_region": regions[index],
                "status": "accepted" if reason == "accepted" else "rejected",
                "reason": reason,
            }
        )
    accepted = np.asarray(accepted_indices, dtype=np.int64)
    return (
        PairMatches(
            side_view=side_view,
            front_points=front[accepted],
            side_points=side[accepted],
            confidence=np.minimum(
                confidence[accepted], reverse_match_confidence[accepted]
            ),
            semantic_regions=tuple(regions[index] for index in accepted_indices),
        ),
        audit,
    )


def build_front_centered_tracks(
    pair_matches: Mapping[str, PairMatches],
    config: Optional[ObservationFilterConfig] = None,
) -> List[CrossViewTrack]:
    cfg = config or ObservationFilterConfig()
    left = pair_matches.get("left")
    right = pair_matches.get("right")
    tracks: List[CrossViewTrack] = []
    used_right: set[int] = set()
    right_indices = np.empty(0, dtype=np.int64)
    right_distances = np.empty(0, dtype=np.float64)
    if left is not None and right is not None and len(right.front_points):
        right_indices, right_distances = _nearest_indices(
            left.front_points, right.front_points
        )

    if left is not None:
        for index in range(len(left.front_points)):
            right_index = int(right_indices[index]) if len(right_indices) else -1
            can_merge = (
                right is not None
                and right_index >= 0
                and right_index not in used_right
                and right_distances[index] <= cfg.front_merge_tolerance_px
                and left.semantic_regions[index] == right.semantic_regions[right_index]
            )
            pixels: Dict[str, np.ndarray] = {
                "front": left.front_points[index],
                "left": left.side_points[index],
            }
            confidence = {"front_left": float(left.confidence[index])}
            if can_merge:
                used_right.add(right_index)
                pixels["front"] = 0.5 * (
                    left.front_points[index] + right.front_points[right_index]
                )
                pixels["right"] = right.side_points[right_index]
                confidence["front_right"] = float(right.confidence[right_index])
            tracks.append(
                CrossViewTrack(
                    pixels_by_view=pixels,
                    confidence_by_pair=confidence,
                    semantic_region=left.semantic_regions[index],
                )
            )
    if right is not None:
        for index in range(len(right.front_points)):
            if index in used_right:
                continue
            tracks.append(
                CrossViewTrack(
                    pixels_by_view={
                        "front": right.front_points[index],
                        "right": right.side_points[index],
                    },
                    confidence_by_pair={
                        "front_right": float(right.confidence[index])
                    },
                    semantic_region=right.semantic_regions[index],
                )
            )
    return tracks


def _project_reference_point(
    point_reference: np.ndarray,
    view: str,
    cameras_by_view: Mapping[str, Camera],
    intrinsics: Mapping[str, np.ndarray],
) -> np.ndarray:
    point = np.asarray(point_reference, dtype=np.float64).reshape(3)
    if view == "front":
        camera_point = point
    else:
        rotation, translation = relative_camera_transform(
            cameras_by_view["front"], cameras_by_view[view]
        )
        camera_point = rotation @ point + translation
    homogeneous = np.asarray(intrinsics[view], dtype=np.float64) @ camera_point
    return homogeneous[:2] / homogeneous[2]


def _ray_angle_degrees(
    point_reference: np.ndarray,
    side_camera: Camera,
    front_camera: Camera,
) -> float:
    rotation, translation = relative_camera_transform(front_camera, side_camera)
    side_center_reference = -rotation.T @ translation
    ray_front = np.asarray(point_reference, dtype=np.float64)
    ray_side = np.asarray(point_reference, dtype=np.float64) - side_center_reference
    denominator = np.linalg.norm(ray_front) * np.linalg.norm(ray_side)
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(ray_front, ray_side) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def triangulate_and_filter_tracks(
    tracks: Sequence[CrossViewTrack],
    cameras_by_view: Mapping[str, Camera],
    intrinsics: Mapping[str, np.ndarray],
    config: Optional[ObservationFilterConfig] = None,
) -> Tuple[List[TriangulatedObservation], List[dict]]:
    cfg = config or ObservationFilterConfig()
    if "front" not in cameras_by_view:
        raise KeyError("front camera is required")
    accepted: List[TriangulatedObservation] = []
    audit: List[dict] = []
    for track_index, track in enumerate(tracks):
        targets: List[np.ndarray] = []
        pair_errors: Dict[str, Dict[str, float]] = {}
        angles: Dict[str, float] = {}
        reason = "accepted"
        for side_view in ("left", "right"):
            if side_view not in track.pixels_by_view:
                continue
            rotation, translation = relative_camera_transform(
                cameras_by_view["front"], cameras_by_view[side_view]
            )
            xyz, positive = triangulate_correspondences(
                np.asarray(track.pixels_by_view["front"], dtype=np.float64)[None],
                np.asarray(track.pixels_by_view[side_view], dtype=np.float64)[None],
                np.asarray(intrinsics["front"], dtype=np.float64),
                np.asarray(intrinsics[side_view], dtype=np.float64),
                rotation,
                translation,
            )
            if not bool(positive[0]) or not np.isfinite(xyz[0]).all():
                reason = "invalid_or_negative_depth"
                break
            point = xyz[0]
            side_depth = float((rotation @ point + translation)[2])
            if not (
                cfg.min_depth_m <= float(point[2]) <= cfg.max_depth_m
                and cfg.min_depth_m <= side_depth <= cfg.max_depth_m
            ):
                reason = "implausible_depth"
                break
            angle = _ray_angle_degrees(
                point, cameras_by_view[side_view], cameras_by_view["front"]
            )
            angles[side_view] = float(angle)
            if angle < cfg.min_ray_angle_deg:
                reason = "ray_angle_too_small"
                break
            front_error = float(
                np.linalg.norm(
                    _project_reference_point(
                        point, "front", cameras_by_view, intrinsics
                    )
                    - track.pixels_by_view["front"]
                )
            )
            side_error = float(
                np.linalg.norm(
                    _project_reference_point(
                        point, side_view, cameras_by_view, intrinsics
                    )
                    - track.pixels_by_view[side_view]
                )
            )
            pair_errors[side_view] = {
                "front": front_error,
                side_view: side_error,
            }
            if max(front_error, side_error) > cfg.max_reprojection_px:
                reason = "reprojection_error"
                break
            targets.append(point)

        three_view_delta = None
        if reason == "accepted" and len(targets) == 2:
            three_view_delta = float(np.linalg.norm(targets[0] - targets[1]))
            if three_view_delta > cfg.max_three_view_delta_m:
                reason = "three_view_inconsistent"
        target = np.mean(targets, axis=0) if targets else np.full(3, np.nan)
        reprojection_errors: Dict[str, float] = {}
        if reason == "accepted":
            for view, target_pixel in track.pixels_by_view.items():
                reprojection_errors[view] = float(
                    np.linalg.norm(
                        _project_reference_point(
                            target, view, cameras_by_view, intrinsics
                        )
                        - target_pixel
                    )
                )
            if max(reprojection_errors.values(), default=0.0) > cfg.max_reprojection_px:
                reason = "merged_reprojection_error"
        confidence = min(track.confidence_by_pair.values(), default=0.0)
        weight = float(confidence * (1.0 if len(targets) == 2 else 0.6))
        record = {
            "stage": "triangulation",
            "source_frame": "front_camera",
            "units": "meters",
            "track_index": int(track_index),
            "pixels_by_view": {
                key: np.asarray(value).tolist()
                for key, value in track.pixels_by_view.items()
            },
            "semantic_region": track.semantic_region,
            "target_reference_point": target.tolist(),
            "reprojection_errors_px": reprojection_errors,
            "pair_reprojection_errors_px": pair_errors,
            "ray_angles_deg": angles,
            "three_view_delta_m": three_view_delta,
            "weight": weight,
            "status": "accepted" if reason == "accepted" else "rejected",
            "reason": reason,
        }
        audit.append(record)
        if reason == "accepted":
            accepted.append(
                TriangulatedObservation(
                    track=track,
                    target_reference_point=target.astype(np.float64),
                    reprojection_errors_px=reprojection_errors,
                    ray_angles_deg=angles,
                    weight=weight,
                )
            )
    return accepted, audit


def sparse_zbuffer_attachments(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_rotation: np.ndarray,
    camera_translation: np.ndarray,
    intrinsics: np.ndarray,
    pixels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the nearest visible triangle only at requested pixels."""
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    camera_vertices = (
        vertices_np @ np.asarray(camera_rotation, dtype=np.float64).reshape(3, 3).T
        + np.asarray(camera_translation, dtype=np.float64).reshape(3)
    )
    projected_h = camera_vertices @ np.asarray(intrinsics, dtype=np.float64).T
    projected = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-12)
    triangles_2d = projected[faces_np]
    triangles_z = camera_vertices[faces_np, 2]
    valid_faces = np.all(triangles_z > 1e-8, axis=1)
    mins = triangles_2d.min(axis=1)
    maxs = triangles_2d.max(axis=1)

    query = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    face_ids = np.full(len(query), -1, dtype=np.int64)
    barycentric = np.full((len(query), 3), np.nan, dtype=np.float64)
    depths = np.full(len(query), np.nan, dtype=np.float64)
    for query_index, point in enumerate(query):
        candidates = np.flatnonzero(
            valid_faces
            & (mins[:, 0] <= point[0])
            & (maxs[:, 0] >= point[0])
            & (mins[:, 1] <= point[1])
            & (maxs[:, 1] >= point[1])
        )
        if not len(candidates):
            continue
        triangles = triangles_2d[candidates]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        denominator = (
            (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
            + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        )
        non_degenerate = np.abs(denominator) > 1e-12
        safe_denominator = np.where(non_degenerate, denominator, 1.0)
        w0 = (
            (b[:, 1] - c[:, 1]) * (point[0] - c[:, 0])
            + (c[:, 0] - b[:, 0]) * (point[1] - c[:, 1])
        ) / safe_denominator
        w1 = (
            (c[:, 1] - a[:, 1]) * (point[0] - c[:, 0])
            + (a[:, 0] - c[:, 0]) * (point[1] - c[:, 1])
        ) / safe_denominator
        screen_bary = np.column_stack((w0, w1, 1.0 - w0 - w1))
        inside = non_degenerate & np.all(screen_bary >= -1e-7, axis=1)
        if not np.any(inside):
            continue
        candidates = candidates[inside]
        screen_bary = screen_bary[inside]
        candidate_z = triangles_z[candidates]
        weighted_inverse_z = screen_bary / candidate_z
        normalization = weighted_inverse_z.sum(axis=1)
        model_bary = weighted_inverse_z / normalization[:, None]
        candidate_depth = 1.0 / normalization
        nearest = int(np.argmin(candidate_depth))
        face_ids[query_index] = int(candidates[nearest])
        barycentric[query_index] = model_bary[nearest]
        depths[query_index] = float(candidate_depth[nearest])
    return face_ids, barycentric, depths


def attach_observations_to_mesh(
    observations: Sequence[TriangulatedObservation],
    vertices: np.ndarray,
    faces: np.ndarray,
    front_rotation: np.ndarray,
    front_translation: np.ndarray,
    front_intrinsics: np.ndarray,
    config: Optional[ObservationFilterConfig] = None,
) -> Tuple[List[SurfaceObservation], List[dict]]:
    cfg = config or ObservationFilterConfig()
    if not observations:
        return [], []
    pixels = np.asarray(
        [obs.track.pixels_by_view["front"] for obs in observations],
        dtype=np.float64,
    )
    face_ids, barycentric, depths = sparse_zbuffer_attachments(
        vertices,
        faces,
        front_rotation,
        front_translation,
        front_intrinsics,
        pixels,
    )
    targets_reference = np.asarray(
        [obs.target_reference_point for obs in observations], dtype=np.float64
    )
    targets_model = reference_camera_to_model(
        targets_reference, front_rotation, front_translation
    )
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    accepted: List[SurfaceObservation] = []
    audit: List[dict] = []
    for index, observation in enumerate(observations):
        reason = "accepted"
        surface_point = np.full(3, np.nan)
        surface_distance = float("inf")
        if face_ids[index] < 0 or not np.isfinite(barycentric[index]).all():
            reason = "no_visible_surface_hit"
        else:
            surface_point = (
                vertices_np[faces_np[face_ids[index]]]
                * barycentric[index][:, None]
            ).sum(axis=0)
            surface_distance = float(
                np.linalg.norm(surface_point - targets_model[index])
            )
            if surface_distance > cfg.max_surface_distance_m:
                reason = "surface_distance_too_large"
        record = {
            "stage": "surface_attachment",
            "source_frame": "front_camera",
            "target_frame": "flame_model",
            "units": "meters",
            "model_to_reference_camera": {
                "R": np.asarray(front_rotation, dtype=np.float64).reshape(3, 3).tolist(),
                "t": np.asarray(front_translation, dtype=np.float64).reshape(3).tolist(),
            },
            "pixels_by_view": {
                key: np.asarray(value).tolist()
                for key, value in observation.track.pixels_by_view.items()
            },
            "semantic_region": observation.track.semantic_region,
            "target_reference_point": observation.target_reference_point.tolist(),
            "target_model_point": targets_model[index].tolist(),
            "face_index": int(face_ids[index]),
            "bary_coords": barycentric[index].tolist(),
            "surface_point": surface_point.tolist(),
            "surface_depth_m": float(depths[index]),
            "surface_distance_m": surface_distance,
            "reprojection_errors_px": dict(observation.reprojection_errors_px),
            "ray_angles_deg": dict(observation.ray_angles_deg),
            "weight": float(observation.weight),
            "status": "accepted" if reason == "accepted" else "rejected",
            "reason": reason,
        }
        audit.append(record)
        if reason == "accepted":
            accepted.append(
                SurfaceObservation(
                    target_model_point=targets_model[index].astype(np.float64),
                    pixels_by_view=observation.track.pixels_by_view,
                    face_index=int(face_ids[index]),
                    bary_coords=barycentric[index].astype(np.float64),
                    surface_point=surface_point.astype(np.float64),
                    surface_distance_m=surface_distance,
                    reprojection_errors_px=observation.reprojection_errors_px,
                    ray_angles_deg=observation.ray_angles_deg,
                    semantic_region=observation.track.semantic_region,
                    weight=float(observation.weight),
                )
            )
    return accepted, audit


def build_observation_summary(
    observations: Sequence[SurfaceObservation],
    attachment_audit: Sequence[Mapping[str, object]],
    pair_audit: Sequence[Mapping[str, object]],
    triangulation_audit: Sequence[Mapping[str, object]],
    config: Optional[ObservationFilterConfig] = None,
) -> dict:
    cfg = config or ObservationFilterConfig()
    side_counts = {
        side: int(sum(side in obs.pixels_by_view for obs in observations))
        for side in ("left", "right")
    }
    regions = sorted({obs.semantic_region for obs in observations})
    required_region_aliases = {
        "nose": "nose",
        "subject_left_cheek": "left_cheek",
        "subject_right_cheek": "right_cheek",
        "chin_or_jaw": "chin_or_jaw",
    }
    covered_required = sorted(
        required_region_aliases[region]
        for region in regions
        if region in required_region_aliases
    )
    reprojection = np.asarray(
        [
            error
            for obs in observations
            for error in obs.reprojection_errors_px.values()
        ],
        dtype=np.float64,
    )
    surface_distances = np.asarray(
        [obs.surface_distance_m for obs in observations], dtype=np.float64
    )
    rejection_counts: Dict[str, int] = {}
    for record in list(pair_audit) + list(triangulation_audit) + list(attachment_audit):
        if record.get("status") != "rejected":
            continue
        reason = str(record.get("reason", "unknown"))
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    p90_reprojection = (
        float(np.percentile(reprojection, 90)) if len(reprojection) else None
    )
    gates = {
        "total_observations": len(observations) >= cfg.min_total_observations,
        "left_observations": side_counts["left"] >= cfg.min_observations_per_side,
        "right_observations": side_counts["right"] >= cfg.min_observations_per_side,
        "required_regions": len(covered_required) >= cfg.min_required_regions,
        "p90_reprojection": p90_reprojection is not None
        and p90_reprojection <= cfg.max_reprojection_px,
    }
    return {
        "m1_passed": bool(all(gates.values())),
        "gates": gates,
        "accepted_observations": int(len(observations)),
        "accepted_by_side": side_counts,
        "covered_regions": regions,
        "covered_required_regions": covered_required,
        "reprojection_error_px": {
            "p50": float(np.percentile(reprojection, 50)) if len(reprojection) else None,
            "p90": p90_reprojection,
            "max": float(reprojection.max()) if len(reprojection) else None,
        },
        "surface_distance_m": {
            "p50": float(np.percentile(surface_distances, 50))
            if len(surface_distances)
            else None,
            "p90": float(np.percentile(surface_distances, 90))
            if len(surface_distances)
            else None,
            "max": float(surface_distances.max()) if len(surface_distances) else None,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "thresholds": {
            "min_total_observations": cfg.min_total_observations,
            "min_observations_per_side": cfg.min_observations_per_side,
            "min_required_regions": cfg.min_required_regions,
            "max_reprojection_px": cfg.max_reprojection_px,
        },
    }


def _draw_observation_overlay(
    image: np.ndarray,
    view: str,
    records: Sequence[Mapping[str, object]],
) -> np.ndarray:
    canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    colors = {
        "accepted": (80, 220, 100),
        "low_confidence": (0, 190, 255),
        "semantic_mismatch": (220, 120, 40),
        "three_view_inconsistent": (220, 60, 220),
        "rejected": (70, 70, 230),
    }
    for record in records:
        pixels = record.get("pixels_by_view", {})
        if view not in pixels:
            continue
        point = tuple(np.rint(pixels[view]).astype(int))
        status = str(record.get("status", "rejected"))
        reason = str(record.get("reason", "rejected"))
        color = (
            colors["accepted"]
            if status == "accepted"
            else colors.get(reason, colors["rejected"])
        )
        cv2.circle(canvas, point, 3, color, -1, lineType=cv2.LINE_AA)
    return canvas


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_observation_audit(
    output_dir: Path,
    images: Mapping[str, np.ndarray],
    pair_audit: Sequence[Mapping[str, object]],
    triangulation_audit: Sequence[Mapping[str, object]],
    attachment_audit: Sequence[Mapping[str, object]],
    observations: Sequence[SurfaceObservation],
    summary: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_ready({
        "pair_matches": list(pair_audit),
        "triangulation": list(triangulation_audit),
        "surface_attachments": list(attachment_audit),
    })
    (output_dir / "observations.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "observation_summary.json").write_text(
        json.dumps(_json_ready(dict(summary)), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pair_overlay_records = []
    for record in pair_audit:
        side = str(record.get("side_view", ""))
        if not side:
            continue
        copied = dict(record)
        copied["pixels_by_view"] = {
            "front": record.get("front_pixel"),
            side: record.get("side_pixel"),
        }
        pair_overlay_records.append(copied)
    final_overlay_records = (
        [r for r in triangulation_audit if r.get("status") == "rejected"]
        + list(attachment_audit)
    )
    raw_overlay_records = pair_overlay_records + final_overlay_records
    for view, image in images.items():
        final_overlay = _draw_observation_overlay(
            image, view, final_overlay_records
        )
        raw_overlay = _draw_observation_overlay(image, view, raw_overlay_records)
        cv2.imwrite(
            str(output_dir / f"triangulation_overlay_{view}.png"), final_overlay
        )
        cv2.imwrite(str(output_dir / f"raw_match_overlay_{view}.png"), raw_overlay)

    points = np.asarray(
        [obs.target_model_point for obs in observations], dtype=np.float64
    ).reshape(-1, 3)
    colors = np.tile(np.array([[80, 220, 100]], dtype=np.uint8), (len(points), 1))
    write_ply(output_dir / "triangulated_points.ply", points, colors)

    gate_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{'pass' if passed else 'fail'}</td></tr>"
        for name, passed in summary.get("gates", {}).items()
    )
    rejection_rows = "".join(
        f"<tr><td>{html.escape(str(name))}</td><td>{int(count)}</td></tr>"
        for name, count in summary.get("rejection_counts", {}).items()
    )
    image_cards = "".join(
        f"<figure><img src='triangulation_overlay_{html.escape(view)}.png'><figcaption>{html.escape(view)}</figcaption></figure>"
        for view in images
    )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>跨视角观测审计</title>
<style>body{{margin:0;background:#10161c;color:#eaf0f4;font-family:system-ui,sans-serif;letter-spacing:0}}main{{max-width:1280px;margin:auto;padding:24px}}.status{{font-size:22px;color:{'#72d69c' if summary.get('m1_passed') else '#ff9c8f'}}}table{{border-collapse:collapse;width:100%;margin:14px 0}}td,th{{padding:8px;border-bottom:1px solid #34424d;text-align:left}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}figure{{margin:0;border:1px solid #34424d}}img{{display:block;width:100%}}figcaption{{padding:8px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>三视角跨视角观测审计</h1><p class="status">M1: {'通过' if summary.get('m1_passed') else '未通过'}</p>
<p>接受观测：{int(summary.get('accepted_observations', 0))}。绿色为最终可信点，红色为几何拒绝点。原始低置信和背景匹配另存为 raw_match_overlay，不参与变形。</p><p>该审计只判断观测能否进入受控脸型实验，不代表医疗级深度精度。</p>
<h2>门槛</h2><table><tbody>{gate_rows}</tbody></table><h2>拒绝原因</h2><table><tbody>{rejection_rows}</tbody></table>
<h2>投影检查</h2><div class="grid">{image_cards}</div></main></body></html>"""
    (output_dir / "observation_audit.html").write_text(document, encoding="utf-8")
