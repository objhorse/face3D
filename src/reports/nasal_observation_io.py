"""Strict loading for persisted multiview nasal observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NASAL_VIEWS,
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observation_coordinates import ObservationCoordinates


_VIEW_TO_FIELD = {
    "front": "front",
    "subject-left": "subject_left",
    "subject-right": "subject_right",
}
_CAMERA_CONTRACT = {
    "front": ("camera2", "front"),
    "subject-left": ("camera1", "left"),
    "subject-right": ("camera3", "right"),
}
_COORDINATE_CORE_FIELDS = {
    "source_pixel_frame",
    "mask_layouts",
    "observation_pixel_frame",
    "original_size_wh",
    "work_size_wh",
    "intrinsics",
    "distortion_coefficients",
    "undistortion_applied",
    "distorted_reverse_available",
    "conversion_source",
}
_COORDINATE_AUDIT_FIELDS = {
    "mask_pixel_layout",
    "distance_field",
    "aggregate_distance_field_usage",
    "mask_canvas_shape_hw",
    "color_space",
}
_PROFILE_SELECTION_FIELDS = {
    "epipolar_lines_work",
    "epipolar_anchor_names",
    "max_epipolar_distance_px",
    "candidate_count",
    "selected_anchor_epipolar_errors_px",
    "selected_anchor_prior_errors_px",
    "side_prior_roi_work",
    "effective_profile_roi_work",
    "max_side_prior_distance_px",
    "selection_score",
    "used_side_prior",
}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(f"{path} schema mismatch ({'; '.join(details)})")


def _safe_json_metadata(value: Any, path: str, *, depth: int = 0) -> None:
    if depth > 32:
        raise ValueError(f"{path} exceeds the safe metadata nesting limit")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _safe_json_metadata(item, f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            _safe_json_metadata(item, f"{path}.{key}", depth=depth + 1)
        return
    raise ValueError(f"{path} contains an unsupported metadata value")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _numeric_array(
    value: Any,
    path: str,
    *,
    shape: tuple[int, ...] | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a JSON array")
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain numeric values") from exc
    if shape is not None and result.shape != shape:
        raise ValueError(f"{path} must have shape {shape}, found {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{path} must contain only finite values")
    return result


def _integer_pair(value: Any, path: str) -> tuple[int, int]:
    array = _numeric_array(value, path, shape=(2,), dtype=np.float64)
    if np.any(array != np.rint(array)) or np.any(array <= 0):
        raise ValueError(f"{path} must contain two positive integers")
    return int(array[0]), int(array[1])


def _float_tuple(
    value: Any,
    path: str,
    *,
    length: int,
) -> tuple[float, ...]:
    array = _numeric_array(value, path, shape=(length,))
    return tuple(float(item) for item in array)


def _json_numeric_mapping(
    value: Any,
    path: str,
    *,
    nested: bool = False,
) -> dict[str, Any]:
    mapping = _mapping(value, path)
    result: dict[str, Any] = {}
    for name, item in mapping.items():
        key = _text(name, f"{path} key")
        if nested:
            result[key] = _json_numeric_mapping(
                item,
                f"{path}.{key}",
                nested=False,
            )
        else:
            array = _numeric_array(item, f"{path}.{key}")
            if array.ndim != 2 or array.shape[1:] != (2,):
                raise ValueError(f"{path}.{key} must have shape (N, 2)")
            result[key] = array
    return result


def _json_point_mapping(value: Any, path: str) -> dict[str, np.ndarray]:
    mapping = _mapping(value, path)
    return {
        _text(name, f"{path} key"): _numeric_array(
            item,
            f"{path}.{name}",
            shape=(2,),
        )
        for name, item in mapping.items()
    }


def _safe_fields_path(metadata_path: Path, relative_value: Any) -> Path:
    relative = Path(_text(relative_value, "fields_npz"))
    if relative.is_absolute():
        raise ValueError("fields_npz must be relative to the observation directory")
    base = metadata_path.parent.resolve()
    target = (base / relative).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError("fields_npz escapes the observation directory") from exc
    if target.suffix.lower() != ".npz":
        raise ValueError("fields_npz must reference an .npz file")
    if not target.is_file():
        raise FileNotFoundError(target)
    return target


def _field_name(value: Any, path: str, used: set[str]) -> str:
    name = _text(value, path)
    if name in used:
        raise ValueError(f"{path} duplicates NPZ field '{name}'")
    used.add(name)
    return name


def _camera_from_payload(
    payload: Mapping[str, Any],
    coordinate_payload: Mapping[str, Any],
    semantic_view: str,
    original_size: tuple[int, int],
    work_size: tuple[int, int],
) -> tuple[Camera, dict[str, Any], dict[str, Any]]:
    path = f"views.{semantic_view}.camera"
    required = {
        "camera_name",
        "camera_view",
        "subject_relative_view",
        "image_size_wh",
        "intrinsics",
        "distortion_coefficients",
        "rig_to_camera_rotation",
        "rig_to_camera_translation",
    }
    _keys(payload, required, path)
    expected_name, expected_camera_view = _CAMERA_CONTRACT[semantic_view]
    name = _text(payload["camera_name"], f"{path}.camera_name")
    camera_view = _text(payload["camera_view"], f"{path}.camera_view")
    subject_view = _text(
        payload["subject_relative_view"],
        f"{path}.subject_relative_view",
    )
    if (name, camera_view, subject_view) != (
        expected_name,
        expected_camera_view,
        semantic_view,
    ):
        raise ValueError(f"{path} violates the fixed three-camera semantic contract")
    image_size = _integer_pair(payload["image_size_wh"], f"{path}.image_size_wh")
    if image_size != original_size:
        raise ValueError(f"{path}.image_size_wh does not match original_size_wh")
    K = _numeric_array(payload["intrinsics"], f"{path}.intrinsics", shape=(3, 3))
    if not np.allclose(
        K[2],
        np.array([0.0, 0.0, 1.0]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"{path}.intrinsics must end with row [0, 0, 1]")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError(f"{path}.intrinsics focal lengths must be positive")
    if abs(float(np.linalg.det(K))) <= 1e-12:
        raise ValueError(f"{path}.intrinsics must be nonsingular")
    dist = _numeric_array(
        payload["distortion_coefficients"],
        f"{path}.distortion_coefficients",
    ).reshape(-1)
    if not len(dist):
        raise ValueError(f"{path}.distortion_coefficients must not be empty")
    rotation = _numeric_array(
        payload["rig_to_camera_rotation"],
        f"{path}.rig_to_camera_rotation",
        shape=(3, 3),
    )
    translation = _numeric_array(
        payload["rig_to_camera_translation"],
        f"{path}.rig_to_camera_translation",
        shape=(3,),
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{path}.rig_to_camera_rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"{path}.rig_to_camera_rotation must be a proper rotation")

    coordinate_path = f"views.{semantic_view}.coordinate_metadata"
    coordinate_required = _COORDINATE_CORE_FIELDS | _COORDINATE_AUDIT_FIELDS
    if semantic_view != "front":
        coordinate_required |= _PROFILE_SELECTION_FIELDS
    _keys(coordinate_payload, coordinate_required, coordinate_path)
    _safe_json_metadata(coordinate_payload, coordinate_path)
    source_frame = _text(
        coordinate_payload["source_pixel_frame"],
        f"{coordinate_path}.source_pixel_frame",
    )
    if source_frame not in {"distorted_original_px", "undistorted_original_px"}:
        raise ValueError(f"{coordinate_path}.source_pixel_frame is unsupported")
    if (
        _text(
            coordinate_payload["observation_pixel_frame"],
            f"{coordinate_path}.observation_pixel_frame",
        )
        != "undistorted_work_px"
    ):
        raise ValueError(
            f"{coordinate_path}.observation_pixel_frame must be undistorted_work_px"
        )
    if (
        _integer_pair(
            coordinate_payload["original_size_wh"],
            f"{coordinate_path}.original_size_wh",
        )
        != original_size
        or _integer_pair(
            coordinate_payload["work_size_wh"],
            f"{coordinate_path}.work_size_wh",
        )
        != work_size
    ):
        raise ValueError(f"{coordinate_path} size metadata is inconsistent")
    coordinate_K = _numeric_array(
        coordinate_payload["intrinsics"],
        f"{coordinate_path}.intrinsics",
        shape=(3, 3),
    )
    coordinate_dist = _numeric_array(
        coordinate_payload["distortion_coefficients"],
        f"{coordinate_path}.distortion_coefficients",
    ).reshape(-1)
    if not np.array_equal(coordinate_K, K) or not np.array_equal(
        coordinate_dist,
        dist,
    ):
        raise ValueError(f"{coordinate_path} calibration disagrees with camera")
    pixel_frame = "distorted" if source_frame == "distorted_original_px" else "undistorted"
    ObservationCoordinates(
        original_size=original_size,
        work_size=work_size,
        K=coordinate_K,
        dist=coordinate_dist,
        pixel_frame=pixel_frame,
    )
    return (
        Camera(
            name=name,
            view=camera_view,
            image_size=image_size,
            K=K,
            dist=dist,
            R_rig_to_camera=rotation,
            t_rig_to_camera=translation,
        ),
        dict(payload),
        dict(coordinate_payload),
    )


def _load_array(
    archive: Mapping[str, np.ndarray],
    name: str,
    path: str,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if name not in archive:
        raise ValueError(f"{path} references missing NPZ field '{name}'")
    try:
        value = np.asarray(archive[name])
    except ValueError as exc:
        raise ValueError(f"{path} cannot load NPZ field '{name}' without pickle") from exc
    if value.dtype.hasobject:
        raise ValueError(f"{path} references forbidden object NPZ field '{name}'")
    if shape is not None and value.shape != shape:
        raise ValueError(
            f"{path} NPZ field '{name}' must have shape {shape}, found {value.shape}"
        )
    if not (
        np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.bool_)
    ):
        raise ValueError(f"{path} NPZ field '{name}' must be numeric or boolean")
    if not np.isfinite(value).all():
        raise ValueError(f"{path} NPZ field '{name}' contains non-finite values")
    return np.array(value, copy=True)


def _load_view(
    payload: Mapping[str, Any],
    archive: Mapping[str, np.ndarray],
    semantic_view: str,
    used_fields: set[str],
) -> NasalViewObservation:
    path = f"views.{semantic_view}"
    required = {
        "semantic_view",
        "camera",
        "original_size_wh",
        "mask_canvas_shape_hw",
        "work_size_wh",
        "roi_work_xyxy",
        "coordinate_metadata",
        "boundaries_work",
        "anchors_work",
        "variant_boundaries_work",
        "fields",
        "summary",
    }
    _keys(payload, required, path)
    _safe_json_metadata(payload["summary"], f"{path}.summary")
    if _text(payload["semantic_view"], f"{path}.semantic_view") != semantic_view:
        raise ValueError(f"{path}.semantic_view is inconsistent")
    original_size = _integer_pair(payload["original_size_wh"], f"{path}.original_size_wh")
    mask_canvas = _integer_pair(
        payload["mask_canvas_shape_hw"],
        f"{path}.mask_canvas_shape_hw",
    )
    work_size = _integer_pair(payload["work_size_wh"], f"{path}.work_size_wh")
    roi = _float_tuple(payload["roi_work_xyxy"], f"{path}.roi_work_xyxy", length=4)
    camera, camera_metadata, coordinate_metadata = _camera_from_payload(
        _mapping(payload["camera"], f"{path}.camera"),
        _mapping(payload["coordinate_metadata"], f"{path}.coordinate_metadata"),
        semantic_view,
        original_size,
        work_size,
    )
    boundaries = _json_numeric_mapping(
        payload["boundaries_work"],
        f"{path}.boundaries_work",
    )
    anchors = _json_point_mapping(
        payload["anchors_work"],
        f"{path}.anchors_work",
    )
    variants_work = _json_numeric_mapping(
        payload["variant_boundaries_work"],
        f"{path}.variant_boundaries_work",
        nested=True,
    )
    fields = _mapping(payload["fields"], f"{path}.fields")
    _keys(
        fields,
        {
            "boundary",
            "distance",
            "distance_by_boundary",
            "confidence",
            "variant_boundaries",
        },
        f"{path}.fields",
    )
    raster_shape = (work_size[1], work_size[0])
    boundary_name = _field_name(
        fields["boundary"],
        f"{path}.fields.boundary",
        used_fields,
    )
    distance_name = _field_name(
        fields["distance"],
        f"{path}.fields.distance",
        used_fields,
    )
    confidence_name = _field_name(
        fields["confidence"],
        f"{path}.fields.confidence",
        used_fields,
    )
    distance_names = _mapping(
        fields["distance_by_boundary"],
        f"{path}.fields.distance_by_boundary",
    )
    if set(distance_names) != set(boundaries):
        raise ValueError(f"{path} named distance fields do not match boundaries")
    variant_names = _mapping(
        fields["variant_boundaries"],
        f"{path}.fields.variant_boundaries",
    )
    if set(variant_names) != set(variants_work):
        raise ValueError(f"{path} raster variants do not match curve variants")
    distances = {
        name: _load_array(
            archive,
            _field_name(
                distance_names[name],
                f"{path}.fields.distance_by_boundary.{name}",
                used_fields,
            ),
            f"{path}.fields.distance_by_boundary.{name}",
            shape=raster_shape,
        )
        for name in boundaries
    }
    variants = {
        name: _load_array(
            archive,
            _field_name(
                variant_names[name],
                f"{path}.fields.variant_boundaries.{name}",
                used_fields,
            ),
            f"{path}.fields.variant_boundaries.{name}",
            shape=raster_shape,
        )
        for name in variants_work
    }
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=original_size,
        mask_canvas_shape=mask_canvas,
        work_size=work_size,
        roi_work_xyxy=roi,
        boundaries_work=boundaries,
        boundary=_load_array(
            archive,
            boundary_name,
            f"{path}.fields.boundary",
            shape=raster_shape,
        ),
        distance_fields=distances,
        distance_field=_load_array(
            archive,
            distance_name,
            f"{path}.fields.distance",
            shape=raster_shape,
        ),
        confidence=_load_array(
            archive,
            confidence_name,
            f"{path}.fields.confidence",
            shape=raster_shape,
        ),
        variant_boundaries_work=variants_work,
        variant_boundaries=variants,
        anchors_work=anchors,
        camera_metadata=camera_metadata,
        coordinate_metadata=coordinate_metadata,
    )


def load_nasal_observation_bundle(
    metadata_path: str | Path,
) -> NasalObservationBundle:
    """Load an A-audit JSON/NPZ pair without permitting pickle or path escape."""
    path = Path(metadata_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from exc
    root = _mapping(payload, str(path))
    _keys(
        root,
        {
            "schema_version",
            "audit_only",
            "geometry_observations_only",
            "texture_scoring_included",
            "fields_npz",
            "camera_name_by_view",
            "metadata",
            "views",
        },
        str(path),
    )
    if root["schema_version"] != 1:
        raise ValueError("unsupported nasal observation schema_version")
    if (
        root["audit_only"] is not True
        or root["geometry_observations_only"] is not True
        or root["texture_scoring_included"] is not False
    ):
        raise ValueError("nasal observation geometry/audit flags are invalid")
    views = _mapping(root["views"], "views")
    if tuple(views) != NASAL_VIEWS:
        raise ValueError(
            "views must use canonical order: front, subject-left, subject-right"
        )
    camera_names = _mapping(root["camera_name_by_view"], "camera_name_by_view")
    expected_names = {
        view: _CAMERA_CONTRACT[view][0]
        for view in NASAL_VIEWS
    }
    if dict(camera_names) != expected_names:
        raise ValueError("camera_name_by_view violates the fixed rig contract")
    metadata = _mapping(root["metadata"], "metadata")
    _safe_json_metadata(metadata, "metadata")
    fields_path = _safe_fields_path(path, root["fields_npz"])
    used_fields: set[str] = set()
    try:
        with np.load(fields_path, allow_pickle=False) as archive:
            observations = {
                semantic_view: _load_view(
                    _mapping(views[semantic_view], f"views.{semantic_view}"),
                    archive,
                    semantic_view,
                    used_fields,
                )
                for semantic_view in NASAL_VIEWS
            }
            if set(archive.files) != used_fields:
                extras = sorted(set(archive.files) - used_fields)
                missing = sorted(used_fields - set(archive.files))
                details = []
                if extras:
                    details.append("unreferenced: " + ", ".join(extras))
                if missing:
                    details.append("missing: " + ", ".join(missing))
                raise ValueError("NPZ field closure mismatch (" + "; ".join(details) + ")")
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and not str(exc).startswith(
            ("Cannot load", "Object arrays")
        ):
            raise
        raise ValueError(f"failed to load safe nasal observation fields: {exc}") from exc
    return NasalObservationBundle(
        front=observations["front"],
        subject_left=observations["subject-left"],
        subject_right=observations["subject-right"],
    )


__all__ = ["load_nasal_observation_bundle"]
