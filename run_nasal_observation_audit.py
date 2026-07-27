"""Build read-only nasal observations from calibrated three-view captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src import config as cfg
from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalObservationConfig,
    build_multiview_nasal_observations,
    build_profile_nasal_observation,
)
from src.geometry.observation_coordinates import ObservationCoordinates
from src.geometry.profile_triangulation import (
    PROFILE_VIEWS,
    ProfileRig,
    load_profile_rig,
)
from src.reports.nasal_observation_report import (
    read_image_file,
    write_image_file,
    write_nasal_observation_data,
    write_nasal_observation_report,
)


BASELINE_ANCHOR_DEFINITIONS = {
    "upper_tip": {
        "landmark_68_index": 29,
        "description": "lower bridge / upper tip",
    },
    "tip_apex": {
        "landmark_68_index": 30,
        "description": "tip apex",
    },
    "lower_tip": {
        "landmark_68_index": 33,
        "description": "subnasale / lower tip",
    },
    "alar_transition": {
        "landmark_68_index_by_subject_side": {
            "subject-left": 35,
            "subject-right": 31,
        },
        "description": "same-subject-side alar transition",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create pure-geometry nasal observations without modifying the "
            "source reconstruction."
        )
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--source-output",
        type=Path,
        required=True,
        help="Baseline stable output containing meshes/stable_fit_meta.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rig-calibration",
        type=Path,
        required=True,
        help="Validated fixed-rig calibration JSON",
    )
    parser.add_argument("--work-width", type=int, default=640)
    parser.add_argument("--work-height", type=int, default=480)
    parser.add_argument("--max-stereo-rms-px", type=float, default=10.0)
    return parser.parse_args()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_separate_output(
    output: str | Path,
    source_output: str | Path,
    capture_dir: str | Path,
) -> None:
    """Reject any audit output that could overwrite immutable inputs."""
    target = Path(output).resolve()
    source = Path(source_output).resolve()
    captures = Path(capture_dir).resolve()
    if _is_within(target, source):
        raise ValueError(
            f"--output must not be inside --source-output: {target}"
        )
    if _is_within(target, captures):
        raise ValueError(
            f"--output must not be inside --capture-dir: {target}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_tree_hashes(root: str | Path) -> dict[str, str]:
    """Return stable hashes for every file under an immutable input tree."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {base}")
    return {
        path.relative_to(base).as_posix(): _sha256_file(path)
        for path in sorted(
            (candidate for candidate in base.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(base).as_posix(),
        )
    }


def assert_file_tree_unchanged(
    root: str | Path,
    expected_hashes: Mapping[str, str],
) -> None:
    actual = file_tree_hashes(root)
    expected = dict(expected_hashes)
    if actual == expected:
        return
    changed = sorted(
        name
        for name in set(actual) | set(expected)
        if actual.get(name) != expected.get(name)
    )
    raise RuntimeError(
        "source files changed during nasal observation audit: "
        + ", ".join(changed[:12])
    )


def _tree_digest(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_capture_images(
    capture_dir: Path,
    rig: ProfileRig,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    supported_suffixes = {".jpg", ".jpeg", ".png"}
    available = [
        path
        for path in capture_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    images: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for view in PROFILE_VIEWS:
        camera = rig.cameras_by_view[view]
        prefix = f"{camera.name}_".lower()
        candidates = sorted(
            path for path in available
            if path.name.lower().startswith(prefix)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"expected one RGB capture for {camera.name}/{view} in "
                f"{capture_dir}, found {len(candidates)}"
            )
        image_bgr = read_image_file(
            candidates[0],
            cv2.IMREAD_COLOR,
            description="capture image",
        )
        actual_size = (image_bgr.shape[1], image_bgr.shape[0])
        if actual_size != tuple(camera.image_size):
            raise ValueError(
                f"{camera.name}/{view} capture size {actual_size} does not "
                f"match calibration {camera.image_size}"
            )
        images[view] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        paths[view] = candidates[0].resolve()
    return images, paths


def build_undistorted_observation_rig(
    rig: ProfileRig,
    new_intrinsics_by_view: Mapping[str, Any],
) -> ProfileRig:
    """Copy a fixed rig into the already-undistorted observation frame."""
    missing = [
        view for view in PROFILE_VIEWS
        if view not in new_intrinsics_by_view
    ]
    if missing:
        raise ValueError(
            "undistortion did not return new K for views: "
            + ", ".join(missing)
        )
    cameras = {}
    for view in PROFILE_VIEWS:
        source = rig.cameras_by_view[view]
        intrinsics = np.asarray(
            new_intrinsics_by_view[view],
            dtype=np.float64,
        ).reshape(3, 3)
        if (
            not np.isfinite(intrinsics).all()
            or abs(float(np.linalg.det(intrinsics))) <= 1e-12
        ):
            raise ValueError(f"invalid undistorted new K for {view}")
        cameras[view] = Camera(
            name=source.name,
            view=source.view,
            image_size=tuple(source.image_size),
            K=intrinsics.copy(),
            dist=np.zeros_like(
                np.asarray(source.dist, dtype=np.float64).reshape(-1)
            ),
            R_rig_to_camera=np.asarray(
                source.R_rig_to_camera,
                dtype=np.float64,
            ).copy(),
            t_rig_to_camera=np.asarray(
                source.t_rig_to_camera,
                dtype=np.float64,
            ).reshape(3).copy(),
        )
    return ProfileRig(
        cameras_by_view=cameras,
        reference_view=rig.reference_view,
        units=rig.units,
        calibration_path=rig.calibration_path,
        stereo_rms_px=dict(rig.stereo_rms_px),
    )


def make_nasal_audit_config(
    work_size: tuple[int, int],
) -> NasalObservationConfig:
    """Scale one fixed audit configuration from the 640x480 reference frame."""
    width, height = (int(value) for value in work_size)
    if width <= 0 or height <= 0:
        raise ValueError("work size dimensions must be positive")
    scale = min(width / 640.0, height / 480.0)
    return NasalObservationConfig(
        work_size=(width, height),
        mask_perturbation_px=max(1, int(round(2.0 * scale))),
        gradient_window_px=max(1, int(round(5.0 * scale))),
        max_side_prior_distance_px=80.0 * scale,
        max_epipolar_distance_px=8.0 * scale,
        distance_clip_px=48.0 * scale,
        confidence_spread_px=3.0 * scale,
        min_boundary_points=max(6, int(round(12.0 * scale))),
    )


def require_parser_nose_mask(
    preprocessed_views: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    """Require real parser labels and a nonempty front CelebAMask-HQ nose."""
    missing_views = [
        view for view in PROFILE_VIEWS
        if view not in preprocessed_views
    ]
    if missing_views:
        raise RuntimeError(
            "preprocess results are missing views: "
            + ", ".join(missing_views)
        )
    unavailable = [
        view for view in PROFILE_VIEWS
        if preprocessed_views[view].get("parser_labels") is None
    ]
    if unavailable:
        raise RuntimeError(
            "face parser unavailable for views: " + ", ".join(unavailable)
        )
    raw_front_mask = preprocessed_views["front"].get("nose_mask")
    if raw_front_mask is None:
        raise RuntimeError(
            "front nose_mask is empty; CelebAMask-HQ label 10 is required"
        )
    front_mask = np.asarray(raw_front_mask, dtype=np.uint8)
    if front_mask.ndim != 2 or not np.any(front_mask > 0):
        raise RuntimeError(
            "front nose_mask is empty; CelebAMask-HQ label 10 is required"
        )
    for side_view in ("left", "right"):
        raw_face_mask = preprocessed_views[side_view].get("face_mask")
        if raw_face_mask is None:
            raise RuntimeError(
                f"{side_view} preprocess face_mask is empty"
            )
        face_mask = np.asarray(raw_face_mask, dtype=np.uint8)
        if face_mask.ndim != 2 or not np.any(face_mask > 0):
            raise RuntimeError(
                f"{side_view} preprocess face_mask is empty"
            )
    return np.where(front_mask > 0, 255, 0).astype(np.uint8)


def flame_landmarks_from_embedding(
    vertices: Any,
    faces: Any,
    landmark_mapping: Mapping[str, Any],
) -> np.ndarray:
    """Evaluate the public FLAME 68-point barycentric embedding."""
    mesh_vertices = np.asarray(vertices, dtype=np.float64)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    face_indices = np.asarray(
        landmark_mapping.get("face_idx"),
        dtype=np.int64,
    ).reshape(-1)
    barycentric = np.asarray(
        landmark_mapping.get("bary_coords"),
        dtype=np.float64,
    ).reshape(-1, 3)
    if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3:
        raise ValueError("FLAME vertices must have shape (N, 3)")
    if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3:
        raise ValueError("FLAME faces must have shape (F, 3)")
    if len(face_indices) != 68 or barycentric.shape != (68, 3):
        raise ValueError("FLAME landmark embedding must contain 68 points")
    if (
        np.any(face_indices < 0)
        or np.any(face_indices >= len(mesh_faces))
        or not np.isfinite(barycentric).all()
    ):
        raise ValueError("FLAME landmark embedding contains invalid values")
    if not np.allclose(
        np.sum(barycentric, axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError("FLAME landmark barycentric weights must sum to one")
    triangle_vertices = mesh_vertices[mesh_faces[face_indices]]
    landmarks = np.sum(
        triangle_vertices * barycentric[:, :, None],
        axis=1,
    )
    if not np.isfinite(landmarks).all():
        raise ValueError("FLAME landmarks contain non-finite values")
    return landmarks


def project_flame_points_through_front_fit(
    points_flame: Any,
    fit_front_rotation: Any,
    fit_front_translation: Any,
    rig: ProfileRig,
) -> dict[str, np.ndarray]:
    """Project FLAME points through fit camera2 coordinates into the fixed rig."""
    points = np.asarray(points_flame, dtype=np.float64).reshape(-1, 3)
    rotation = np.asarray(
        fit_front_rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    translation = np.asarray(
        fit_front_translation,
        dtype=np.float64,
    ).reshape(3)
    if (
        not np.isfinite(points).all()
        or not np.isfinite(rotation).all()
        or not np.isfinite(translation).all()
    ):
        raise ValueError("baseline projection inputs must be finite")
    front_camera = rig.cameras_by_view.get("front")
    if front_camera is None or front_camera.name != "camera2":
        raise ValueError("fixed rig must use camera2 as the front camera")

    points_front_camera = points @ rotation.T + translation
    points_rig = (
        points_front_camera
        - np.asarray(front_camera.t_rig_to_camera, dtype=np.float64)
    ) @ np.asarray(front_camera.R_rig_to_camera, dtype=np.float64)

    projected = {}
    for view in PROFILE_VIEWS:
        camera = rig.cameras_by_view[view]
        points_camera = (
            points_rig
            @ np.asarray(camera.R_rig_to_camera, dtype=np.float64).T
            + np.asarray(camera.t_rig_to_camera, dtype=np.float64)
        )
        depth = points_camera[:, 2]
        if np.any(~np.isfinite(depth)) or np.any(depth <= 1e-6):
            raise ValueError(
                f"baseline FLAME priors have invalid depth in {view}"
            )
        homogeneous = (
            points_camera @ np.asarray(camera.K, dtype=np.float64).T
        )
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        if not np.isfinite(pixels).all():
            raise ValueError(
                f"baseline FLAME priors project non-finitely in {view}"
            )
        projected[view] = pixels
    return projected


def _required_mapping(
    container: Mapping[str, Any],
    key: str,
    *,
    field_path: str,
    metadata_path: Path,
) -> Mapping[str, Any]:
    if key not in container:
        raise ValueError(
            f"{metadata_path}: missing required field '{field_path}'"
        )
    value = container[key]
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must be a JSON object"
        )
    return value


def _required_numeric_array(
    container: Mapping[str, Any],
    key: str,
    *,
    field_path: str,
    metadata_path: Path,
    dtype: Any,
    size: int | None = None,
) -> np.ndarray:
    if key not in container:
        raise ValueError(
            f"{metadata_path}: missing required field '{field_path}'"
        )
    value = container[key]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must be a JSON array"
        )
    try:
        result = np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must contain numbers"
        ) from exc
    if not result.size:
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must not be empty"
        )
    if size is not None and result.size != size:
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must contain "
            f"{size} numbers, found {result.size}"
        )
    if not np.isfinite(result).all():
        raise ValueError(
            f"{metadata_path}: field '{field_path}' must contain finite numbers"
        )
    return result


def load_baseline_fit_parameters(
    metadata_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the baseline FLAME parameters needed by the audit."""
    target = Path(metadata_path)
    fit_meta = _read_json(target)
    parameters = _required_mapping(
        fit_meta,
        "parameters",
        field_path="parameters",
        metadata_path=target,
    )
    optimized = _required_mapping(
        parameters,
        "optimized_parameters",
        field_path="parameters.optimized_parameters",
        metadata_path=target,
    )
    shape = _required_numeric_array(
        optimized,
        "shape_params",
        field_path="parameters.optimized_parameters.shape_params",
        metadata_path=target,
        dtype=np.float32,
    )
    expression = _required_numeric_array(
        optimized,
        "expression_params",
        field_path="parameters.optimized_parameters.expression_params",
        metadata_path=target,
        dtype=np.float32,
    )
    per_view = _required_mapping(
        optimized,
        "per_view",
        field_path="parameters.optimized_parameters.per_view",
        metadata_path=target,
    )
    front = _required_mapping(
        per_view,
        "front",
        field_path="parameters.optimized_parameters.per_view.front",
        metadata_path=target,
    )
    rotation = _required_numeric_array(
        front,
        "R",
        field_path="parameters.optimized_parameters.per_view.front.R",
        metadata_path=target,
        dtype=np.float64,
        size=9,
    ).reshape(3, 3)
    translation = _required_numeric_array(
        front,
        "t",
        field_path="parameters.optimized_parameters.per_view.front.t",
        metadata_path=target,
        dtype=np.float64,
        size=3,
    )
    return shape, expression, rotation, translation


def _load_baseline_flame_landmarks(
    source_output: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    import torch

    from src.module2_geometry import (
        FLAMEModel,
        load_flame_landmark_mapping,
    )

    fit_meta_path = source_output / "meshes" / "stable_fit_meta.json"
    shape, expression, fit_rotation, fit_translation = (
        load_baseline_fit_parameters(fit_meta_path)
    )

    flame = FLAMEModel(
        cfg.FLAME_MODEL_PATH,
        n_shape=len(shape),
        n_exp=len(expression),
    ).cpu()
    mapping = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
    if mapping is None:
        raise RuntimeError(
            "FLAME landmark embedding is required for projected nasal priors"
        )
    with torch.no_grad():
        vertices = flame(
            torch.from_numpy(shape),
            torch.from_numpy(expression),
        ).cpu().numpy()
    landmarks = flame_landmarks_from_embedding(
        vertices,
        flame.faces.cpu().numpy(),
        mapping,
    )
    return (
        landmarks,
        fit_rotation,
        fit_translation,
        fit_meta_path.resolve(),
    )


def _anchor_indices(subject_side: str) -> dict[str, int]:
    if subject_side not in {"subject-left", "subject-right"}:
        raise ValueError(f"unsupported subject side: {subject_side}")
    return {
        "upper_tip": int(
            BASELINE_ANCHOR_DEFINITIONS["upper_tip"]["landmark_68_index"]
        ),
        "tip_apex": int(
            BASELINE_ANCHOR_DEFINITIONS["tip_apex"]["landmark_68_index"]
        ),
        "lower_tip": int(
            BASELINE_ANCHOR_DEFINITIONS["lower_tip"]["landmark_68_index"]
        ),
        "alar_transition": int(
            BASELINE_ANCHOR_DEFINITIONS["alar_transition"][
                "landmark_68_index_by_subject_side"
            ][subject_side]
        ),
    }


def _baseline_prior_tracks(
    landmarks: np.ndarray,
    fit_rotation: np.ndarray,
    fit_translation: np.ndarray,
    rig: ProfileRig,
) -> dict[str, dict[str, dict[str, list[float]]]]:
    tracks = {}
    for subject_side in ("subject-left", "subject-right"):
        indices = _anchor_indices(subject_side)
        names = list(indices)
        points = np.asarray(
            [landmarks[indices[name]] for name in names],
            dtype=np.float64,
        )
        projected = project_flame_points_through_front_fit(
            points,
            fit_rotation,
            fit_translation,
            rig,
        )
        tracks[subject_side] = {
            view: {
                name: projected[view][index].astype(float).tolist()
                for index, name in enumerate(names)
            }
            for view in PROFILE_VIEWS
        }
    return tracks


def _validate_priors_in_frame(
    tracks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    rig: ProfileRig,
) -> None:
    for subject_side, priors_by_view in tracks.items():
        for view, priors in priors_by_view.items():
            camera = rig.cameras_by_view[view]
            values = np.asarray(list(priors.values()), dtype=np.float64)
            width, height = camera.image_size
            inside = (
                (values[:, 0] >= 0.0)
                & (values[:, 0] < width)
                & (values[:, 1] >= 0.0)
                & (values[:, 1] < height)
            )
            if not np.all(inside):
                raise ValueError(
                    f"baseline {subject_side} priors leave the "
                    f"undistorted {view} frame"
                )


def _save_preprocess_nose_masks(
    debug_dir: Path,
    preprocessed_views: Mapping[str, Mapping[str, Any]],
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for view in PROFILE_VIEWS:
        mask = np.asarray(
            preprocessed_views[view].get("nose_mask"),
            dtype=np.uint8,
        )
        target = debug_dir / f"{view}_nose_mask.png"
        write_image_file(
            target,
            mask,
            description="preprocess nose mask",
        )


def release_raw_image_references(
    raw_images: dict[str, np.ndarray],
    undistorted_images: Mapping[str, np.ndarray],
) -> None:
    """Release source image arrays unless undistortion returned the same map."""
    if raw_images is not undistorted_images:
        raw_images.clear()


def run_nasal_observation_audit(
    capture_dir: str | Path,
    source_output: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    work_size: tuple[int, int] = (640, 480),
    max_stereo_rms_px: float = 10.0,
) -> Path:
    """Run the read-only observation audit and return the offline report."""
    captures = Path(capture_dir).resolve()
    source = Path(source_output).resolve()
    target = Path(output).resolve()
    calibration = Path(rig_calibration).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if not source.is_dir():
        raise FileNotFoundError(f"source output does not exist: {source}")
    if not calibration.is_file():
        raise FileNotFoundError(f"rig calibration does not exist: {calibration}")
    validate_separate_output(target, source, captures)
    source_hashes = file_tree_hashes(source)

    try:
        original_rig = load_profile_rig(
            calibration,
            max_stereo_rms_px=max_stereo_rms_px,
        )
        raw_images, capture_paths = _load_capture_images(
            captures,
            original_rig,
        )
        from src.module0_intrinsics import (
            undistort_images_with_calibration,
        )

        undistorted_images, new_intrinsics = (
            undistort_images_with_calibration(
                raw_images,
                calibration_path=calibration,
                alpha=float(cfg.UNDISTORT_ALPHA),
            )
        )
        release_raw_image_references(raw_images, undistorted_images)
        del raw_images
        if new_intrinsics is None:
            raise RuntimeError(
                "calibrated undistortion did not return full-resolution new K"
            )
        observation_rig = build_undistorted_observation_rig(
            original_rig,
            new_intrinsics,
        )
        config = make_nasal_audit_config(work_size)
        coordinates = {
            view: ObservationCoordinates.from_camera(
                camera,
                work_size=config.work_size,
                pixel_frame="undistorted",
            )
            for view, camera in observation_rig.cameras_by_view.items()
        }

        target.mkdir(parents=True, exist_ok=True)
        preprocess_debug = target / "debug" / "preprocess"
        from src.module1_preprocess import preprocess_all_views

        preprocessed = preprocess_all_views(
            undistorted_images,
            debug_dir=preprocess_debug,
            target_size=int(cfg.WORK_IMAGE_SIZE),
        )
        front_nose_mask = require_parser_nose_mask(preprocessed)
        _save_preprocess_nose_masks(preprocess_debug, preprocessed)
        side_face_masks = {
            view: np.asarray(
                preprocessed[view]["face_mask"],
                dtype=np.uint8,
            )
            for view in ("left", "right")
        }

        (
            flame_landmarks,
            fit_rotation,
            fit_translation,
            fit_meta_path,
        ) = _load_baseline_flame_landmarks(source)
        prior_tracks = _baseline_prior_tracks(
            flame_landmarks,
            fit_rotation,
            fit_translation,
            observation_rig,
        )
        _validate_priors_in_frame(prior_tracks, observation_rig)

        front_anchors = prior_tracks["subject-left"]["front"]
        side_priors = {
            "left": prior_tracks["subject-left"]["left"],
            "right": prior_tracks["subject-right"]["right"],
        }
        center_points = np.asarray(
            [
                front_anchors["upper_tip"],
                front_anchors["tip_apex"],
                front_anchors["lower_tip"],
            ],
            dtype=np.float64,
        )
        centerline_x = float(np.median(center_points[:, 0]))
        initial_bundle = build_multiview_nasal_observations(
            images_by_view=undistorted_images,
            front_semantic_nose_mask=front_nose_mask,
            side_face_masks=side_face_masks,
            rig=observation_rig,
            front_anchors_original=front_anchors,
            side_priors_original=side_priors,
            coordinates_by_view=coordinates,
            centerline_x_original=centerline_x,
            config=config,
        )
        # A3 accepts one front track for both profiles. Rebuild the right
        # profile so its alar epipolar line uses the subject-right 3D anchor.
        subject_right = build_profile_nasal_observation(
            undistorted_images["right"],
            side_face_masks["right"],
            observation_rig,
            prior_tracks["subject-right"]["front"],
            "right",
            prior_tracks["subject-right"]["right"],
            coordinates_by_view=coordinates,
            config=config,
        )
        bundle = NasalObservationBundle(
            front=initial_bundle.front,
            subject_left=initial_bundle.subject_left,
            subject_right=subject_right,
        )

        rig_hash = _sha256_file(calibration)
        metadata = {
            "paths": {
                "capture_dir": str(captures),
                "capture_images": {
                    view: str(path)
                    for view, path in capture_paths.items()
                },
                "source_output": str(source),
                "stable_fit_meta": str(fit_meta_path),
                "preprocess_debug": str(preprocess_debug),
            },
            "source_read_only": {
                "file_count": len(source_hashes),
                "tree_sha256": _tree_digest(source_hashes),
            },
            "rig": {
                "calibration_path": str(calibration),
                "sha256": rig_hash,
                "reference_view": observation_rig.reference_view,
                "reference_camera": observation_rig.cameras_by_view[
                    observation_rig.reference_view
                ].name,
                "units": observation_rig.units,
                "stereo_rms_px": dict(observation_rig.stereo_rms_px),
                "undistortion_alpha": float(cfg.UNDISTORT_ALPHA),
                "observation_intrinsics": "full-resolution new K",
                "observation_distortion": "zero; images already undistorted",
            },
            "work_size_wh": list(config.work_size),
            "observation_config": {
                "mask_perturbation_px": config.mask_perturbation_px,
                "gradient_window_px": config.gradient_window_px,
                "front_alar_vertical_fraction": list(
                    config.front_alar_vertical_fraction
                ),
                "max_side_prior_distance_px": (
                    config.max_side_prior_distance_px
                ),
                "max_epipolar_distance_px": (
                    config.max_epipolar_distance_px
                ),
                "distance_clip_px": config.distance_clip_px,
                "confidence_spread_px": config.confidence_spread_px,
                "min_boundary_points": config.min_boundary_points,
                "reference_work_size_wh": [640, 480],
                "dataset_specific_overrides": False,
            },
            "preprocess_target_size": int(cfg.WORK_IMAGE_SIZE),
            "baseline_projected_priors": {
                "source": "FLAME 68-point barycentric embedding",
                "target_usage": (
                    "ROI and epipolar prior only; parser/silhouette are targets"
                ),
                "anchor_definitions": BASELINE_ANCHOR_DEFINITIONS,
                "landmark_indexing": "zero-based iBUG 68",
                "front_builder_anchor_track": "subject-left",
                "side_builder_anchor_tracks": {
                    "left": "subject-left",
                    "right": "subject-right",
                },
                "right_profile_rebuilt_for_same_side_front_alar": True,
                "projection_chain": (
                    "FLAME -> fit front camera2 -> fixed rig -> "
                    "undistorted full-resolution view"
                ),
                "projected_original_px_by_subject_side": prior_tracks,
            },
            "front_centerline": {
                "x_original_px": centerline_x,
                "source": (
                    "median baseline upper_tip/tip_apex/lower_tip projection"
                ),
            },
            "observation_target_sources": {
                "front": "CelebAMask-HQ parser nose boundary, label 10",
                "subject-left": "left preprocess face_mask silhouette",
                "subject-right": "right preprocess face_mask silhouette",
                "detector_landmarks_as_nasal_targets": False,
            },
            "texture_scoring_included": False,
        }
        write_nasal_observation_data(
            bundle,
            target / "nasal_observations.json",
            target / "nasal_observation_fields.npz",
            metadata=metadata,
        )
        report_path = write_nasal_observation_report(
            target / "debug" / "nasal_observations",
            bundle,
            images_by_view=undistorted_images,
            side_face_masks=side_face_masks,
            baseline_priors_by_subject_side=prior_tracks,
            coordinates_by_view=coordinates,
            centerline_x_original=centerline_x,
        )
    finally:
        assert_file_tree_unchanged(source, source_hashes)
    return report_path


def main() -> None:
    args = _parse_args()
    report = run_nasal_observation_audit(
        args.capture_dir,
        args.source_output,
        args.output,
        rig_calibration=args.rig_calibration,
        work_size=(args.work_width, args.work_height),
        max_stereo_rms_px=args.max_stereo_rms_px,
    )
    print(f"Nasal observation audit: {report}")
    print("Pure geometry observations only; texture scoring is not included.")


if __name__ == "__main__":
    main()
