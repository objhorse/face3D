"""Run the A1 strict-projective texture truth diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src import config as cfg
from src.appearance.baseline_texture_lock import BaselineTextureLock
from src.module0_intrinsics import get_intrinsics, undistort_images_with_calibration
from src.module1_preprocess import load_images, preprocess_all_views
from src.module3_texture import load_cameras
from src.reports.projective_texture_report import (
    publish_report_directory_transactionally,
    write_projective_texture_truth_report,
)


_CAPTURE_PATTERN = re.compile(r"^camera([123])(?:_.*)?\.(?:jpe?g)$", re.IGNORECASE)
_CAMERA_TO_VIEW = {
    "1": "left",
    "2": "front",
    "3": "right",
}
K_MAX_ABS_TOLERANCE = 1e-4


class DiagnosticVerificationError(RuntimeError):
    """Both the diagnostic operation and baseline verification failed."""

    def __init__(
        self,
        operation_error: Exception,
        verification_error: Exception,
    ) -> None:
        self.operation_error = operation_error
        self.verification_error = verification_error
        self.operation_traceback = "".join(
            traceback.format_exception(
                type(operation_error),
                operation_error,
                operation_error.__traceback__,
            )
        )
        self.verification_traceback = "".join(
            traceback.format_exception(
                type(verification_error),
                verification_error,
                verification_error.__traceback__,
            )
        )
        super().__init__(
            "diagnostic operation and baseline verification both failed: "
            f"{type(operation_error).__name__}: {operation_error}; "
            f"{type(verification_error).__name__}: {verification_error}"
        )


def _prepare_legacy_registration(**kwargs):
    from src.appearance.stable_texture_registration import (
        prepare_stable_texture_registration,
    )

    return prepare_stable_texture_registration(**kwargs)


def _validate_non_overlapping_paths(
    capture_dir: Path,
    mesh_dir: Path,
    output_dir: Path,
) -> None:
    for input_name, input_path in (
        ("capture_dir", capture_dir),
        ("mesh_dir", mesh_dir),
    ):
        overlaps = (
            output_dir == input_path
            or output_dir in input_path.parents
            or input_path in output_dir.parents
        )
        if overlaps:
            raise ValueError(
                "output_dir and input directories must not overlap: "
                f"output_dir={output_dir}, {input_name}={input_path}"
            )


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"calibration file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_sha256(image: np.ndarray) -> str:
    array = np.asarray(image)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _verify_image_hashes(images: dict[str, np.ndarray], expected: dict[str, str]) -> None:
    changed = [
        view
        for view, image in images.items()
        if _image_sha256(image) != expected[view]
    ]
    if changed:
        raise RuntimeError(
            "projective truth images changed during legacy diagnostics: "
            + ", ".join(sorted(changed))
        )


def _verify_capture_provenance(provenance: dict) -> None:
    changed = []
    for view, source in provenance["capture_images"].items():
        if _file_sha256(Path(source["path"])) != source["sha256"]:
            changed.append(view)
    if changed:
        raise RuntimeError(
            "capture images changed during projective truth diagnostics: "
            + ", ".join(sorted(changed))
        )


def _validate_preprocess_intrinsics(
    intrinsics: dict,
    saved_cameras: dict,
    views: Sequence[str],
) -> dict:
    view_metrics = {}
    for view in views:
        if view not in intrinsics:
            raise ValueError(f"recomputed intrinsics are missing view '{view}'")
        if view not in saved_cameras:
            raise ValueError(f"saved cameras are missing view '{view}'")
        recomputed_k = np.asarray(intrinsics[view], dtype=np.float64)
        saved_k = np.asarray(saved_cameras[view].get("K"), dtype=np.float64)
        if recomputed_k.shape != (3, 3) or saved_k.shape != (3, 3):
            raise ValueError(f"{view} camera K must have shape (3, 3)")
        if not np.isfinite(recomputed_k).all() or not np.isfinite(saved_k).all():
            raise ValueError(f"{view} camera K must contain only finite values")
        max_abs_error = float(np.max(np.abs(recomputed_k - saved_k)))
        if max_abs_error > K_MAX_ABS_TOLERANCE:
            raise ValueError(
                f"{view} camera K mismatch: max abs error "
                f"{max_abs_error:.9g} exceeds {K_MAX_ABS_TOLERANCE:.9g}"
            )
        view_metrics[view] = {
            "k_max_abs_error": max_abs_error,
            "validated": True,
        }
    return view_metrics


def _build_preprocess_provenance(
    *,
    intrinsics: dict,
    saved_cameras: dict,
    views: Sequence[str],
    calibration_path: Path,
    work_image_size: int,
    undistort_images: bool,
    undistort_alpha: float,
    capture_images: dict[str, Path],
    manual_intrinsics,
) -> dict:
    calibration_path = Path(calibration_path).resolve()
    view_metrics = _validate_preprocess_intrinsics(
        intrinsics,
        saved_cameras,
        views,
    )
    return {
        "schema_version": 2,
        "validated": True,
        "work_image_size": int(work_image_size),
        "undistort_images": bool(undistort_images),
        "undistort_alpha": float(undistort_alpha),
        "calibration": {
            "path": str(calibration_path),
            "sha256": _file_sha256(calibration_path),
            "used_for_undistortion": bool(undistort_images),
        },
        "intrinsics_source": {
            "mode": (
                "manual_intrinsics"
                if manual_intrinsics is not None
                else "calibration_or_pipeline_default"
            ),
            "resolved_k": {
                view: np.asarray(intrinsics[view], dtype=np.float64).tolist()
                for view in sorted(views)
            },
        },
        "camera_to_subject_view": {
            f"camera{camera}": view for camera, view in _CAMERA_TO_VIEW.items()
        },
        "capture_images": {
            view: {
                "camera": next(
                    f"camera{camera}"
                    for camera, mapped_view in _CAMERA_TO_VIEW.items()
                    if mapped_view == view
                ),
                "path": str(Path(capture_images[view]).resolve()),
                "sha256": _file_sha256(Path(capture_images[view]).resolve()),
            }
            for view in sorted(views)
        },
        "k_max_abs_tolerance": K_MAX_ABS_TOLERANCE,
        "views": view_metrics,
    }


def _write_preprocess_provenance(output_dir: Path, provenance: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preprocess_provenance.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            provenance,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    return output_path


def discover_capture_images(capture_dir: Path) -> dict[str, Path]:
    """Find exactly one JPEG for each fixed-rig camera.

    View names describe the subject: camera1 observes the subject's left side,
    camera2 is frontal, and camera3 observes the subject's right side.
    """
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {capture_dir}")

    matches: dict[str, list[Path]] = {camera: [] for camera in _CAMERA_TO_VIEW}
    for candidate in capture_dir.iterdir():
        if not candidate.is_file():
            continue
        matched = _CAPTURE_PATTERN.fullmatch(candidate.name)
        if matched is not None:
            matches[matched.group(1)].append(candidate)

    errors = []
    for camera in _CAMERA_TO_VIEW:
        found = sorted(matches[camera], key=lambda path: path.name.casefold())
        matches[camera] = found
        if len(found) != 1:
            names = ", ".join(path.name for path in found) or "none"
            errors.append(
                f"camera{camera}: expected exactly one JPG/JPEG, "
                f"found {len(found)} ({names})"
            )
    if errors:
        raise ValueError(
            f"invalid three-camera capture set in {capture_dir}: " + "; ".join(errors)
        )

    return {
        _CAMERA_TO_VIEW[camera]: matches[camera][0]
        for camera in ("1", "2", "3")
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an unwarped projective texture truth report."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--with-legacy-warp",
        action="store_true",
        help="Show legacy warp displacement as diagnostics only.",
    )
    return parser.parse_args(argv)


def run_diagnostic(
    capture_dir: Path,
    mesh_dir: Path,
    output_dir: Path,
    *,
    with_legacy_warp: bool = False,
    manual_intrinsics=None,
) -> dict:
    """Preprocess three views and write the immutable A1 truth report."""
    capture_dir = Path(capture_dir).resolve()
    mesh_dir = Path(mesh_dir).resolve()
    output_dir = Path(output_dir).resolve()
    _validate_non_overlapping_paths(capture_dir, mesh_dir, output_dir)

    baseline_lock = BaselineTextureLock.capture(
        mesh_dir,
        ["face_mesh.obj", "cameras.json"],
    )
    operation_error: Optional[BaseException] = None
    metrics = None
    provenance = None
    candidate_output_dir = (
        output_dir.parent
        / f".{output_dir.name}.candidate-{uuid.uuid4().hex}"
    ).resolve()
    try:
        try:
            capture_images = discover_capture_images(capture_dir)
            image_names = {
                view: image_path.name for view, image_path in capture_images.items()
            }
            raw_images = load_images(capture_dir, image_names)
            undistort_enabled = bool(getattr(cfg, "UNDISTORT_IMAGES", True))
            calibration_intrinsics = None
            if undistort_enabled:
                hires_images, _calibration_intrinsics = (
                    undistort_images_with_calibration(
                        raw_images,
                        cfg.CAMERA_CALIBRATION_PATH,
                        alpha=cfg.UNDISTORT_ALPHA,
                    )
                )
                calibration_intrinsics = _calibration_intrinsics
            else:
                hires_images = raw_images

            effective_manual_intrinsics = (
                manual_intrinsics
                if manual_intrinsics is not None
                else getattr(cfg, "MANUAL_INTRINSICS", None)
            )
            intrinsics = get_intrinsics(
                images=hires_images,
                manual_intrinsics=effective_manual_intrinsics,
                calibration_path=cfg.CAMERA_CALIBRATION_PATH,
                calibration_intrinsics=calibration_intrinsics,
                work_image_size=cfg.WORK_IMAGE_SIZE,
                dust3r_dir=cfg.DUST3R_DIR,
            )
            saved_cameras = load_cameras(mesh_dir / "cameras.json")
            provenance = _build_preprocess_provenance(
                intrinsics=intrinsics,
                saved_cameras=saved_cameras,
                views=tuple(sorted(hires_images)),
                calibration_path=cfg.CAMERA_CALIBRATION_PATH,
                work_image_size=cfg.WORK_IMAGE_SIZE,
                undistort_images=undistort_enabled,
                undistort_alpha=cfg.UNDISTORT_ALPHA,
                capture_images=capture_images,
                manual_intrinsics=effective_manual_intrinsics,
            )

            preprocess_debug_dir = (
                output_dir.parent / f"{output_dir.name}_preprocess_a1"
            ).resolve()
            assert preprocess_debug_dir != output_dir, (
                "preprocess debug directory must be a sibling of the report directory"
            )
            preprocessed = preprocess_all_views(
                hires_images,
                debug_dir=preprocess_debug_dir,
                target_size=cfg.WORK_IMAGE_SIZE,
            )
            work_images = {
                view: np.ascontiguousarray(view_data["image"]).copy()
                for view, view_data in preprocessed.items()
            }
            work_image_hashes = {
                view: _image_sha256(image) for view, image in work_images.items()
            }
            for image in work_images.values():
                image.setflags(write=False)

            sampling_warps = None
            if with_legacy_warp:
                legacy_debug_dir = (
                    output_dir.parent
                    / f"{output_dir.name}_legacy_registration_a1"
                ).resolve()
                assert legacy_debug_dir != output_dir, (
                    "legacy debug directory must be a sibling of the report directory"
                )
                registration = _prepare_legacy_registration(
                    mesh_dir=mesh_dir,
                    preprocessed_views=copy.deepcopy(preprocessed),
                    hires_images={
                        view: np.ascontiguousarray(image).copy()
                        for view, image in hires_images.items()
                    },
                    cfg=cfg,
                    debug_dir=legacy_debug_dir,
                )
                sampling_warps = registration["sampling_warps"]
                _verify_image_hashes(work_images, work_image_hashes)

            semantic_path = mesh_dir / "stable_semantic_regions.json"
            if not semantic_path.is_file():
                semantic_path = None
            else:
                semantic_path = semantic_path.resolve()

            metrics = write_projective_texture_truth_report(
                mesh_dir=mesh_dir,
                images=work_images,
                output_dir=candidate_output_dir,
                expected_image_size=(cfg.WORK_IMAGE_SIZE, cfg.WORK_IMAGE_SIZE),
                sampling_warps=sampling_warps,
                semantic_regions_path=semantic_path,
            )
            _verify_image_hashes(work_images, work_image_hashes)
            _verify_capture_provenance(provenance)
        except BaseException as error:
            operation_error = error
            raise
        finally:
            try:
                baseline_lock.verify()
            except BaseException as verification_error:
                if operation_error is None:
                    raise
                if isinstance(operation_error, (KeyboardInterrupt, SystemExit)):
                    operation_error.verification_error = verification_error
                    operation_error.verification_traceback = "".join(
                        traceback.format_exception(
                            type(verification_error),
                            verification_error,
                            verification_error.__traceback__,
                        )
                    )
                    raise operation_error.with_traceback(
                        operation_error.__traceback__
                    ) from None
                if isinstance(operation_error, Exception) and isinstance(
                    verification_error, Exception
                ):
                    raise DiagnosticVerificationError(
                        operation_error,
                        verification_error,
                    ) from verification_error
                raise

        if metrics is None or provenance is None:
            raise RuntimeError("diagnostic completed without metrics or provenance")
        _write_preprocess_provenance(candidate_output_dir, provenance)
        publish_report_directory_transactionally(candidate_output_dir, output_dir)
        return metrics
    finally:
        shutil.rmtree(candidate_output_dir, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    run_diagnostic(
        args.capture_dir,
        args.mesh_dir,
        output_dir,
        with_legacy_warp=args.with_legacy_warp,
    )
    print(f"Projective texture truth report: {(output_dir / 'index.html').resolve()}")
    print(f"Truth metrics: {(output_dir / 'truth_metrics.json').resolve()}")


if __name__ == "__main__":
    main()
