"""Run a pure-geometry semantic eyelid A/B experiment on a frozen baseline."""

from __future__ import annotations

import argparse
import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from src.appearance.baseline_texture_lock import BaselineTextureLock
from src.geometry.eye_state import aggregate_eye_state, eye_view_evidence
from src.geometry.eyelid_fit import (
    EYE_LANDMARKS,
    EyelidFitConfig,
    fit_semantic_eyelids,
    project_landmarks,
)
from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig


ROOT = Path(__file__).resolve().parent
BASELINE_NAME = "captures_20260612_135253_controlled_identity_v1"
BASELINE_HASHES = {
    "face_mesh.obj": "ED1CC8B48AFB781788B4D0AC11F5F84F9169AE81CB39E57CA4EF74AE2BE2BF5E",
    "cameras.json": "36F7D0D2AD7E7E8F2B2812EAEB103C26A1A18C569402B21CAC39E28AB596C65C",
    "face.glb": "B17558815F862DA9EE86999551294D082C316BBF8CEE1CD3219638B8AE311860",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=ROOT.parent.parent / "captures_20260612_135253",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "output" / "experiments" / BASELINE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "output"
            / "experiments"
            / "captures_20260612_135253_semantic_eyelid_v1"
        ),
    )
    parser.add_argument("--support-rings", type=int, default=16)
    return parser.parse_args()


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    names = {}
    for view, camera in (("left", "camera1"), ("front", "camera2"), ("right", "camera3")):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {camera} JPG in {capture_dir}, found {len(matches)}"
            )
        names[view] = matches[0].name
    return names


def _prepare_landmarks(capture_dir: Path, debug_dir: Path) -> tuple[dict, dict]:
    from src import config as cfg
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import (
        _compute_resize_params,
        _resize_to_target,
        detect_landmarks_mediapipe,
        load_images,
    )

    images = load_images(capture_dir, _capture_image_names(capture_dir))
    images, _ = undistort_images_with_calibration(
        images,
        cfg.CAMERA_CALIBRATION_PATH,
        alpha=cfg.UNDISTORT_ALPHA,
    )
    landmarks = {}
    canvases = {}
    debug_dir.mkdir(parents=True, exist_ok=True)
    for view, image in images.items():
        detected, _visibility = detect_landmarks_mediapipe(image, view)
        if detected is None:
            raise RuntimeError(f"MediaPipe did not detect a face in {view}")
        scale, _new_w, _new_h, x_offset, y_offset = _compute_resize_params(
            image.shape,
            cfg.WORK_IMAGE_SIZE,
        )
        working = detected.copy()
        working[:, 0] = working[:, 0] * scale + x_offset
        working[:, 1] = working[:, 1] * scale + y_offset
        landmarks[view] = working.astype(np.float32)
        canvases[view] = _resize_to_target(image, cfg.WORK_IMAGE_SIZE)
    return landmarks, canvases


def _draw_eye_overlay(
    image: np.ndarray,
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    output_path: Path,
) -> None:
    canvas = np.asarray(image).copy()
    paths = ((36, 37, 38, 39, 40, 41), (42, 43, 44, 45, 46, 47))
    for points, color in (
        (observed, (210, 60, 210)),
        (baseline, (30, 215, 255)),
        (candidate, (80, 235, 80)),
    ):
        for indices in paths:
            polygon = np.rint(points[np.asarray(indices)]).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
            for point in polygon:
                cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "observed", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 60, 210), 2)
    cv2.putText(canvas, "baseline", (170, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 215, 255), 2)
    cv2.putText(canvas, "candidate", (310, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 235, 80), 2)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _write_embedded_viewer(
    template_path: Path,
    baseline_glb: Path,
    candidate_glb: Path,
    output_path: Path,
    report: dict,
) -> Path:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    encoded = {
        "baseline": base64.b64encode(baseline_glb.read_bytes()).decode("ascii"),
        "candidate": base64.b64encode(candidate_glb.read_bytes()).decode("ascii"),
    }
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("baseline: '"):
            lines[index] = f"      baseline: '{encoded['baseline']}',"
        elif stripped.startswith("candidate: '"):
            lines[index] = f"      candidate: '{encoded['candidate']}',"
    html = "\n".join(lines)
    html = html.replace("Controlled Identity Comparison", "Semantic Eyelid Geometry A/B")
    html = html.replace(
        "Candidate: front +18.8%, profiles preserved, rejected by 30% gate",
        (
            "Candidate: semantic eyelid rig, "
            f"eye shape {report['eye_shape_error_before_px']:.2f}px -> "
            f"{report['eye_shape_error_after_px']:.2f}px"
        ),
    )
    html = html.replace(
        "Baseline selected for final output",
        "Baseline: frozen user-approved geometry",
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    from src import config as cfg
    from src.module2_geometry import (
        FLAMEModel,
        _mediapipe_to_68,
        export_mesh_glb,
        export_mesh_obj,
        load_flame_landmark_mapping,
    )
    from src.module3_texture import load_cameras, load_mesh_obj

    args = _parse_args()
    capture_dir = args.capture_dir.resolve()
    baseline_dir = args.baseline.resolve()
    output_dir = args.output.resolve()
    source_mesh_dir = baseline_dir / "meshes"
    mesh_dir = output_dir / "meshes"
    debug_dir = output_dir / "debug"
    overlay_dir = debug_dir / "projection_overlays"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    lock = BaselineTextureLock(root=source_mesh_dir, hashes=BASELINE_HASHES)
    verified_hashes = lock.verify()
    lock.write_manifest(output_dir / "baseline_lock.json")
    logger.info("Frozen baseline verified: %s", source_mesh_dir)

    vertices, faces, uv, uv_faces = load_mesh_obj(source_mesh_dir / "face_mesh.obj")
    cameras = load_cameras(source_mesh_dir / "cameras.json")
    mp_landmarks, canvases = _prepare_landmarks(capture_dir, debug_dir / "landmarks")
    evidence = {view: eye_view_evidence(points) for view, points in mp_landmarks.items()}
    consensus = aggregate_eye_state(evidence)
    view_eye_weights = {}
    for view, view_evidence in evidence.items():
        view_eye_weights[view] = {}
        for eye_name, state in view_evidence.eyes.items():
            if state.state == consensus.states[eye_name]:
                weight = max(0.25, float(state.confidence))
            elif state.state == "uncertain":
                weight = 0.2
            else:
                weight = 0.05
            view_eye_weights[view][eye_name] = float(weight)
    (debug_dir / "eye_state.json").write_text(
        json.dumps(consensus.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Eye-state consensus: %s", consensus.states)

    flame = FLAMEModel(cfg.FLAME_MODEL_PATH, n_shape=1, n_exp=1)
    mapping = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
    if mapping is None:
        raise RuntimeError("FLAME landmark mapping is required")
    flame_faces = flame.faces.detach().cpu().numpy()
    landmark_triangles = flame_faces[np.asarray(mapping["face_idx"], dtype=np.int64)]
    barycentric = np.asarray(mapping["bary_coords"], dtype=np.float32)
    observed = {view: _mediapipe_to_68(points) for view, points in mp_landmarks.items()}

    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        landmark_triangles,
        support_rings=int(args.support_rings),
        core_rings=2,
        sigma_rings=3.0,
    )
    face_width = float(np.ptp(vertices[:, 0]))
    fit_config = EyelidFitConfig(
        max_control_offset=0.015 * face_width,
        bulge_prior=0.003 * face_width,
        regularization_weight=0.05,
        symmetry_weight=0.05,
        gap_weight=0.35,
        bulge_weight=0.04,
        max_nfev=350,
    )
    result = fit_semantic_eyelids(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=landmark_triangles,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observed,
        eye_states=consensus.states,
        view_eye_weights=view_eye_weights,
        cfg=fit_config,
    )

    baseline_obj = mesh_dir / "baseline_geometry.obj"
    candidate_obj = mesh_dir / "eyelid_candidate.obj"
    baseline_glb = mesh_dir / "baseline_geometry.glb"
    candidate_glb = mesh_dir / "eyelid_candidate.glb"
    export_mesh_obj(vertices, faces, uv, uv_faces, baseline_obj)
    export_mesh_obj(result.vertices, faces, uv, uv_faces, candidate_obj)
    export_mesh_glb(vertices, faces, uv, uv_faces, baseline_glb)
    export_mesh_glb(result.vertices, faces, uv, uv_faces, candidate_glb)

    control_report = rig.to_dict()
    control_report["parameters"] = {
        name: {
            "vertical": float(result.parameters[index, 0]),
            "normal": float(result.parameters[index, 1]),
            "offset_xyz": result.control_offsets[index].astype(float).tolist(),
        }
        for index, name in enumerate(rig.control_names)
    }
    (debug_dir / "eyelid_controls.json").write_text(
        json.dumps(control_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    quality_report = dict(result.report)
    quality_report.update(
        {
            "baseline_hashes": verified_hashes,
            "capture_dir": str(capture_dir),
            "baseline_dir": str(baseline_dir),
            "camera_mapping": {"camera1": "left", "camera2": "front", "camera3": "right"},
            "eye_states": dict(consensus.states),
            "conflicting_views": list(consensus.conflicting_views),
            "view_eye_weights": view_eye_weights,
            "face_width": face_width,
            "support_rings": int(args.support_rings),
            "topology": {"vertices": int(len(vertices)), "faces": int(len(faces))},
        }
    )
    (debug_dir / "eyelid_quality.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for view, camera in cameras.items():
        baseline_projection = project_landmarks(
            vertices, landmark_triangles, barycentric, camera
        )
        candidate_projection = project_landmarks(
            result.vertices, landmark_triangles, barycentric, camera
        )
        _draw_eye_overlay(
            canvases[view],
            observed[view],
            baseline_projection,
            candidate_projection,
            overlay_dir / f"{view}_eye_projection.png",
        )

    template_path = baseline_dir / "controlled_identity_compare.html"
    viewer_path = output_dir / "semantic_eyelid_compare.html"
    _write_embedded_viewer(
        template_path,
        baseline_glb,
        candidate_glb,
        viewer_path,
        quality_report,
    )
    from render_semantic_eyelid_diagnostics import render_semantic_eyelid_comparison

    render_semantic_eyelid_comparison(
        baseline_obj=baseline_obj,
        candidate_obj=candidate_obj,
        controls_json=debug_dir / "eyelid_controls.json",
        output_dir=output_dir / "renders",
    )
    logger.info("Fit accepted: %s (%s)", result.report["accepted"], result.report["reason"])
    logger.info(
        "Eye shape error: %.3fpx -> %.3fpx",
        result.report["eye_shape_error_before_px"],
        result.report["eye_shape_error_after_px"],
    )
    logger.info("Viewer: %s", viewer_path)


if __name__ == "__main__":
    main()
