"""Audit profile depth across FLAME initialization and fitting stages."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src import config as cfg
from src.geometry.profile_observations import (
    ObservationThresholds,
    collect_profile_observations,
    write_profile_points_ply,
)
from src.geometry.profile_depth_quality import (
    build_profile_depth_stage_report,
    compare_expression_depth,
    profile_depth_metrics,
)
from src.geometry.profile_triangulation import (
    TriangulationThresholds,
    load_profile_rig,
)
from src.geometry.semantic_epipolar_refinement import (
    SemanticRefinementThresholds,
)
from src.initializers.face_alignment_initializer import get_fa_per_view
from src.module1_preprocess import detect_landmarks_mediapipe
from src.module2_geometry import FLAMEModel


ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
        help="Stable reconstruction output containing meshes/stable_fit_meta.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (default: <reconstruction>/debug/profile_depth/profile_depth_baseline.json)",
    )
    parser.add_argument(
        "--max-expression-forward-mm",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--captures",
        type=Path,
        default=None,
        help="Optional three-camera capture directory for semantic 3D observations",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Fixed-rig calibration JSON; required when --captures is supplied",
    )
    parser.add_argument("--max-stereo-rms-px", type=float, default=10.0)
    parser.add_argument("--max-semantic-correction-px", type=float, default=40.0)
    parser.add_argument(
        "--dense-match-dir",
        type=Path,
        default=None,
        help=(
            "Optional joint-rig output containing accepted LoFTR NPZ files. "
            "When supplied, side semantic indices are replaced by local "
            "image-evidence mappings from the front semantic anchors."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _parameter_array(
    parameters: dict[str, Any],
    key: str,
    expected_length: int,
) -> np.ndarray:
    value = np.asarray(parameters.get(key, []), dtype=np.float32).reshape(-1)
    if len(value) != expected_length:
        raise ValueError(
            f"{key} has {len(value)} values; expected {expected_length}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _load_mesh_vertices(path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"invalid mesh vertices in {path}")
    return vertices


def audit_profile_depth(
    reconstruction: Path,
    *,
    max_expression_forward_mm: float = 0.5,
) -> dict[str, Any]:
    reconstruction = Path(reconstruction).resolve()
    mesh_dir = reconstruction / "meshes"
    fit_meta_path = mesh_dir / "stable_fit_meta.json"
    semantic_path = mesh_dir / "stable_semantic_regions.json"
    fit_meta = _read_json(fit_meta_path)
    semantic_payload = _read_json(semantic_path)
    regions = semantic_payload.get("regions")
    if not isinstance(regions, dict):
        raise ValueError(f"regions object is missing from {semantic_path}")

    optimized = (
        fit_meta.get("parameters", {})
        .get("optimized_parameters", {})
    )
    if not isinstance(optimized, dict):
        raise ValueError(f"optimized_parameters are missing from {fit_meta_path}")

    mica_shape = _parameter_array(
        optimized,
        "mica_identity_anchor",
        cfg.N_SHAPE_PARAMS,
    )
    optimized_shape = _parameter_array(
        optimized,
        "shape_params",
        cfg.N_SHAPE_PARAMS,
    )
    optimized_expression = _parameter_array(
        optimized,
        "expression_params",
        cfg.N_EXP_PARAMS,
    )

    flame = FLAMEModel(
        cfg.FLAME_MODEL_PATH,
        n_shape=cfg.N_SHAPE_PARAMS,
        n_exp=cfg.N_EXP_PARAMS,
    ).cpu()
    zero_shape = torch.zeros(cfg.N_SHAPE_PARAMS, dtype=torch.float32)
    zero_expression = torch.zeros(cfg.N_EXP_PARAMS, dtype=torch.float32)
    with torch.no_grad():
        stages = {
            "flame_mean": flame(zero_shape, zero_expression).cpu().numpy(),
            "mica_identity_anchor": flame(
                torch.from_numpy(mica_shape), zero_expression
            ).cpu().numpy(),
            "optimized_neutral": flame(
                torch.from_numpy(optimized_shape), zero_expression
            ).cpu().numpy(),
            "optimized_expression": flame(
                torch.from_numpy(optimized_shape),
                torch.from_numpy(optimized_expression),
            ).cpu().numpy(),
        }

    stage_report = build_profile_depth_stage_report(stages, regions)
    expression_gate = compare_expression_depth(
        stages["optimized_neutral"],
        stages["optimized_expression"],
        regions,
        max_mean_forward=max_expression_forward_mm,
    )

    exported = {}
    neutral_mesh_path = mesh_dir / "face_mesh_neutral.glb"
    expression_mesh_path = mesh_dir / "face_mesh.glb"
    if neutral_mesh_path.exists() and expression_mesh_path.exists():
        neutral_vertices = _load_mesh_vertices(neutral_mesh_path)
        expression_vertices = _load_mesh_vertices(expression_mesh_path)
        exported = {
            "neutral": profile_depth_metrics(neutral_vertices, regions),
            "expression": profile_depth_metrics(expression_vertices, regions),
            "expression_gate": compare_expression_depth(
                neutral_vertices,
                expression_vertices,
                regions,
                max_mean_forward=max_expression_forward_mm,
            ),
        }

    return {
        "audit_version": 1,
        "audit_only": True,
        "reconstruction": str(reconstruction),
        "source_paths": {
            "stable_fit_meta": str(fit_meta_path),
            "semantic_regions": str(semantic_path),
            "flame_model": str(cfg.FLAME_MODEL_PATH),
        },
        "semantic_source": semantic_payload.get("source"),
        "topology": semantic_payload.get("topology"),
        **stage_report,
        "expression_gate": expression_gate,
        "exported_mesh_verification": exported,
    }


def _load_capture_images(captures: Path, rig) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for view, camera in rig.cameras_by_view.items():
        candidates = sorted(captures.glob(f"{camera.name}_*.jpg"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one {camera.name}_*.jpg in {captures}, found {len(candidates)}"
            )
        image_bgr = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"cannot read {candidates[0]}")
        actual_size = (image_bgr.shape[1], image_bgr.shape[0])
        if actual_size != tuple(camera.image_size):
            raise ValueError(
                f"{camera.name}/{view} image size {actual_size} does not match "
                f"calibration {camera.image_size}"
            )
        images[view] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return images


def _draw_point(
    image: np.ndarray,
    point: list[float] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if point is None:
        return
    x, y = np.rint(point).astype(np.int32)
    cv2.circle(image, (int(x), int(y)), 14, color, 4, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (int(x) + 18, int(y) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA,
    )


def _save_resized(path: Path, image_bgr: np.ndarray, max_width: int = 1500) -> None:
    if image_bgr.shape[1] > max_width:
        scale = max_width / float(image_bgr.shape[1])
        image_bgr = cv2.resize(
            image_bgr,
            (max_width, int(round(image_bgr.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"failed to write {path}")


def _write_observation_artifacts(
    output_dir: Path,
    images: dict[str, np.ndarray],
    rig,
    observation_report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for view, image_rgb in images.items():
        raw_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        observation_overlay = raw_bgr.copy()
        for name, detector in observation_report["detectors"].get(view, {}).items():
            _draw_point(
                observation_overlay,
                detector["mediapipe_px"],
                (70, 220, 120),
                f"MP {name}",
            )
            _draw_point(
                observation_overlay,
                detector["face_alignment_px"],
                (220, 90, 220),
                f"FA {name}",
            )
        _save_resized(output_dir / f"observations_{view}.png", observation_overlay)

        camera = rig.cameras_by_view[view]
        undistorted = cv2.undistort(raw_bgr, camera.K, camera.dist, None, camera.K)
        reprojection_overlay = undistorted.copy()
        for name, point_report in observation_report["points"].items():
            observed = point_report["triangulation"]["observations_px"].get(view)
            consensus = point_report["epipolar_consensus"]["reprojected_px"].get(view)
            if observed is not None and consensus is not None:
                first = tuple(np.rint(observed).astype(np.int32))
                second = tuple(np.rint(consensus).astype(np.int32))
                cv2.line(reprojection_overlay, first, second, (40, 190, 255), 3)
            _draw_point(reprojection_overlay, observed, (255, 210, 70), f"obs {name}")
            _draw_point(reprojection_overlay, consensus, (80, 80, 255), f"rig {name}")
        _save_resized(output_dir / f"reprojection_{view}.png", reprojection_overlay)

    write_profile_points_ply(
        observation_report,
        output_dir / "triangulated_points.ply",
    )
    rows = []
    for name, point_report in observation_report["points"].items():
        triangle = point_report["triangulation"]
        consensus = point_report["epipolar_consensus"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{'PASS' if point_report['passed'] else 'REJECT'}</td>"
            f"<td>{triangle['reprojection_p90_px']:.2f}</td>"
            f"<td>{point_report['observation_view_count']}</td>"
            f"<td>{consensus['correction_p90_px']:.2f}</td>"
            f"<td>{triangle['max_pair_delta_m'] * 1000.0:.2f}</td>"
            f"<td>{html.escape(', '.join(point_report['issues']) or 'none')}</td>"
            "</tr>"
        )
    gate = observation_report["quality_gate"]
    figures = "".join(
        f"<section><h2>{view.title()}</h2>"
        f"<img src='observations_{view}.png' alt='{view} detector observations'>"
        f"<img src='reprojection_{view}.png' alt='{view} rig reprojection'></section>"
        for view in ("left", "front", "right")
    )
    refinement = observation_report.get("image_refinement")
    method_note = (
        "The semantic 3D point comes from a robust local depth surface fitted "
        "to exact LoFTR triangulations around the front semantic pixel. The "
        "displayed reprojection is constructed from that 3D point and is not "
        "an independent accuracy score; depth-surface residual and cross-side "
        "depth agreement are the independent gates."
        if refinement is not None
        else "Side observations are raw detector semantic indices."
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Profile Depth Observation Audit</title><style>
body{{margin:0;background:#10151b;color:#edf2f7;font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}}
main{{width:min(1400px,calc(100% - 32px));margin:auto;padding:24px 0 40px}}
h1{{font-size:28px}} h2{{font-size:20px;margin-top:28px}} p{{color:#aebac6;line-height:1.55}}
img{{display:block;width:100%;height:auto;margin:10px 0;border:1px solid #34404c}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:9px;border-bottom:1px solid #34404c;text-align:left}}
.pass{{color:#65d693}} .fail{{color:#ff8a8a}}
</style></head><body><main><h1>Profile Depth Observation Audit</h1>
<p class="{'pass' if gate['passed'] else 'fail'}">Gate: {'PASS' if gate['passed'] else 'REJECT'}; valid points: {gate['valid_point_count']} / {gate['minimum_valid_points']}.</p>
<p>{html.escape(method_note)}</p>
<p>Green: MediaPipe seed. Purple: independent 68-point detector. Cyan: selected observation. Red: fixed-rig reprojection. The connecting line is the remaining correction.</p>
<table><thead><tr><th>Point</th><th>Status</th><th>Reprojection P90 (px)</th><th>Views</th><th>Remaining correction P90 (px)</th><th>Pair delta (mm)</th><th>Issues</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{figures}</main></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def audit_semantic_observations(
    captures: Path,
    calibration: Path,
    *,
    device: str,
    max_stereo_rms_px: float,
    max_semantic_correction_px: float,
    output_dir: Path,
    dense_match_dir: Path | None = None,
) -> dict[str, Any]:
    rig = load_profile_rig(
        calibration,
        max_stereo_rms_px=max_stereo_rms_px,
    )
    images = _load_capture_images(captures.resolve(), rig)
    actual_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    mediapipe_landmarks = {
        view: detect_landmarks_mediapipe(image, view, max_size=1280)[0]
        for view, image in images.items()
    }
    face_alignment_landmarks = get_fa_per_view(
        images,
        actual_device,
        max_size=1280,
    )
    dense_matches = None
    if dense_match_dir is not None:
        dense_matches = _load_dense_matches(
            dense_match_dir.resolve(),
            captures.resolve().name,
            rig,
        )
    report = collect_profile_observations(
        mediapipe_landmarks,
        face_alignment_landmarks,
        rig,
        {view: image.shape for view, image in images.items()},
        dense_matches_by_view=dense_matches,
        semantic_refinement_thresholds=SemanticRefinementThresholds(),
        images_by_view=images,
        observation_thresholds=ObservationThresholds(
            max_consensus_correction_px=max_semantic_correction_px,
        ),
        triangulation_thresholds=TriangulationThresholds(
            max_reprojection_px=5.0,
        ),
    )
    report["captures"] = str(captures.resolve())
    report["device"] = actual_device
    _write_observation_artifacts(output_dir, images, rig, report)
    return report


def _load_dense_matches(
    dense_match_dir: Path,
    dataset: str,
    rig,
) -> dict[str, dict[str, np.ndarray]]:
    front_camera = rig.cameras_by_view["front"]
    matches: dict[str, dict[str, np.ndarray]] = {}
    for side_view in ("left", "right"):
        side_camera = rig.cameras_by_view[side_view]
        pair = f"{front_camera.name}_{side_camera.name}"
        path = dense_match_dir / f"{dataset}_{pair}_accepted_matches.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing dense semantic match file: {path}")
        with np.load(path, allow_pickle=False) as payload:
            work_size = tuple(int(value) for value in payload["work_size"])
            if work_size != (640, 480):
                raise ValueError(f"unsupported dense match work size in {path}: {work_size}")
            matches[side_view] = {
                "points_front_work": np.asarray(
                    payload["points_front_work"], dtype=np.float64
                ),
                "points_side_work": np.asarray(
                    payload["points_side_work"], dtype=np.float64
                ),
                "confidence": np.asarray(payload["confidence"], dtype=np.float64),
            }
    return matches


def main() -> None:
    args = _parse_args()
    if (args.captures is None) != (args.calibration is None):
        raise ValueError("--captures and --calibration must be supplied together")
    if args.dense_match_dir is not None and args.captures is None:
        raise ValueError("--dense-match-dir requires --captures and --calibration")
    reconstruction = args.reconstruction.resolve()
    output_path = args.output
    if output_path is None:
        output_path = (
            reconstruction
            / "debug"
            / "profile_depth"
            / "profile_depth_baseline.json"
        )
    output_path = output_path.resolve()
    report = audit_profile_depth(
        reconstruction,
        max_expression_forward_mm=args.max_expression_forward_mm,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.captures is not None:
        observation_report = audit_semantic_observations(
            args.captures,
            args.calibration,
            device=args.device,
            max_stereo_rms_px=args.max_stereo_rms_px,
            max_semantic_correction_px=args.max_semantic_correction_px,
            output_dir=output_path.parent,
            dense_match_dir=args.dense_match_dir,
        )
        report["semantic_observations"] = observation_report
        with (output_path.parent / "profile_depth_quality.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(observation_report, handle, ensure_ascii=False, indent=2)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"Profile depth audit: {output_path}")
    for name, metrics in report["stages"].items():
        print(
            f"  {name}: nose lead={metrics['nose_tip_minus_mouth']:.3f} mm, "
            f"mouth-chin={metrics['mouth_minus_chin']:.3f} mm"
        )
    gate = report["expression_gate"]
    print(
        "  expression mouth forward: "
        f"mean={gate['mouth_forward_mean']:.3f} mm, "
        f"p95={gate['mouth_forward_p95']:.3f} mm, "
        f"passed={gate['passed']}"
    )
    observation_report = report.get("semantic_observations")
    if observation_report is not None:
        observation_gate = observation_report["quality_gate"]
        print(
            "  semantic 3D observations: "
            f"valid={observation_gate['valid_point_count']}/"
            f"{observation_gate['minimum_valid_points']}, "
            f"passed={observation_gate['passed']}"
        )
        for name, point_report in observation_report["points"].items():
            print(
                f"    {name}: reprojection P90="
                f"{point_report['triangulation']['reprojection_p90_px']:.2f}px, "
                f"required correction P90="
                f"{point_report['epipolar_consensus']['correction_p90_px']:.2f}px, "
                f"passed={point_report['passed']}"
            )


if __name__ == "__main__":
    main()
