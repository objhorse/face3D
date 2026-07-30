"""Build a textured A/B model with expression-only profile depth constraints."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mouth-mean-mm", type=float, default=0.5)
    parser.add_argument("--mouth-p95-mm", type=float, default=1.5)
    parser.add_argument("--nose-mean-mm", type=float, default=0.5)
    parser.add_argument("--chin-mean-mm", type=float, default=0.5)
    parser.add_argument("--rebake", action="store_true")
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    for view, camera in (("left", "camera1"), ("front", "camera2"), ("right", "camera3")):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {camera} JPG in {capture_dir}, found {len(matches)}"
            )
        names[view] = matches[0].name
    return names


def _default_viewer_template() -> Path:
    return (
        ROOT
        / "output"
        / "experiments"
        / "captures_20260612_135253_controlled_identity_textured_v1"
        / "textured_identity_compare.html"
    )


def _write_embedded_compare_viewer(
    template_path: Path,
    baseline_glb: Path,
    candidate_glb: Path,
    output_path: Path,
    *,
    baseline_label: str = "Baseline: original expression",
    candidate_label: str = "Candidate: protected mouth-depth expression",
    title: str = "Expression profile depth A/B",
) -> Path:
    if not template_path.exists():
        raise FileNotFoundError(f"viewer template not found: {template_path}")
    encoded = {
        "baseline": base64.b64encode(baseline_glb.read_bytes()).decode("ascii"),
        "candidate": base64.b64encode(candidate_glb.read_bytes()).decode("ascii"),
    }
    lines = template_path.read_text(encoding="utf-8").splitlines()
    found = set()
    for index, line in enumerate(lines):
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped.startswith("baseline: '"):
            lines[index] = f"{indent}baseline: '{encoded['baseline']}',"
            found.add("baseline")
        elif stripped.startswith("candidate: '"):
            lines[index] = f"{indent}candidate: '{encoded['candidate']}',"
            found.add("candidate")
        elif 'id="baseline-label"' in line:
            lines[index] = (
                f'{indent}<div id="baseline-label" class="label">'
                f"{baseline_label}</div>"
            )
        elif 'id="candidate-label"' in line:
            lines[index] = (
                f'{indent}<div id="candidate-label" class="label">'
                f"{candidate_label}</div>"
            )
        elif "<title>" in line:
            lines[index] = f"{indent}<title>{title}</title>"
    if found != {"baseline", "candidate"}:
        raise RuntimeError(f"viewer template is missing embedded model slots: {template_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _load_fit_payload(source_mesh_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = source_mesh_dir / "stable_fit_meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        parameters = payload["parameters"]["optimized_parameters"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"optimized FLAME parameters are missing from {path}") from exc
    if not parameters.get("shape_params") or not parameters.get("expression_params"):
        raise RuntimeError(f"shape/expression parameters are empty in {path}")
    return payload, parameters


def _export_depth_safe_geometry(
    *,
    source_mesh_dir: Path,
    output_mesh_dir: Path,
    thresholds: Any,
    cfg: Any,
) -> dict[str, Any]:
    import torch
    import trimesh

    from src.geometry.expression_depth import (
        constrain_expression_mouth_depth_protected,
        expression_regions_from_landmarks,
    )
    from src.module2_geometry import (
        FLAMEModel,
        _get_flame_uv,
        export_mesh_glb,
        export_mesh_obj,
        load_flame_landmark_mapping,
    )

    fit_payload, optimized = _load_fit_payload(source_mesh_dir)
    shape = np.asarray(optimized["shape_params"], dtype=np.float32)
    expression = np.asarray(optimized["expression_params"], dtype=np.float32)
    flame = FLAMEModel(
        Path(cfg.FLAME_MODEL_PATH),
        n_shape=len(shape),
        n_exp=len(expression),
    ).cpu()
    faces = flame.faces.detach().cpu().numpy()
    landmark_data = load_flame_landmark_mapping(Path(cfg.FLAME_LANDMARK_PATH))
    if landmark_data is None:
        raise RuntimeError("FLAME landmark embedding is required for expression depth regions")
    landmark_triangles = faces[np.asarray(landmark_data["face_idx"], dtype=np.int64)]
    regions = expression_regions_from_landmarks(landmark_triangles, flame.n_verts)
    basis = (
        flame.exp_basis.detach()
        .cpu()
        .numpy()
        .reshape(flame.n_verts, 3, flame.n_exp)
    )
    result = constrain_expression_mouth_depth_protected(
        basis,
        expression,
        regions,
        thresholds=thresholds,
    )
    if not result["selected"]["passed"]:
        raise RuntimeError(f"expression depth gate failed: {result['selected']['issues']}")
    selected_expression = np.asarray(result["parameters"], dtype=np.float32)

    with torch.no_grad():
        shape_tensor = torch.from_numpy(shape)
        vertices = flame(shape_tensor, torch.from_numpy(selected_expression)).cpu().numpy()
        neutral_vertices = flame(
            shape_tensor,
            torch.zeros(flame.n_exp, dtype=torch.float32),
        ).cpu().numpy()

    uv_vertices, uv_faces = _get_flame_uv(Path(cfg.FLAME_MODEL_PATH), faces)
    vertices_sub, faces_sub = trimesh.remesh.subdivide_loop(
        vertices, faces, iterations=2
    )
    neutral_sub, neutral_faces = trimesh.remesh.subdivide_loop(
        neutral_vertices, faces, iterations=2
    )
    if not np.array_equal(faces_sub, neutral_faces):
        raise RuntimeError("neutral and expression subdivision topology diverged")
    uv_sub, uv_faces_sub = uv_vertices, uv_faces
    for _ in range(2):
        uv_sub, uv_faces_sub = trimesh.remesh.subdivide(uv_sub, uv_faces_sub)
    if len(faces_sub) != len(uv_faces_sub):
        raise RuntimeError("geometry and UV subdivision topology diverged")

    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    export_mesh_obj(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_mesh.obj",
    )
    export_mesh_glb(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_mesh.glb",
    )
    export_mesh_glb(
        neutral_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        output_mesh_dir / "face_stable_neutral.glb",
    )
    shutil.copy2(output_mesh_dir / "face_mesh.glb", output_mesh_dir / "face_stable_geometry.glb")

    optimized["expression_params"] = selected_expression.tolist()
    optimized["expression_depth_constraint"] = _jsonable(result)
    fit_payload["parameters"]["optimized_parameters"] = optimized
    (output_mesh_dir / "stable_fit_meta.json").write_text(
        json.dumps(fit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "constraint": _jsonable(result),
        "vertices": int(len(vertices_sub)),
        "faces": int(len(faces_sub)),
        "regions": {name: values.tolist() for name, values in regions.items()},
    }


def _export_with_baseline_texture(
    *,
    mesh_path: Path,
    texture_path: Path,
    output_path: Path,
) -> None:
    from PIL import Image

    from src.module3_texture import (
        export_glb,
        load_mesh_obj,
        transparent_bottom_face_mask,
    )

    vertices, faces, uv_vertices, uv_faces = load_mesh_obj(mesh_path)
    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    transparent_faces, _ = transparent_bottom_face_mask(vertices, faces, 0.05)
    export_glb(
        vertices,
        faces,
        uv_vertices,
        uv_faces,
        texture,
        output_path,
        lighting_type="white",
        lighting_display_name="white",
        smooth_geometry=False,
        transparent_face_mask=transparent_faces,
    )


def _rebake_texture(
    *,
    capture_dir: Path,
    output: Path,
    cfg: Any,
) -> dict[str, Any]:
    from src.appearance.stable_texture import run_stable_texture_pipeline
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import load_images, preprocess_all_views

    raw_images = load_images(capture_dir, _capture_image_names(capture_dir))
    hires_images, _ = undistort_images_with_calibration(
        raw_images,
        cfg.CAMERA_CALIBRATION_PATH,
        alpha=cfg.UNDISTORT_ALPHA,
    )
    preprocessed = preprocess_all_views(
        hires_images,
        debug_dir=output / "debug" / "preprocess",
        target_size=cfg.WORK_IMAGE_SIZE,
    )
    return run_stable_texture_pipeline(
        mesh_dir=output / "meshes",
        images={view: data["image"] for view, data in preprocessed.items()},
        output_texture_dir=output / "textures",
        output_mesh_dir=output / "meshes",
        cfg=cfg,
        preprocessed_views=preprocessed,
        face_masks={view: data["face_mask"] for view, data in preprocessed.items()},
        hires_images=hires_images,
        working_image_size=cfg.WORK_IMAGE_SIZE,
    )


def main() -> None:
    from src import config as cfg
    from src.geometry.expression_depth import ExpressionDepthThresholds

    args = _parse_args()
    capture_dir = args.capture_dir.resolve()
    source_output = args.source_output.resolve()
    output = args.output.resolve()
    source_mesh_dir = source_output / "meshes"
    output_mesh_dir = output / "meshes"
    output_texture_dir = output / "textures"
    output.mkdir(parents=True, exist_ok=True)

    thresholds = ExpressionDepthThresholds(
        max_mouth_forward_mean_mm=args.mouth_mean_mm,
        max_mouth_forward_p95_mm=args.mouth_p95_mm,
        max_nose_abs_mean_mm=args.nose_mean_mm,
        max_chin_abs_mean_mm=args.chin_mean_mm,
    )
    geometry_report = _export_depth_safe_geometry(
        source_mesh_dir=source_mesh_dir,
        output_mesh_dir=output_mesh_dir,
        thresholds=thresholds,
        cfg=cfg,
    )
    for name in ("cameras.json", "stable_semantic_regions.json"):
        _copy_if_exists(source_mesh_dir / name, output_mesh_dir / name)

    baseline_texture = source_output / "textures" / "albedo_white.png"
    if not baseline_texture.exists():
        raise FileNotFoundError(f"baseline texture not found: {baseline_texture}")
    output_texture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_texture, output_texture_dir / "albedo_baseline_locked.png")
    same_texture_glb = output_mesh_dir / "face_same_texture.glb"
    _export_with_baseline_texture(
        mesh_path=output_mesh_dir / "face_mesh.obj",
        texture_path=baseline_texture,
        output_path=same_texture_glb,
    )

    rebake_report = None
    if args.rebake:
        rebake_report = _rebake_texture(
            capture_dir=capture_dir,
            output=output,
            cfg=cfg,
        )

    template = args.viewer_template.resolve() if args.viewer_template else _default_viewer_template()
    viewer_path = _write_embedded_compare_viewer(
        template,
        source_mesh_dir / "face.glb",
        same_texture_glb,
        output / "expression_depth_compare.html",
    )
    rebaked_viewer_path = None
    if args.rebake:
        rebaked_viewer_path = _write_embedded_compare_viewer(
            template,
            source_mesh_dir / "face.glb",
            output_mesh_dir / "face.glb",
            output / "expression_depth_rebaked_compare.html",
        )
    report = {
        "capture_dir": str(capture_dir),
        "source_output": str(source_output),
        "geometry": geometry_report,
        "same_texture_glb": str(same_texture_glb),
        "rebake": _jsonable(rebake_report),
        "viewer": str(viewer_path),
        "rebaked_viewer": str(rebaked_viewer_path) if rebaked_viewer_path else None,
    }
    report_path = output / "expression_depth_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    original = geometry_report["constraint"]["original"]["regions"]["mouth"]
    selected = geometry_report["constraint"]["selected"]["regions"]["mouth"]
    logger.info(
        "Mouth expression depth: mean %.3f -> %.3f mm, P95 %.3f -> %.3f mm",
        original["forward_mean_mm"],
        selected["forward_mean_mm"],
        original["forward_p95_mm"],
        selected["forward_p95_mm"],
    )
    logger.info("A/B viewer: %s", viewer_path)
    if rebaked_viewer_path is not None:
        logger.info("Rebaked A/B viewer: %s", rebaked_viewer_path)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
