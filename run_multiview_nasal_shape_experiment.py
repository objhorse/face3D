"""Run the unified three-view nasal shape experiment from a frozen baseline."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from run_expression_depth_experiment import (
    _default_viewer_template,
    _export_with_baseline_texture,
    _write_embedded_compare_viewer,
)
from run_nasal_observation_audit import (
    assert_file_tree_unchanged,
    file_tree_hashes,
    load_baseline_fit_parameters,
    run_nasal_observation_audit as _run_nasal_observation_audit,
    validate_separate_output,
)
from src.geometry.nasal_observations import NASAL_VIEWS, NasalObservationBundle
from src.geometry.observation_coordinates import ObservationCoordinates
from src.geometry.observable_flame_subspace import ProjectionView
from src.reports.nasal_geometry_report import (
    render_nasal_geometry_screenshots,
    validate_minimal_nasal_candidate,
    write_nasal_geometry_report,
)
from src.reports.nasal_observation_io import load_nasal_observation_bundle


@dataclass(frozen=True)
class BaselineState:
    shape_parameters: np.ndarray
    expression_parameters: np.ndarray
    front_rotation: np.ndarray
    front_translation: np.ndarray
    vertices: np.ndarray
    neutral_vertices: np.ndarray
    faces: np.ndarray
    shape_basis: np.ndarray
    landmark_triangles: np.ndarray
    landmark_barycentric: np.ndarray
    uv_vertices: np.ndarray
    uv_faces: np.ndarray


@dataclass(frozen=True)
class ComputedCandidate:
    baseline: BaselineState
    views: tuple[ProjectionView, ...]
    semantic_basis: Any
    observable_subspace: Any
    objective_context: Any
    objective_config: Any
    optimization_config: Any
    baseline_objective: Any
    optimization_result: Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one topology-preserving multiview nasal candidate without "
            "changing the default pipeline."
        )
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig-calibration", type=Path, required=True)
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_digest(hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _coordinates_for_observation(observation: Any) -> ObservationCoordinates:
    metadata = observation.coordinate_metadata
    source_frame = str(metadata["source_pixel_frame"])
    if source_frame not in {
        "distorted_original_px",
        "undistorted_original_px",
    }:
        raise ValueError(
            f"unsupported observation source pixel frame: {source_frame}"
        )
    pixel_frame = (
        "distorted"
        if source_frame == "distorted_original_px"
        else "undistorted"
    )
    return ObservationCoordinates(
        original_size=tuple(observation.original_size),
        work_size=tuple(observation.work_size),
        K=np.asarray(metadata["intrinsics"], dtype=np.float64),
        dist=np.asarray(metadata["distortion_coefficients"], dtype=np.float64),
        pixel_frame=pixel_frame,
    )


def build_model_projection_views(
    observations: NasalObservationBundle,
    front_fit_rotation: Any,
    front_fit_translation: Any,
) -> tuple[ProjectionView, ...]:
    """Compose model->front fit with fixed rig extrinsics in semantic order."""
    if not isinstance(observations, NasalObservationBundle):
        raise ValueError("observations must be a NasalObservationBundle")
    fit_rotation = np.asarray(front_fit_rotation, dtype=np.float64)
    fit_translation = np.asarray(front_fit_translation, dtype=np.float64)
    if fit_rotation.shape != (3, 3) or fit_translation.shape != (3,):
        raise ValueError("front fit must contain a 3x3 R and 3-vector t")
    if not np.isfinite(fit_rotation).all() or not np.isfinite(
        fit_translation
    ).all():
        raise ValueError("front fit R/t must be finite")
    front_camera = observations.front.camera
    front_rig_rotation = np.asarray(
        front_camera.R_rig_to_camera,
        dtype=np.float64,
    )
    front_rig_translation = np.asarray(
        front_camera.t_rig_to_camera,
        dtype=np.float64,
    )
    views = []
    for semantic_view in NASAL_VIEWS:
        observation = observations.by_view[semantic_view]
        camera = observation.camera
        target_rig_rotation = np.asarray(
            camera.R_rig_to_camera,
            dtype=np.float64,
        )
        target_rig_translation = np.asarray(
            camera.t_rig_to_camera,
            dtype=np.float64,
        )
        target_from_front_rotation = (
            target_rig_rotation @ front_rig_rotation.T
        )
        target_from_front_translation = (
            target_rig_translation
            - target_from_front_rotation @ front_rig_translation
        )
        coordinates = _coordinates_for_observation(observation)
        views.append(
            ProjectionView(
                name=semantic_view,
                K=coordinates.work_intrinsics,
                R_model_to_camera=target_from_front_rotation @ fit_rotation,
                t_model_to_camera=(
                    target_from_front_rotation @ fit_translation
                    + target_from_front_translation
                ),
            )
        )
    return tuple(views)


def _load_baseline_state(source_output: Path) -> BaselineState:
    import torch

    from src import config as cfg
    from src.module2_geometry import (
        FLAMEModel,
        _get_flame_uv,
        load_flame_landmark_mapping,
    )

    metadata_path = source_output / "meshes" / "stable_fit_meta.json"
    shape, expression, rotation, translation = load_baseline_fit_parameters(
        metadata_path
    )
    flame = FLAMEModel(
        Path(cfg.FLAME_MODEL_PATH),
        n_shape=len(shape),
        n_exp=len(expression),
    ).cpu()
    faces = flame.faces.detach().cpu().numpy()
    mapping = load_flame_landmark_mapping(Path(cfg.FLAME_LANDMARK_PATH))
    if mapping is None:
        raise RuntimeError("FLAME landmark embedding is required")
    landmark_triangles = faces[
        np.asarray(mapping["face_idx"], dtype=np.int64)
    ]
    barycentric = np.asarray(mapping["bary_coords"], dtype=np.float64)
    with torch.no_grad():
        shape_tensor = torch.from_numpy(shape)
        vertices = flame(
            shape_tensor,
            torch.from_numpy(expression),
        ).cpu().numpy()
        neutral_vertices = flame(
            shape_tensor,
            torch.zeros(flame.n_exp, dtype=torch.float32),
        ).cpu().numpy()
    shape_basis = (
        flame.shape_basis.detach()
        .cpu()
        .numpy()
        .reshape(flame.n_verts, 3, flame.n_shape)
    )
    uv_vertices, uv_faces = _get_flame_uv(Path(cfg.FLAME_MODEL_PATH), faces)
    return BaselineState(
        shape_parameters=np.asarray(shape, dtype=np.float32),
        expression_parameters=np.asarray(expression, dtype=np.float32),
        front_rotation=np.asarray(rotation, dtype=np.float64),
        front_translation=np.asarray(translation, dtype=np.float64),
        vertices=np.asarray(vertices),
        neutral_vertices=np.asarray(neutral_vertices),
        faces=np.asarray(faces, dtype=np.int64),
        shape_basis=np.asarray(shape_basis),
        landmark_triangles=np.asarray(landmark_triangles, dtype=np.int64),
        landmark_barycentric=barycentric,
        uv_vertices=np.asarray(uv_vertices),
        uv_faces=np.asarray(uv_faces, dtype=np.int64),
    )


def _compute_candidate(
    source_output: Path,
    observations: NasalObservationBundle,
) -> ComputedCandidate:
    from src.geometry.multiview_nasal_objective import (
        MultiviewNasalObjectiveConfig,
        evaluate_multiview_nasal_objective,
        prepare_multiview_nasal_objective_context,
        prepare_nasal_projection_context,
    )
    from src.geometry.multiview_nasal_optimizer import (
        NasalOptimizationConfig,
        fit_multiview_nasal_shape,
    )
    from src.geometry.nasal_semantic_basis import build_nasal_semantic_basis
    from src.geometry.observable_flame_subspace import (
        build_observable_flame_subspace,
    )

    baseline = _load_baseline_state(source_output)
    views = build_model_projection_views(
        observations,
        baseline.front_rotation,
        baseline.front_translation,
    )
    semantic_basis = build_nasal_semantic_basis(
        baseline.vertices,
        baseline.faces,
        baseline.landmark_triangles,
        baseline.landmark_barycentric,
        baseline.front_rotation,
    )
    observable = build_observable_flame_subspace(
        baseline.vertices,
        baseline.shape_basis,
        semantic_basis.support_mask,
        semantic_basis.protected_mask,
        views,
    )
    projection_context = prepare_nasal_projection_context(
        baseline.faces,
        semantic_basis,
        observations,
        views,
    )
    objective_context = prepare_multiview_nasal_objective_context(
        baseline.vertices,
        observable,
        semantic_basis,
        observations,
        projection_context=projection_context,
    )
    objective_config = MultiviewNasalObjectiveConfig()
    optimization_config = NasalOptimizationConfig()
    zero = np.zeros(objective_context.parameter_count, dtype=np.float64)
    baseline_objective = evaluate_multiview_nasal_objective(
        zero,
        objective_context,
        objective_config,
    )
    result = fit_multiview_nasal_shape(
        objective_context,
        objective_config,
        optimization_config,
        initial_coefficients=zero,
    )
    return ComputedCandidate(
        baseline=baseline,
        views=views,
        semantic_basis=semantic_basis,
        observable_subspace=observable,
        objective_context=objective_context,
        objective_config=objective_config,
        optimization_config=optimization_config,
        baseline_objective=baseline_objective,
        optimization_result=result,
    )


def _objective_summary(objective: Any) -> dict[str, Any] | None:
    if objective is None:
        return None
    return {
        "total_raw_cost": float(objective.total_raw_cost),
        "total_robust_cost": float(objective.total_robust_cost),
        "raw_costs": _jsonable(objective.raw_costs),
        "robust_costs": _jsonable(objective.robust_costs),
        "effective_observation_counts": _jsonable(
            objective.effective_observation_counts
        ),
        "sample_counts": _jsonable(objective.sample_counts),
        "effective_confidence_sums": _jsonable(
            objective.effective_confidence_sums
        ),
        "per_view_effective_observation_counts": _jsonable(
            objective.per_view_effective_observation_counts
        ),
        "per_view_sample_counts": _jsonable(
            objective.per_view_sample_counts
        ),
        "per_view_effective_confidence_sums": _jsonable(
            objective.per_view_effective_confidence_sums
        ),
        "symmetry_evidence_factors": _jsonable(
            objective.symmetry_evidence_factors
        ),
        "robust_loss": str(objective.robust_loss),
        "robust_f_scale": float(objective.robust_f_scale),
        "report_data": _jsonable(objective.report_data),
        "projection": _jsonable(objective.projection),
    }


def _profile_shape_reference(source_output: Path) -> dict[str, Any]:
    source_name = source_output.name
    candidate_names = []
    if "_protected_expression_depth_v3" in source_name:
        candidate_names.append(
            source_name.replace(
                "_protected_expression_depth_v3",
                "_profile_shape_v2",
            )
        )
    candidate_names.append(source_name + "_profile_shape_v2")
    checked = []
    for name in dict.fromkeys(candidate_names):
        path = source_output.parent / name / "profile_shape_report.json"
        checked.append(str(path))
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                return {
                    "status": "invalid",
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return {
                "status": "found",
                "path": str(path),
                "sha256": _sha256_file(path),
                "report": payload,
            }
    return {
        "status": "missing",
        "checked_paths": checked,
        "note": "profile_shape_v2 single-scalar reference was not found",
    }


def _semantic_regions(semantic_basis: Any) -> dict[str, Any]:
    regions = {
        str(name): [
            int(index)
            for index in np.flatnonzero(np.asarray(mask, dtype=bool))
        ]
        for name, mask in semantic_basis.region_masks.items()
    }
    regions["support"] = [
        int(index)
        for index in np.flatnonzero(semantic_basis.support_mask)
    ]
    regions["protected"] = [
        int(index)
        for index in np.flatnonzero(semantic_basis.protected_mask)
    ]
    return regions


def _fit_report(computed: ComputedCandidate) -> dict[str, Any]:
    result = computed.optimization_result
    final = result.final_objective
    context = computed.objective_context
    return {
        "parameterization": {
            "ordering": list(context.parameter_ordering),
            "coefficients": result.coefficients.tolist(),
            "observable_flame_rank": int(context.observable_rank),
            "semantic_parameter_count": int(
                context.parameter_count - context.observable_rank
            ),
            "representation_note": (
                "Candidate vertices are the frozen-expression baseline plus "
                "observable FLAME and semantic low-dimensional displacement. "
                "These coefficients are not standalone FLAME shape_params."
            ),
        },
        "baseline": {
            "selected_coefficients": "all zeros",
            "objective": _objective_summary(computed.baseline_objective),
            "shape_parameter_count": int(
                len(computed.baseline.shape_parameters)
            ),
            "expression_parameter_count": int(
                len(computed.baseline.expression_parameters)
            ),
            "shape_parameters_sha256": _array_digest(
                computed.baseline.shape_parameters
            ),
            "frozen_expression_parameters_sha256": _array_digest(
                computed.baseline.expression_parameters
            ),
            "neutral_vertices_sha256": _array_digest(
                computed.baseline.neutral_vertices
            ),
        },
        "candidate": {
            "objective": _objective_summary(final),
            "vertices_source": (
                "optimization_result.final_objective.candidate.vertices"
            ),
            "vertices_sha256": (
                None
                if final is None
                else _array_digest(final.candidate.vertices)
            ),
        },
        "optimizer": {
            "success": bool(result.success),
            "failure_reason": result.failure_reason,
            "solver_status": int(result.solver_status),
            "solver_message": str(result.solver_message),
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "objective_evaluation_count": int(
                result.objective_evaluation_count
            ),
            "optimality": result.optimality,
            "jacobian_rank": result.jacobian_rank,
            "active_mask": result.active_mask.tolist(),
            "iteration_trace": _jsonable(result.iteration_trace),
            "objective_term_trace": _jsonable(
                result.objective_term_trace
            ),
            "report": _jsonable(result.report),
            "config": _jsonable(computed.optimization_config),
        },
        "objective_config": _jsonable(computed.objective_config),
        "observable_subspace": _jsonable(
            computed.observable_subspace.report_data
        ),
        "semantic_regions": _semantic_regions(computed.semantic_basis),
        "projection_views": [
            {
                "semantic_view": view.name,
                "K_work": view.K.tolist(),
                "R_model_to_camera": view.R_model_to_camera.tolist(),
                "t_model_to_camera": view.t_model_to_camera.tolist(),
            }
            for view in computed.views
        ],
    }


def _subdivide_candidate(
    baseline: BaselineState,
    candidate_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    vertices_sub, faces_sub = trimesh.remesh.subdivide_loop(
        candidate_vertices,
        baseline.faces,
        iterations=2,
    )
    uv_sub = np.asarray(baseline.uv_vertices)
    uv_faces_sub = np.asarray(baseline.uv_faces)
    for _iteration in range(2):
        uv_sub, uv_faces_sub = trimesh.remesh.subdivide(
            uv_sub,
            uv_faces_sub,
        )
    if len(faces_sub) != len(uv_faces_sub):
        raise RuntimeError("geometry and UV subdivision topology diverged")
    return vertices_sub, faces_sub, uv_sub, uv_faces_sub


def _validate_embedded_textured_glb(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 20:
        raise RuntimeError(f"candidate textured GLB is missing or empty: {path}")
    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise RuntimeError(f"candidate output is not a binary GLB: {path}")
    _magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if version != 2 or total_length != len(data):
        raise RuntimeError(f"candidate GLB header is invalid: {path}")
    offset = 12
    json_payload = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_payload = json.loads(chunk.rstrip(b" \0").decode("utf-8"))
    if not isinstance(json_payload, Mapping):
        raise RuntimeError("candidate GLB has no JSON chunk")
    images = json_payload.get("images", [])
    materials = json_payload.get("materials", [])
    embedded_images = [
        item
        for item in images
        if isinstance(item, Mapping)
        and "bufferView" in item
        and "uri" not in item
    ]
    textured_materials = [
        item
        for item in materials
        if isinstance(item, Mapping)
        and isinstance(item.get("pbrMetallicRoughness"), Mapping)
        and "baseColorTexture" in item["pbrMetallicRoughness"]
    ]
    if not embedded_images or not textured_materials:
        raise RuntimeError(
            "candidate GLB must contain embedded images and textured materials"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
        "embedded_image_count": len(embedded_images),
        "textured_material_count": len(textured_materials),
    }


def _export_candidate(
    *,
    source_output: Path,
    output: Path,
    baseline: BaselineState,
    candidate_vertices: np.ndarray,
) -> dict[str, Any]:
    from src.module2_geometry import export_mesh_glb, export_mesh_obj

    meshes = output / "meshes"
    textures = output / "textures"
    meshes.mkdir(parents=True, exist_ok=True)
    textures.mkdir(parents=True, exist_ok=True)
    vertices_sub, faces_sub, uv_sub, uv_faces_sub = _subdivide_candidate(
        baseline,
        candidate_vertices,
    )
    obj_path = meshes / "face_mesh.obj"
    raw_glb_path = meshes / "face_mesh.glb"
    export_mesh_obj(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        obj_path,
    )
    export_mesh_glb(
        vertices_sub,
        faces_sub,
        uv_sub,
        uv_faces_sub,
        raw_glb_path,
    )
    texture_source = (
        source_output / "textures" / "albedo_baseline_locked.png"
    )
    if not texture_source.is_file():
        raise FileNotFoundError(
            f"baseline-locked texture is missing: {texture_source}"
        )
    texture_output = textures / "albedo_baseline_locked.png"
    shutil.copy2(texture_source, texture_output)
    source_texture_hash = _sha256_file(texture_source)
    output_texture_hash = _sha256_file(texture_output)
    if output_texture_hash != source_texture_hash:
        raise RuntimeError("copied baseline-locked texture hash does not match source")
    textured_glb = meshes / "face_same_texture.glb"
    _export_with_baseline_texture(
        mesh_path=obj_path,
        texture_path=texture_output,
        output_path=textured_glb,
    )
    glb_validation = _validate_embedded_textured_glb(textured_glb)
    return {
        "candidate_obj": obj_path,
        "candidate_geometry_glb": raw_glb_path,
        "candidate_textured_glb": textured_glb,
        "baseline_locked_texture": texture_output,
        "baseline_locked_texture_sha256": output_texture_hash,
        "texture_matches_source": True,
        "subdivided_vertex_count": int(len(vertices_sub)),
        "subdivided_face_count": int(len(faces_sub)),
        "glb_validation": glb_validation,
    }


def _baseline_textured_glb(source_output: Path) -> Path:
    path = source_output / "meshes" / "face_same_texture.glb"
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(
            "source output must contain meshes/face_same_texture.glb "
            "for an exact same-texture comparison"
        )
    return path


def _write_viewer(
    *,
    template: Path,
    baseline_glb: Path,
    candidate_glb: Path,
    output: Path,
    dataset_label: str,
) -> Path:
    viewer = _write_embedded_compare_viewer(
        template,
        baseline_glb,
        candidate_glb,
        output,
        baseline_label=(
            f"{dataset_label} | Baseline: protected expression depth v3"
        ),
        candidate_label=(
            f"{dataset_label} | New: unified multiview nasal shape"
        ),
        title=f"{dataset_label} | Baseline vs unified nasal shape",
    )
    text = viewer.read_text(encoding="utf-8")
    required = (
        "baseline: '",
        "candidate: '",
        'data-view="front"',
        'data-view="left"',
        'data-view="right"',
        dataset_label,
    )
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(
            "offline viewer is missing embedded/synchronized controls: "
            + ", ".join(missing)
        )
    return viewer


def _ensure_no_candidate_collision(output: Path) -> None:
    collisions = (
        output / "meshes" / "face_same_texture.glb",
        output / "nasal_shape_compare.html",
    )
    existing = [str(path) for path in collisions if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing candidate artifacts: "
            + ", ".join(existing)
        )


def run_multiview_nasal_shape_experiment(
    capture_dir: str | Path,
    source_output: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    viewer_template: str | Path | None = None,
) -> Path:
    """Run Batch D for one dataset and return nasal_fit_report.json."""
    captures = Path(capture_dir).resolve()
    source = Path(source_output).resolve()
    target = Path(output).resolve()
    rig = Path(rig_calibration).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if not source.is_dir():
        raise FileNotFoundError(f"source output does not exist: {source}")
    if not rig.is_file():
        raise FileNotFoundError(f"rig calibration does not exist: {rig}")
    validate_separate_output(target, source, captures)
    _ensure_no_candidate_collision(target)
    source_hashes = file_tree_hashes(source)
    report_path = target / "nasal_fit_report.json"
    dataset_label = captures.name
    report: dict[str, Any] = {
        "schema": "multiview-nasal-shape-experiment-v1",
        "status": "initializing",
        "dataset": dataset_label,
        "default_pipeline_modified": False,
        "texture_scoring_in_objective": False,
        "paths": {
            "capture_dir": str(captures),
            "source_output": str(source),
            "output": str(target),
            "rig_calibration": str(rig),
            "nasal_observations": str(
                target / "nasal_observations.json"
            ),
        },
        "source_read_only": {
            "file_count": len(source_hashes),
            "tree_sha256": _tree_digest(source_hashes),
            "files": dict(source_hashes),
        },
        "rig": {
            "sha256": _sha256_file(rig),
        },
        "profile_shape_v2_reference": _profile_shape_reference(source),
    }
    target.mkdir(parents=True, exist_ok=True)
    try:
        report["status"] = "building_observations"
        _run_nasal_observation_audit(
            captures,
            source,
            target,
            rig_calibration=rig,
        )
        observations = load_nasal_observation_bundle(
            target / "nasal_observations.json"
        )
        report["observation_camera_order"] = [
            observations.camera_name_by_view[view]
            for view in NASAL_VIEWS
        ]

        report["status"] = "optimizing"
        computed = _compute_candidate(source, observations)
        report["fit"] = _fit_report(computed)
        result = computed.optimization_result
        if not result.success or result.final_objective is None:
            report["status"] = "failed_optimization"
            _write_report(report_path, report)
            reason = result.failure_reason or result.solver_message
            raise RuntimeError(f"multiview nasal optimization failed: {reason}")

        candidate_vertices = np.asarray(
            result.final_objective.candidate.vertices
        )
        baseline = computed.baseline
        validity = validate_minimal_nasal_candidate(
            parameters=result.coefficients,
            baseline_vertices=baseline.vertices,
            candidate_vertices=candidate_vertices,
            baseline_faces=baseline.faces,
            candidate_faces=result.final_objective.candidate.faces,
            baseline_uv_vertices=baseline.uv_vertices,
            candidate_uv_vertices=baseline.uv_vertices,
            baseline_uv_faces=baseline.uv_faces,
            candidate_uv_faces=baseline.uv_faces,
        )
        report["minimal_validity"] = validity
        if not validity["passed"]:
            report["status"] = "failed_minimal_validity"
            _write_report(report_path, report)
            raise RuntimeError(
                "multiview nasal candidate failed minimal validity: "
                + ", ".join(validity["issues"])
            )

        report["status"] = "exporting"
        artifacts = _export_candidate(
            source_output=source,
            output=target,
            baseline=baseline,
            candidate_vertices=candidate_vertices,
        )
        baseline_glb = _baseline_textured_glb(source)
        template = (
            Path(viewer_template).resolve()
            if viewer_template is not None
            else _default_viewer_template().resolve()
        )
        viewer = _write_viewer(
            template=template,
            baseline_glb=baseline_glb,
            candidate_glb=Path(artifacts["candidate_textured_glb"]),
            output=target / "nasal_shape_compare.html",
            dataset_label=dataset_label,
        )

        report["status"] = "rendering"
        screenshots = render_nasal_geometry_screenshots(
            baseline_glb=baseline_glb,
            candidate_glb=artifacts["candidate_textured_glb"],
            output_dir=target / "debug" / "nasal_geometry",
        )
        final_objective = _objective_summary(result.final_objective)
        geometry_report = write_nasal_geometry_report(
            target / "debug" / "nasal_geometry",
            dataset_label=dataset_label,
            screenshots=screenshots,
            baseline_objective=_objective_summary(
                computed.baseline_objective
            )
            or {},
            candidate_objective=final_objective or {},
            evidence=final_objective or {},
            validity=validity,
        )
        report["paths"].update(
            {
                "candidate_same_texture_glb": str(
                    artifacts["candidate_textured_glb"]
                ),
                "baseline_same_texture_glb": str(baseline_glb),
                "viewer": str(viewer),
                "geometry_report": str(geometry_report),
                "screenshots": _jsonable(screenshots),
            }
        )
        report["artifacts"] = _jsonable(artifacts)
        report["status"] = "success"
        _write_report(report_path, report)
    except Exception as exc:
        if not str(report.get("status", "")).startswith("failed_"):
            report["status"] = (
                "failed_rendering"
                if report.get("status") == "rendering"
                else "failed"
            )
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_report(report_path, report)
        raise
    finally:
        assert_file_tree_unchanged(source, source_hashes)
    return report_path


def main() -> None:
    args = _parse_args()
    report = run_multiview_nasal_shape_experiment(
        args.capture_dir,
        args.source_output,
        args.output,
        rig_calibration=args.rig_calibration,
        viewer_template=args.viewer_template,
    )
    print(f"Multiview nasal fit report: {report}")


if __name__ == "__main__":
    main()


__all__ = [
    "BaselineState",
    "ComputedCandidate",
    "build_model_projection_views",
    "run_multiview_nasal_shape_experiment",
]
