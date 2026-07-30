"""Run the unified three-view nasal shape experiment from a frozen baseline."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import html
import io
import json
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from run_expression_depth_experiment import (
    _export_with_baseline_texture,
)
from run_nasal_observation_audit import (
    assert_file_tree_unchanged,
    file_tree_hashes,
    load_baseline_fit_parameters,
    run_nasal_observation_audit as _run_nasal_observation_audit,
    validate_separate_output,
)
from src.geometry.nasal_observations import NASAL_VIEWS, NasalObservationBundle
from src.geometry.observation_coordinates import (
    ObservationCoordinates,
    normalize_image_to_work,
)
from src.geometry.observable_flame_subspace import ProjectionView
from src.reports.nasal_geometry_report import (
    render_nasal_geometry_screenshots,
    validate_minimal_nasal_candidate,
    write_nasal_evidence_overlays,
    write_nasal_geometry_report,
)
from src.reports.nasal_observation_io import load_nasal_observation_bundle
from src.reports.nasal_observation_report import read_image_file


ROOT = Path(__file__).resolve().parent


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
    optimization_strategy: str = "unified_v3"
    staged_optimization: Any = None


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
        optimization_strategy="unified_v3",
    )


def _compute_balanced_candidate(
    source_output: Path,
    observations: NasalObservationBundle,
) -> ComputedCandidate:
    from src.geometry.balanced_nasal_optimizer import (
        fit_balanced_multiview_nasal_shape,
    )
    from src.geometry.multiview_nasal_objective import (
        MultiviewNasalObjectiveConfig,
        evaluate_multiview_nasal_objective,
        prepare_multiview_nasal_objective_context,
        prepare_nasal_projection_context,
    )
    from src.geometry.multiview_nasal_optimizer import (
        NasalOptimizationConfig,
    )
    from src.geometry.nasal_semantic_basis import build_nasal_semantic_basis

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
    shape_mode_count = int(baseline.shape_basis.shape[2])
    semantic_only = SimpleNamespace(
        vertex_basis=np.zeros(
            (len(baseline.vertices), 3, 0),
            dtype=np.float64,
        ),
        coefficient_basis=np.zeros(
            (shape_mode_count, 0),
            dtype=np.float64,
        ),
        retained_rank=0,
        report_data={
            "status": "disabled_for_balanced_semantic_v4",
            "retained_rank": 0,
            "reason": (
                "Stage A optimizes only the eight semantic nasal modes"
            ),
        },
    )
    projection_context = prepare_nasal_projection_context(
        baseline.faces,
        semantic_basis,
        observations,
        views,
    )
    objective_context = prepare_multiview_nasal_objective_context(
        baseline.vertices,
        semantic_only,
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
    staged = fit_balanced_multiview_nasal_shape(
        objective_context,
        objective_config,
        optimization_config,
        initial_coefficients=zero,
    )
    return ComputedCandidate(
        baseline=baseline,
        views=views,
        semantic_basis=semantic_basis,
        observable_subspace=semantic_only,
        objective_context=objective_context,
        objective_config=objective_config,
        optimization_config=optimization_config,
        baseline_objective=baseline_objective,
        optimization_result=staged.final_result,
        optimization_strategy="balanced_semantic_v4",
        staged_optimization=staged,
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
    report = {
        "optimization_strategy": computed.optimization_strategy,
        "parameterization": {
            "ordering": list(context.parameter_ordering),
            "coefficients": result.coefficients.tolist(),
            "observable_flame_rank": int(context.observable_rank),
            "semantic_parameter_count": int(
                context.parameter_count - context.observable_rank
            ),
            "representation_note": (
                "Candidate vertices are the frozen-expression baseline plus "
                "the enabled low-dimensional displacement basis. "
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
    if computed.staged_optimization is not None:
        report["balanced_stages"] = _jsonable(
            computed.staged_optimization.to_report_data()
        )
    return report


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
    from PIL import Image

    if not path.is_file() or path.stat().st_size < 20:
        raise RuntimeError(f"candidate textured GLB is missing or empty: {path}")
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise RuntimeError(f"candidate output is not a binary GLB: {path}")
    _magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if version != 2 or total_length != len(data):
        raise RuntimeError(f"candidate GLB header is invalid: {path}")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError("candidate GLB has a truncated chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        if chunk_length % 4:
            raise RuntimeError("candidate GLB chunk length is not 4-byte aligned")
        offset += 8
        if offset + chunk_length > len(data):
            raise RuntimeError("candidate GLB chunk exceeds the declared file bounds")
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        chunks.append((chunk_type, chunk))
    if offset != len(data):
        raise RuntimeError("candidate GLB chunk table does not consume the file")
    json_chunks = [chunk for kind, chunk in chunks if kind == 0x4E4F534A]
    bin_chunks = [chunk for kind, chunk in chunks if kind == 0x004E4942]
    if len(json_chunks) != 1 or not chunks or chunks[0][0] != 0x4E4F534A:
        raise RuntimeError("candidate GLB must contain one leading JSON chunk")
    if len(bin_chunks) != 1:
        raise RuntimeError("candidate GLB must contain exactly one BIN chunk")
    try:
        json_payload = json.loads(
            json_chunks[0].rstrip(b" \0").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("candidate GLB JSON chunk is invalid") from exc
    if not isinstance(json_payload, Mapping):
        raise RuntimeError("candidate GLB JSON root must be an object")
    asset = json_payload.get("asset")
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise RuntimeError("candidate GLB JSON asset.version must be 2.0")

    binary = bin_chunks[0]
    buffers = json_payload.get("buffers")
    if (
        not isinstance(buffers, list)
        or len(buffers) != 1
        or not isinstance(buffers[0], Mapping)
        or "uri" in buffers[0]
    ):
        raise RuntimeError("candidate GLB must use one embedded binary buffer")
    declared_buffer_length = buffers[0].get("byteLength")
    if (
        not isinstance(declared_buffer_length, int)
        or isinstance(declared_buffer_length, bool)
        or declared_buffer_length < 1
        or declared_buffer_length > len(binary)
        or len(binary) - declared_buffer_length > 3
    ):
        raise RuntimeError("candidate GLB BIN length is inconsistent with buffers[0]")

    buffer_views = json_payload.get("bufferViews")
    if not isinstance(buffer_views, list) or not buffer_views:
        raise RuntimeError("candidate GLB has no bufferViews")
    validated_views: list[tuple[int, int]] = []
    for index, view in enumerate(buffer_views):
        if not isinstance(view, Mapping):
            raise RuntimeError(f"candidate GLB bufferViews[{index}] is invalid")
        buffer_index = view.get("buffer")
        byte_offset = view.get("byteOffset", 0)
        byte_length = view.get("byteLength")
        if (
            buffer_index != 0
            or not isinstance(byte_offset, int)
            or isinstance(byte_offset, bool)
            or byte_offset < 0
            or not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 1
            or byte_offset + byte_length > declared_buffer_length
        ):
            raise RuntimeError(
                f"candidate GLB bufferViews[{index}] exceeds BIN bounds"
            )
        validated_views.append((byte_offset, byte_length))

    images = json_payload.get("images")
    textures = json_payload.get("textures")
    materials = json_payload.get("materials")
    if not all(isinstance(value, list) for value in (images, textures, materials)):
        raise RuntimeError("candidate GLB image/texture/material tables are invalid")
    decoded_images: set[int] = set()
    textured_material_count = 0
    textured_material_indices: set[int] = set()
    for material_index, material in enumerate(materials):
        if not isinstance(material, Mapping):
            raise RuntimeError(
                f"candidate GLB materials[{material_index}] is invalid"
            )
        pbr = material.get("pbrMetallicRoughness")
        if not isinstance(pbr, Mapping) or "baseColorTexture" not in pbr:
            continue
        texture_info = pbr["baseColorTexture"]
        texture_index = (
            texture_info.get("index")
            if isinstance(texture_info, Mapping)
            else None
        )
        if (
            not isinstance(texture_index, int)
            or isinstance(texture_index, bool)
            or texture_index < 0
            or texture_index >= len(textures)
            or not isinstance(textures[texture_index], Mapping)
        ):
            raise RuntimeError(
                f"candidate GLB material {material_index} has an invalid texture index"
            )
        image_index = textures[texture_index].get("source")
        if (
            not isinstance(image_index, int)
            or isinstance(image_index, bool)
            or image_index < 0
            or image_index >= len(images)
            or not isinstance(images[image_index], Mapping)
        ):
            raise RuntimeError(
                f"candidate GLB texture {texture_index} has an invalid image index"
            )
        image = images[image_index]
        view_index = image.get("bufferView")
        mime_type = image.get("mimeType")
        if (
            "uri" in image
            or not isinstance(view_index, int)
            or isinstance(view_index, bool)
            or view_index < 0
            or view_index >= len(validated_views)
            or not isinstance(mime_type, str)
            or mime_type not in {"image/png", "image/jpeg"}
        ):
            raise RuntimeError(
                f"candidate GLB image {image_index} is not an embedded PNG/JPEG"
            )
        byte_offset, byte_length = validated_views[view_index]
        image_bytes = binary[byte_offset : byte_offset + byte_length]
        try:
            with Image.open(io.BytesIO(image_bytes)) as embedded:
                embedded.verify()
                decoded_format = str(embedded.format).upper()
        except Exception as exc:
            raise RuntimeError(
                f"candidate GLB image {image_index} cannot be decoded"
            ) from exc
        expected_format = "PNG" if mime_type == "image/png" else "JPEG"
        if decoded_format != expected_format:
            raise RuntimeError(
                f"candidate GLB image {image_index} MIME type disagrees with data"
            )
        decoded_images.add(image_index)
        textured_material_count += 1
        textured_material_indices.add(material_index)
    if not decoded_images or not textured_material_count:
        raise RuntimeError(
            "candidate GLB must contain embedded images and textured materials"
        )

    accessors = json_payload.get("accessors")
    meshes = json_payload.get("meshes")
    if not isinstance(accessors, list) or not accessors:
        raise RuntimeError("candidate GLB has no accessors")
    if not isinstance(meshes, list) or not meshes:
        raise RuntimeError("candidate GLB has no meshes")
    component_sizes = {
        5120: 1,
        5121: 1,
        5122: 2,
        5123: 2,
        5125: 4,
        5126: 4,
    }
    component_formats = {
        5120: "b",
        5121: "B",
        5122: "h",
        5123: "H",
        5125: "I",
        5126: "f",
    }
    type_components = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
    }

    def accessor_contract(
        accessor_index: Any,
        label: str,
        *,
        allowed_component_types: set[int],
        expected_type: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(accessor_index, int)
            or isinstance(accessor_index, bool)
            or accessor_index < 0
            or accessor_index >= len(accessors)
            or not isinstance(accessors[accessor_index], Mapping)
        ):
            raise RuntimeError(f"candidate GLB {label} accessor index is invalid")
        accessor = accessors[accessor_index]
        if "sparse" in accessor:
            raise RuntimeError(f"candidate GLB {label} sparse accessor is unsupported")
        view_index = accessor.get("bufferView")
        component_type = accessor.get("componentType")
        accessor_type = accessor.get("type")
        count = accessor.get("count")
        accessor_offset = accessor.get("byteOffset", 0)
        if (
            not isinstance(view_index, int)
            or isinstance(view_index, bool)
            or view_index < 0
            or view_index >= len(buffer_views)
            or not isinstance(component_type, int)
            or isinstance(component_type, bool)
            or component_type not in allowed_component_types
            or component_type not in component_sizes
            or accessor_type != expected_type
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
            or not isinstance(accessor_offset, int)
            or isinstance(accessor_offset, bool)
            or accessor_offset < 0
        ):
            raise RuntimeError(f"candidate GLB {label} accessor contract is invalid")
        component_size = component_sizes[component_type]
        component_count = type_components[accessor_type]
        element_size = component_size * component_count
        view = buffer_views[view_index]
        stride = view.get("byteStride", element_size)
        if (
            not isinstance(stride, int)
            or isinstance(stride, bool)
            or stride < element_size
            or stride > 252
            or stride % component_size
            or ("byteStride" in view and stride % 4)
            or accessor_offset % component_size
            or (
                validated_views[view_index][0] + accessor_offset
            ) % component_size
        ):
            raise RuntimeError(f"candidate GLB {label} accessor stride is invalid")
        required_length = accessor_offset + (count - 1) * stride + element_size
        if required_length > validated_views[view_index][1]:
            raise RuntimeError(f"candidate GLB {label} accessor exceeds bufferView")
        return {
            "accessor_index": accessor_index,
            "view_index": view_index,
            "component_type": component_type,
            "component_count": component_count,
            "component_size": component_size,
            "count": count,
            "stride": stride,
            "accessor_offset": accessor_offset,
        }

    def accessor_values(contract: Mapping[str, Any]) -> Any:
        view_offset = validated_views[int(contract["view_index"])][0]
        component_type = int(contract["component_type"])
        component_size = int(contract["component_size"])
        component_count = int(contract["component_count"])
        count = int(contract["count"])
        stride = int(contract["stride"])
        accessor_offset = int(contract["accessor_offset"])
        format_string = "<" + component_formats[component_type]
        for row in range(count):
            row_offset = view_offset + accessor_offset + row * stride
            yield tuple(
                struct.unpack_from(
                    format_string,
                    binary,
                    row_offset + column * component_size,
                )[0]
                for column in range(component_count)
            )

    valid_triangle_primitives = 0
    for mesh_index, mesh in enumerate(meshes):
        if not isinstance(mesh, Mapping):
            raise RuntimeError(f"candidate GLB meshes[{mesh_index}] is invalid")
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            raise RuntimeError(
                f"candidate GLB meshes[{mesh_index}] has no primitives"
            )
        for primitive_index, primitive in enumerate(primitives):
            label = f"mesh {mesh_index} primitive {primitive_index}"
            if not isinstance(primitive, Mapping):
                raise RuntimeError(f"candidate GLB {label} is invalid")
            if primitive.get("mode", 4) != 4:
                continue
            attributes = primitive.get("attributes")
            if not isinstance(attributes, Mapping):
                raise RuntimeError(f"candidate GLB {label} attributes are invalid")
            if "POSITION" not in attributes or "TEXCOORD_0" not in attributes:
                raise RuntimeError(
                    f"candidate GLB {label} requires POSITION and TEXCOORD_0"
                )
            if "indices" not in primitive:
                raise RuntimeError(f"candidate GLB {label} requires indices")
            material_index = primitive.get("material")
            if (
                not isinstance(material_index, int)
                or isinstance(material_index, bool)
                or material_index < 0
                or material_index >= len(materials)
                or material_index not in textured_material_indices
            ):
                raise RuntimeError(
                    f"candidate GLB {label} material reference is invalid"
                )
            position = accessor_contract(
                attributes["POSITION"],
                f"{label} POSITION",
                allowed_component_types={5126},
                expected_type="VEC3",
            )
            texcoord = accessor_contract(
                attributes["TEXCOORD_0"],
                f"{label} TEXCOORD_0",
                allowed_component_types={5121, 5123, 5126},
                expected_type="VEC2",
            )
            indices = accessor_contract(
                primitive["indices"],
                f"{label} indices",
                allowed_component_types={5121, 5123, 5125},
                expected_type="SCALAR",
            )
            if (
                position["count"] < 3
                or texcoord["count"] != position["count"]
                or indices["count"] < 3
                or indices["count"] % 3
            ):
                raise RuntimeError(
                    f"candidate GLB {label} accessor counts are invalid"
                )
            if texcoord["component_type"] in {5121, 5123} and (
                accessors[int(texcoord["accessor_index"])].get("normalized")
                is not True
            ):
                raise RuntimeError(
                    f"candidate GLB {label} integer TEXCOORD_0 must be normalized"
                )
            for attribute_name, contract in (
                ("POSITION", position),
                ("TEXCOORD_0", texcoord),
            ):
                if any(
                    not math.isfinite(float(value))
                    for row in accessor_values(contract)
                    for value in row
                ):
                    raise RuntimeError(
                        f"candidate GLB {label} {attribute_name} is non-finite"
                    )
            max_index = max(
                int(row[0])
                for row in accessor_values(indices)
            )
            if max_index >= int(position["count"]):
                raise RuntimeError(
                    f"candidate GLB {label} index exceeds POSITION count"
                )
            valid_triangle_primitives += 1
    if valid_triangle_primitives < 1:
        raise RuntimeError(
            "candidate GLB must contain at least one valid TRIANGLES primitive"
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "byte_count": path.stat().st_size,
        "embedded_image_count": len(decoded_images),
        "textured_material_count": textured_material_count,
        "buffer_view_count": len(validated_views),
        "declared_binary_byte_count": declared_buffer_length,
        "valid_triangle_primitive_count": valid_triangle_primitives,
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


def _projection_points_by_view(objective: Any) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {view: [] for view in NASAL_VIEWS}
    projection = getattr(objective, "projection", None)
    for term in getattr(projection, "per_term", ()):
        semantic_view = str(getattr(term, "semantic_view", ""))
        if semantic_view not in grouped:
            raise ValueError(f"objective projection has unknown view: {semantic_view}")
        points = np.asarray(getattr(term, "pixel_xy", None), dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError(
                f"objective projection for {semantic_view} must be finite (N, 2)"
            )
        grouped[semantic_view].append(points)
    missing = [view for view, parts in grouped.items() if not parts]
    if missing:
        raise ValueError(
            "objective projection is missing semantic views: " + ", ".join(missing)
        )
    return {
        view: np.concatenate(parts, axis=0)
        for view, parts in grouped.items()
    }


def _load_observation_work_images(
    metadata_path: Path,
    observations: NasalObservationBundle,
    rig_calibration: Path,
) -> dict[str, np.ndarray]:
    import cv2

    from src import config as cfg
    from src.module0_intrinsics import undistort_images_with_calibration

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    capture_images = payload["metadata"]["paths"]["capture_images"]
    raw_images = {}
    for camera_view in ("left", "front", "right"):
        image_bgr = read_image_file(
            capture_images[camera_view],
            cv2.IMREAD_COLOR,
            description=f"{camera_view} capture image",
        )
        raw_images[camera_view] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    undistorted, new_intrinsics = undistort_images_with_calibration(
        raw_images,
        calibration_path=rig_calibration,
        alpha=float(cfg.UNDISTORT_ALPHA),
    )
    if new_intrinsics is None:
        raise RuntimeError("rig calibration did not produce undistorted intrinsics")
    camera_view_by_semantic = {
        "front": "front",
        "subject-left": "left",
        "subject-right": "right",
    }
    result = {}
    for semantic_view, camera_view in camera_view_by_semantic.items():
        observation = observations.by_view[semantic_view]
        expected_K = np.asarray(
            observation.coordinate_metadata["intrinsics"],
            dtype=np.float64,
        )
        actual_K = np.asarray(new_intrinsics[camera_view], dtype=np.float64)
        if not np.allclose(actual_K, expected_K, atol=1e-6, rtol=0.0):
            raise RuntimeError(
                f"{semantic_view} undistorted intrinsics disagree with observations"
            )
        result[semantic_view] = normalize_image_to_work(
            undistorted[camera_view],
            _coordinates_for_observation(observation),
        )
    return result


def _write_lightweight_compare_viewer(
    *,
    baseline_glb: Path,
    candidate_glb: Path,
    output: Path,
    dataset_label: str,
) -> Path:
    encoded = {
        "baseline": base64.b64encode(baseline_glb.read_bytes()).decode("ascii"),
        "candidate": base64.b64encode(candidate_glb.read_bytes()).decode("ascii"),
    }
    title = html.escape(
        f"{dataset_label} | Baseline vs unified nasal shape",
        quote=True,
    )
    dataset = html.escape(dataset_label, quote=True)
    payload = json.dumps(encoded, separators=(",", ":")).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; min-height: 100%; background: #10151b; color: #eef2f5; font: 14px Arial, sans-serif; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #34404b; }}
    h1 {{ margin: 0 0 10px; font-size: 19px; }}
    .controls {{ display: flex; gap: 7px; }}
    button {{ color: #eef2f5; background: #26313b; border: 1px solid #4b5b68; padding: 7px 11px; cursor: pointer; }}
    button:focus {{ outline: 2px solid #74b9e6; outline-offset: 1px; }}
    main {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); min-height: calc(100vh - 92px); }}
    section {{ min-width: 0; border-right: 1px solid #34404b; }}
    section:last-child {{ border-right: 0; }}
    h2 {{ box-sizing: border-box; height: 43px; margin: 0; padding: 12px 14px; font-size: 14px; }}
    .viewer-frame {{ position: relative; }}
    canvas {{ display: block; width: 100%; height: calc(100vh - 135px); min-height: 360px; touch-action: none; }}
    .viewer-status {{
      position: absolute; inset: 0; display: grid; place-items: center;
      box-sizing: border-box; padding: 20px; pointer-events: none;
      color: #d9e1e7; background: #0f141c; text-align: center;
    }}
    .viewer-status.error {{ color: #ffb0b0; }}
    .viewer-status[hidden] {{ display: none; }}
    @media (max-width: 760px) {{
      main {{ grid-template-columns: 1fr; }}
      canvas {{ height: 52vh; min-height: 300px; }}
      section {{ border-right: 0; border-bottom: 1px solid #34404b; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{dataset}: unified multiview nasal shape A/B</h1>
  <div class="controls">
    <button type="button" data-view="front">Front</button>
    <button type="button" data-view="left">Subject-left</button>
    <button type="button" data-view="right">Subject-right</button>
  </div>
</header>
<main>
  <section>
    <h2>{dataset} | Baseline: protected expression depth v3</h2>
    <div class="viewer-frame">
      <canvas id="baseline"></canvas>
      <div class="viewer-status" id="baseline-status" role="status" aria-live="polite">Loading embedded model...</div>
    </div>
  </section>
  <section>
    <h2>{dataset} | New: unified multiview nasal shape</h2>
    <div class="viewer-frame">
      <canvas id="candidate"></canvas>
      <div class="viewer-status" id="candidate-status" role="status" aria-live="polite">Loading embedded model...</div>
    </div>
  </section>
</main>
<script>
const embeddedGlbs = {payload};
const state = {{yaw: 0, pitch: 0, zoom: 1}};
const renderers = [];
window.viewerReady = false;
window.viewerError = null;

function setCanvasStatus(id, message, isError = false) {{
  const status = document.getElementById(`${{id}}-status`);
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("error", isError);
  status.dataset.error = isError ? "true" : "false";
}}
function hideCanvasStatus(id) {{
  const status = document.getElementById(`${{id}}-status`);
  status.hidden = true;
  status.classList.remove("error");
  status.dataset.error = "false";
}}
function errorDetail(error) {{
  return error instanceof Error && error.message
    ? error.message
    : String(error);
}}

function decodeBase64(value) {{
  const raw = atob(value), bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}}
function parseGlb(bytes) {{
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2) throw new Error("Invalid GLB");
  let offset = 12, jsonChunk = null, binChunk = null;
  while (offset < bytes.byteLength) {{
    const length = view.getUint32(offset, true), type = view.getUint32(offset + 4, true);
    offset += 8;
    const chunk = bytes.slice(offset, offset + length);
    offset += length;
    if (type === 0x4e4f534a) jsonChunk = chunk;
    if (type === 0x004e4942) binChunk = chunk;
  }}
  if (!jsonChunk || !binChunk) throw new Error("GLB JSON/BIN chunks are required");
  const gltf = JSON.parse(new TextDecoder().decode(jsonChunk).replace(/[\\u0000 ]+$/, ""));
  return {{gltf, bin: binChunk}};
}}
const componentReaders = {{
  5120: ["getInt8", 1], 5121: ["getUint8", 1], 5122: ["getInt16", 2],
  5123: ["getUint16", 2], 5125: ["getUint32", 4], 5126: ["getFloat32", 4]
}};
const componentCounts = {{SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16}};
function accessorData(model, index) {{
  const accessor = model.gltf.accessors[index], bufferView = model.gltf.bufferViews[accessor.bufferView];
  const [reader, bytes] = componentReaders[accessor.componentType];
  const count = componentCounts[accessor.type], stride = bufferView.byteStride || count * bytes;
  const start = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
  const dataView = new DataView(model.bin.buffer, model.bin.byteOffset, model.bin.byteLength);
  const values = new Float32Array(accessor.count * count);
  for (let row = 0; row < accessor.count; row++) {{
    for (let column = 0; column < count; column++) {{
      values[row * count + column] = dataView[reader](start + row * stride + column * bytes, true);
    }}
  }}
  return {{values, count: accessor.count, components: count}};
}}
async function textureFor(model, materialIndex, gl) {{
  const material = model.gltf.materials?.[materialIndex], info = material?.pbrMetallicRoughness?.baseColorTexture;
  if (!info) return null;
  const texture = model.gltf.textures[info.index], image = model.gltf.images[texture.source];
  const bufferView = model.gltf.bufferViews[image.bufferView], start = bufferView.byteOffset || 0;
  const blob = new Blob([model.bin.slice(start, start + bufferView.byteLength)], {{type: image.mimeType}});
  const bitmap = await createImageBitmap(blob);
  const handle = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, handle);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
  gl.generateMipmap(gl.TEXTURE_2D);
  bitmap.close();
  return handle;
}}
async function materialFor(model, materialIndex, gl) {{
  const material = model.gltf.materials?.[materialIndex] || {{}};
  const pbr = material.pbrMetallicRoughness || {{}};
  const alphaMode = ["OPAQUE", "MASK", "BLEND"].includes(material.alphaMode) ? material.alphaMode : "OPAQUE";
  const alphaCutoff = Number.isFinite(material.alphaCutoff) ? material.alphaCutoff : 0.5;
  const factor = Array.isArray(pbr.baseColorFactor) && pbr.baseColorFactor.length === 4 ? pbr.baseColorFactor : [1,1,1,1];
  return {{
    texture: await textureFor(model, materialIndex, gl),
    alphaMode,
    alphaCutoff,
    baseColorFactor: factor
  }};
}}
function shader(gl, type, source) {{
  const value = gl.createShader(type); gl.shaderSource(value, source); gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
  return value;
}}
async function makeRenderer(canvas, encoded) {{
  const gl = canvas.getContext("webgl2", {{antialias: true, alpha: false}});
  if (!gl) throw new Error("WebGL2 is unavailable");
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl, gl.VERTEX_SHADER, `#version 300 es
    in vec3 aPosition; in vec2 aUv; out vec2 vUv;
    uniform vec3 uCenter; uniform float uScale, uAspect, uYaw, uPitch;
    void main() {{
      vec3 p = aPosition - uCenter;
      float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
      p = vec3(cy*p.x+sy*p.z, p.y, -sy*p.x+cy*p.z);
      p = vec3(p.x, cp*p.y-sp*p.z, sp*p.y+cp*p.z);
      gl_Position = vec4(p.x/(uScale*uAspect), p.y/uScale, -p.z/uScale, 1.0);
      vUv = aUv;
    }}`));
  gl.attachShader(program, shader(gl, gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float; in vec2 vUv; out vec4 color;
    uniform sampler2D uTexture; uniform bool uHasTexture;
    uniform int uAlphaMode; uniform float uAlphaCutoff;
    uniform vec4 uBaseColorFactor;
    void main() {{
      vec4 sampled = uHasTexture ? texture(uTexture, vUv) : vec4(1.0);
      vec4 shaded = sampled * uBaseColorFactor;
      if (uAlphaMode == 1 && shaded.a <= max(uAlphaCutoff, 0.001)) discard;
      if (uAlphaMode == 2 && shaded.a <= 0.001) discard;
      color = uAlphaMode == 0 ? vec4(shaded.rgb, 1.0) : shaded;
    }}
  `));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const model = parseGlb(decodeBase64(encoded)), drawables = [], bounds = [[Infinity,Infinity,Infinity],[-Infinity,-Infinity,-Infinity]];
  for (const mesh of model.gltf.meshes || []) for (const primitive of mesh.primitives || []) {{
    if (primitive.mode !== undefined && primitive.mode !== 4) continue;
    const positions = accessorData(model, primitive.attributes.POSITION), uv = primitive.attributes.TEXCOORD_0 === undefined ? null : accessorData(model, primitive.attributes.TEXCOORD_0);
    for (let i=0; i<positions.values.length; i+=3) for (let axis=0; axis<3; axis++) {{
      bounds[0][axis] = Math.min(bounds[0][axis], positions.values[i+axis]);
      bounds[1][axis] = Math.max(bounds[1][axis], positions.values[i+axis]);
    }}
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const positionBuffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer); gl.bufferData(gl.ARRAY_BUFFER, positions.values, gl.STATIC_DRAW);
    const positionLocation = gl.getAttribLocation(program, "aPosition"); gl.enableVertexAttribArray(positionLocation); gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
    const uvLocation = gl.getAttribLocation(program, "aUv");
    if (uv) {{ const uvBuffer=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uvBuffer); gl.bufferData(gl.ARRAY_BUFFER, uv.values, gl.STATIC_DRAW); gl.enableVertexAttribArray(uvLocation); gl.vertexAttribPointer(uvLocation,2,gl.FLOAT,false,0,0); }}
    else {{ gl.disableVertexAttribArray(uvLocation); gl.vertexAttrib2f(uvLocation,0,0); }}
    let count=positions.count, indexed=false;
    if (primitive.indices !== undefined) {{ const source=accessorData(model, primitive.indices).values, indices=new Uint32Array(source); const indexBuffer=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW); count=indices.length; indexed=true; }}
    drawables.push({{vao, count, indexed, material: await materialFor(model, primitive.material, gl)}});
  }}
  const center=bounds[0].map((value,index)=>(value+bounds[1][index])/2), extent=bounds[0].map((value,index)=>bounds[1][index]-value), scale=Math.max(...extent)*0.62 || 1;
  return {{canvas, gl, program, drawables, center, scale}};
}}
async function initializeRenderer(id, encoded) {{
  try {{
    return await makeRenderer(document.getElementById(id), encoded);
  }} catch (error) {{
    const detail = `${{id}}: ${{errorDetail(error)}}`;
    setCanvasStatus(id, `Error: ${{detail}}`, true);
    throw new Error(detail);
  }}
}}
function draw() {{
  for (const renderer of renderers) {{
    const {{canvas, gl, program}}=renderer, width=Math.max(1,Math.floor(canvas.clientWidth*devicePixelRatio)), height=Math.max(1,Math.floor(canvas.clientHeight*devicePixelRatio));
    if (canvas.width!==width || canvas.height!==height) {{canvas.width=width;canvas.height=height;}}
    gl.viewport(0,0,width,height); gl.clearColor(.06,.08,.11,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST); gl.useProgram(program);
    gl.uniform3fv(gl.getUniformLocation(program,"uCenter"), renderer.sharedCenter || renderer.center);
    gl.uniform1f(gl.getUniformLocation(program,"uScale"), (renderer.sharedScale || renderer.scale)*state.zoom);
    gl.uniform1f(gl.getUniformLocation(program,"uAspect"), width/height);
    gl.uniform1f(gl.getUniformLocation(program,"uYaw"), state.yaw); gl.uniform1f(gl.getUniformLocation(program,"uPitch"), state.pitch);
    for (const alphaMode of ["OPAQUE","MASK","BLEND"]) {{
      const blended = alphaMode === "BLEND";
      blended ? gl.enable(gl.BLEND) : gl.disable(gl.BLEND);
      if (blended) gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(!blended);
      for (const item of renderer.drawables) {{
        if (item.material.alphaMode !== alphaMode) continue;
        gl.bindVertexArray(item.vao);
        gl.uniform1i(gl.getUniformLocation(program,"uHasTexture"),!!item.material.texture);
        gl.uniform1i(gl.getUniformLocation(program,"uAlphaMode"),{{OPAQUE:0,MASK:1,BLEND:2}}[alphaMode]);
        gl.uniform1f(gl.getUniformLocation(program,"uAlphaCutoff"),item.material.alphaCutoff);
        gl.uniform4fv(gl.getUniformLocation(program,"uBaseColorFactor"),item.material.baseColorFactor);
        if(item.material.texture)gl.bindTexture(gl.TEXTURE_2D,item.material.texture);
        item.indexed?gl.drawElements(gl.TRIANGLES,item.count,gl.UNSIGNED_INT,0):gl.drawArrays(gl.TRIANGLES,0,item.count);
      }}
    }}
    gl.depthMask(true); gl.disable(gl.BLEND);
  }}
  requestAnimationFrame(draw);
}}
async function start() {{
  const [baseline, candidate] = await Promise.all([
    initializeRenderer("baseline", embeddedGlbs.baseline),
    initializeRenderer("candidate", embeddedGlbs.candidate)
  ]);
  candidate.sharedCenter=baseline.center; candidate.sharedScale=baseline.scale; renderers.push(baseline,candidate);
  for (const canvas of document.querySelectorAll("canvas")) {{
    let dragging=false,lastX=0,lastY=0; canvas.addEventListener("pointerdown",event=>{{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId);}});
    canvas.addEventListener("pointermove",event=>{{if(!dragging)return;state.yaw+=(event.clientX-lastX)*.01;state.pitch=Math.max(-1.2,Math.min(1.2,state.pitch+(event.clientY-lastY)*.01));lastX=event.clientX;lastY=event.clientY;}});
    canvas.addEventListener("pointerup",()=>dragging=false); canvas.addEventListener("wheel",event=>{{event.preventDefault();state.zoom=Math.max(.55,Math.min(2.2,state.zoom*Math.exp(event.deltaY*.001)));}},{{passive:false}});
  }}
  document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{{state.pitch=0;state.yaw={{front:0,left:.73,right:-.73}}[button.dataset.view];}}));
  draw();
  hideCanvasStatus("baseline");
  hideCanvasStatus("candidate");
  window.viewerError = null;
  window.viewerReady = true;
}}
start().catch(error=>{{
  const message = errorDetail(error);
  window.viewerReady = false;
  window.viewerError = message;
  for (const id of ["baseline", "candidate"]) {{
    const status = document.getElementById(`${{id}}-status`);
    if (status.dataset.error !== "true") {{
      setCanvasStatus(id, `Error: ${{message}}`, true);
    }}
  }}
}});
</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def _write_threejs_compare_viewer(
    *,
    baseline_glb: Path,
    candidate_glb: Path,
    output: Path,
    dataset_label: str,
) -> Path:
    template = (
        ROOT
        / "frontend"
        / "templates"
        / "nasal_ab_compare_viewer.html"
    )
    if not template.is_file():
        raise FileNotFoundError(f"Three.js viewer template not found: {template}")
    replacements = {
        "__TITLE__": html.escape(
            f"{dataset_label} | Baseline vs unified nasal shape",
            quote=True,
        ),
        "__BASELINE_LABEL__": html.escape(
            f"{dataset_label} | Baseline: protected expression depth v3",
            quote=True,
        ),
        "__CANDIDATE_LABEL__": html.escape(
            f"{dataset_label} | New: unified multiview nasal shape",
            quote=True,
        ),
        "__BASELINE_BASE64__": base64.b64encode(
            baseline_glb.read_bytes()
        ).decode("ascii"),
        "__CANDIDATE_BASE64__": base64.b64encode(
            candidate_glb.read_bytes()
        ).decode("ascii"),
        "__THREE_MODULE__": (
            ROOT / "frontend" / "vendor" / "three.module.js"
        ).resolve().as_uri(),
        "__ORBIT_CONTROLS__": (
            ROOT
            / "frontend"
            / "vendor"
            / "three"
            / "controls"
            / "OrbitControls.js"
        ).resolve().as_uri(),
        "__GLTF_LOADER__": (
            ROOT
            / "frontend"
            / "vendor"
            / "three"
            / "loaders"
            / "GLTFLoader.js"
        ).resolve().as_uri(),
    }
    page = template.read_text(encoding="utf-8")
    for token, value in replacements.items():
        if token not in page:
            raise RuntimeError(f"Three.js viewer template is missing {token}")
        page = page.replace(token, value)
    leftovers = [
        token
        for token in replacements
        if token in page
    ]
    if leftovers:
        raise RuntimeError(
            "Three.js viewer template contains unresolved placeholders: "
            + ", ".join(leftovers)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


def _write_viewer(
    *,
    template: Path | None,
    baseline_glb: Path,
    candidate_glb: Path,
    output: Path,
    dataset_label: str,
) -> Path:
    if template is None:
        viewer = _write_threejs_compare_viewer(
            baseline_glb=baseline_glb,
            candidate_glb=candidate_glb,
            output=output,
            dataset_label=dataset_label,
        )
    else:
        from run_expression_depth_experiment import _write_embedded_compare_viewer

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
        'data-view="front"',
        'data-view="left"',
        'data-view="right"',
        dataset_label,
    )
    if template is None:
        required += (
            "embeddedModels",
            "baseline: '",
            "candidate: '",
            'role="status"',
            "window.viewerReady",
            "window.viewerError",
            "Promise.all",
        )
    else:
        required += ("baseline: '", "candidate: '")
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(
            "offline viewer is missing embedded/synchronized controls: "
            + ", ".join(missing)
        )
    return viewer


def _ensure_no_candidate_collision(output: Path) -> None:
    collisions = (
        output / "meshes",
        output / "textures",
        output / "nasal_shape_compare.html",
        output / "debug" / "nasal_geometry",
    )
    existing = [str(path) for path in collisions if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing candidate artifacts: "
            + ", ".join(existing)
        )


def _create_staging(output: Path) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=".nasal-shape-staging-",
            dir=output,
        )
    )


def _published_path(path: Path, staging: Path, output: Path) -> Path:
    return output / path.resolve().relative_to(staging.resolve())


def _remap_staging_paths(value: Any, staging: Path, output: Path) -> Any:
    if isinstance(value, Path):
        try:
            return _published_path(value, staging, output)
        except ValueError:
            return value
    if isinstance(value, str):
        try:
            candidate = Path(value)
            return str(_published_path(candidate, staging, output))
        except (OSError, ValueError):
            return value
    if isinstance(value, Mapping):
        return {
            str(key): _remap_staging_paths(item, staging, output)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _remap_staging_paths(item, staging, output)
            for item in value
        ]
    return value


def _publish_staging(
    staging: Path,
    output: Path,
    *,
    source: Path,
    source_hashes: Mapping[str, str],
) -> None:
    candidate_publications = (
        (staging / "meshes", output / "meshes"),
        (staging / "textures", output / "textures"),
        (
            staging / "nasal_shape_compare.html",
            output / "nasal_shape_compare.html",
        ),
        (
            staging / "debug" / "nasal_geometry",
            output / "debug" / "nasal_geometry",
        ),
    )
    staged_report = staging / "nasal_fit_report.json"
    published_report = output / "nasal_fit_report.json"
    report_backup = staging / ".nasal-fit-report.backup.json"
    moved: list[tuple[Path, Path]] = []
    report_was_backed_up = False
    report_was_replaced = False
    try:
        assert_file_tree_unchanged(source, source_hashes)
        for staged_source, _destination in candidate_publications:
            if not staged_source.exists():
                raise FileNotFoundError(
                    f"staged artifact is missing: {staged_source}"
                )
        if not staged_report.is_file():
            raise FileNotFoundError(
                f"staged artifact is missing: {staged_report}"
            )
        for _staged_source, destination in candidate_publications:
            if destination.exists():
                raise FileExistsError(
                    f"refusing to overwrite candidate artifact: {destination}"
                )
        if published_report.exists():
            if not published_report.is_file():
                raise FileExistsError(
                    "existing nasal fit report is not a regular file: "
                    f"{published_report}"
                )
            shutil.copyfile(published_report, report_backup)
            report_was_backed_up = True
        for staged_source, destination in candidate_publications:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_source.replace(destination)
            moved.append((staged_source, destination))
        os.replace(staged_report, published_report)
        report_was_replaced = True
        assert_file_tree_unchanged(source, source_hashes)
        if report_was_backed_up:
            report_backup.unlink()
    except Exception:
        if report_was_replaced:
            if report_was_backed_up and report_backup.exists():
                os.replace(report_backup, published_report)
            elif published_report.exists() and not staged_report.exists():
                os.replace(published_report, staged_report)
        for rollback_source, destination in reversed(moved):
            if destination.exists() and not rollback_source.exists():
                rollback_source.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(rollback_source)
        if report_backup.exists():
            report_backup.unlink()
        raise


def run_multiview_nasal_shape_experiment(
    capture_dir: str | Path,
    source_output: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    viewer_template: str | Path | None = None,
    optimization_strategy: str = "unified_v3",
    expected_baseline_glb_sha256: str | None = None,
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
    strategy = str(optimization_strategy)
    if strategy not in ("unified_v3", "balanced_semantic_v4"):
        raise ValueError(
            "optimization_strategy must be 'unified_v3' or "
            "'balanced_semantic_v4'"
        )
    baseline_glb = _baseline_textured_glb(source)
    baseline_glb_sha256 = _sha256_file(baseline_glb)
    if expected_baseline_glb_sha256 is not None:
        expected_hash = str(expected_baseline_glb_sha256).lower()
        if (
            len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(
                "expected_baseline_glb_sha256 must be 64 lowercase hex digits"
            )
        if baseline_glb_sha256.lower() != expected_hash:
            raise RuntimeError(
                "baseline GLB hash mismatch: expected "
                f"{expected_hash}, got {baseline_glb_sha256}"
            )
    validate_separate_output(target, source, captures)
    _ensure_no_candidate_collision(target)
    source_hashes = file_tree_hashes(source)
    report_path = target / "nasal_fit_report.json"
    dataset_label = captures.name
    report: dict[str, Any] = {
        "schema": "multiview-nasal-shape-experiment-v1",
        "status": "initializing",
        "dataset": dataset_label,
        "optimization_strategy": strategy,
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
        "canonical_baseline_glb": {
            "path": str(baseline_glb),
            "sha256": baseline_glb_sha256,
            "expected_sha256": expected_baseline_glb_sha256,
            "verified": (
                expected_baseline_glb_sha256 is None
                or baseline_glb_sha256.lower()
                == str(expected_baseline_glb_sha256).lower()
            ),
        },
        "rig": {
            "sha256": _sha256_file(rig),
        },
        "profile_shape_v2_reference": _profile_shape_reference(source),
    }
    target.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    publish_completed = False
    publish_started_with_existing_report = False
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
        computed = (
            _compute_balanced_candidate(source, observations)
            if strategy == "balanced_semantic_v4"
            else _compute_candidate(source, observations)
        )
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
        staging = _create_staging(target)
        artifacts = _export_candidate(
            source_output=source,
            output=staging,
            baseline=baseline,
            candidate_vertices=candidate_vertices,
        )
        template = (
            Path(viewer_template).resolve()
            if viewer_template is not None
            else None
        )
        viewer = _write_viewer(
            template=template,
            baseline_glb=baseline_glb,
            candidate_glb=Path(artifacts["candidate_textured_glb"]),
            output=staging / "nasal_shape_compare.html",
            dataset_label=dataset_label,
        )

        report["status"] = "rendering"
        screenshots = render_nasal_geometry_screenshots(
            baseline_glb=baseline_glb,
            candidate_glb=artifacts["candidate_textured_glb"],
            output_dir=staging / "debug" / "nasal_geometry",
        )
        work_images = _load_observation_work_images(
            target / "nasal_observations.json",
            observations,
            rig,
        )
        evidence_overlays = write_nasal_evidence_overlays(
            staging / "debug" / "nasal_geometry",
            work_images_by_view=work_images,
            observation_curves_by_view={
                view: observations.by_view[view].boundaries_work
                for view in NASAL_VIEWS
            },
            baseline_projection_by_view=_projection_points_by_view(
                computed.baseline_objective
            ),
            candidate_projection_by_view=_projection_points_by_view(
                result.final_objective
            ),
        )
        final_objective = _objective_summary(result.final_objective)
        geometry_report = write_nasal_geometry_report(
            staging / "debug" / "nasal_geometry",
            dataset_label=dataset_label,
            screenshots=screenshots,
            evidence_overlays=evidence_overlays,
            baseline_objective=_objective_summary(
                computed.baseline_objective
            )
            or {},
            candidate_objective=final_objective or {},
            evidence=final_objective or {},
            validity=validity,
        )
        success_report = _jsonable(report)
        success_report["paths"].update(
            {
                "candidate_same_texture_glb": str(
                    _published_path(
                        Path(artifacts["candidate_textured_glb"]),
                        staging,
                        target,
                    )
                ),
                "baseline_same_texture_glb": str(baseline_glb),
                "viewer": str(_published_path(viewer, staging, target)),
                "geometry_report": str(
                    _published_path(geometry_report, staging, target)
                ),
                "screenshots": _jsonable(
                    _remap_staging_paths(screenshots, staging, target)
                ),
                "evidence_overlays": _jsonable(
                    _remap_staging_paths(
                        evidence_overlays,
                        staging,
                        target,
                    )
                ),
            }
        )
        success_report["artifacts"] = _jsonable(
            _remap_staging_paths(artifacts, staging, target)
        )
        success_report["status"] = "success"
        _write_report(staging / "nasal_fit_report.json", success_report)
        publish_started_with_existing_report = report_path.is_file()
        _publish_staging(
            staging,
            target,
            source=source,
            source_hashes=source_hashes,
        )
        publish_completed = True
    except Exception as exc:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
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
        if not publish_started_with_existing_report:
            _write_report(report_path, report)
        raise
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if not publish_completed:
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
