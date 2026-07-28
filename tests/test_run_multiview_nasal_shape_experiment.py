from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import run_multiview_nasal_shape_experiment as runner
from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observation_coordinates import ObservationCoordinates


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )


def _camera(
    name: str,
    view: str,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Camera:
    return Camera(
        name=name,
        view=view,
        image_size=(120, 80),
        K=np.array(
            [[180.0, 0.0, 60.0], [0.0, 176.0, 40.0], [0.0, 0.0, 1.0]]
        ),
        dist=np.zeros(5),
        R_rig_to_camera=rotation,
        t_rig_to_camera=translation,
    )


def _observation(semantic_view: str, camera: Camera) -> NasalViewObservation:
    work_size = (60, 40)
    boundary_names = (
        ("subject-left-alar", "subject-right-alar")
        if semantic_view == "front"
        else ("nasal-profile",)
    )
    curves = {
        name: np.array([[20.0, 15.0], [24.0, 20.0], [25.0, 26.0]])
        for name in boundary_names
    }
    raster = np.zeros((40, 60), dtype=np.uint8)
    raster[15:27, 24] = 1
    distance = np.ones((40, 60), dtype=np.float64)
    coordinates = ObservationCoordinates.from_camera(
        camera,
        work_size=work_size,
        pixel_frame="undistorted",
    )
    camera_metadata = {
        "camera_name": camera.name,
        "camera_view": camera.view,
        "subject_relative_view": semantic_view,
        "image_size_wh": list(camera.image_size),
        "intrinsics": camera.K.tolist(),
        "distortion_coefficients": camera.dist.tolist(),
        "rig_to_camera_rotation": camera.R_rig_to_camera.tolist(),
        "rig_to_camera_translation": camera.t_rig_to_camera.tolist(),
    }
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=camera.image_size,
        mask_canvas_shape=(80, 120),
        work_size=work_size,
        roi_work_xyxy=(10.0, 8.0, 50.0, 34.0),
        boundaries_work=curves,
        boundary=raster,
        distance_fields={name: distance for name in boundary_names},
        distance_field=distance,
        confidence=np.ones((40, 60), dtype=np.float64),
        variant_boundaries_work={"base": curves},
        variant_boundaries={"base": raster},
        anchors_work={"tip": np.array([24.0, 20.0])},
        camera_metadata=camera_metadata,
        coordinate_metadata=coordinates.metadata(),
    )


def _bundle() -> NasalObservationBundle:
    front_rotation = _rotation_y(7.0)
    return NasalObservationBundle(
        front=_observation(
            "front",
            _camera(
                "camera2",
                "front",
                front_rotation,
                np.array([0.01, -0.02, 0.03]),
            ),
        ),
        subject_left=_observation(
            "subject-left",
            _camera(
                "camera1",
                "left",
                _rotation_y(-28.0),
                np.array([-0.18, 0.01, 0.06]),
            ),
        ),
        subject_right=_observation(
            "subject-right",
            _camera(
                "camera3",
                "right",
                _rotation_y(31.0),
                np.array([0.19, -0.01, 0.05]),
            ),
        ),
    )


def test_model_projection_views_preserve_front_fit_and_rig_chain() -> None:
    observations = _bundle()
    fit_rotation = _rotation_y(4.0)
    fit_translation = np.array([0.02, -0.03, 0.74])

    views = runner.build_model_projection_views(
        observations,
        fit_rotation,
        fit_translation,
    )

    assert tuple(view.name for view in views) == (
        "front",
        "subject-left",
        "subject-right",
    )
    assert np.allclose(views[0].R_model_to_camera, fit_rotation)
    assert np.allclose(views[0].t_model_to_camera, fit_translation)
    assert np.allclose(
        views[0].K,
        np.array(
            [[90.0, 0.0, 30.0], [0.0, 88.0, 20.0], [0.0, 0.0, 1.0]]
        ),
    )

    points = np.array(
        [[0.01, 0.02, 0.0], [-0.03, 0.04, 0.01], [0.05, -0.02, -0.01]]
    )
    front_camera = observations.front.camera
    points_front = points @ fit_rotation.T + fit_translation
    points_rig = (
        points_front - front_camera.t_rig_to_camera
    ) @ front_camera.R_rig_to_camera
    for view, semantic_view in zip(views, observations.by_view):
        camera = observations.by_view[semantic_view].camera
        expected_camera = (
            points_rig @ camera.R_rig_to_camera.T
            + camera.t_rig_to_camera
        )
        direct_camera = (
            points @ view.R_model_to_camera.T
            + view.t_model_to_camera
        )
        assert np.allclose(direct_camera, expected_camera)
        expected_pixels = expected_camera @ view.K.T
        expected_pixels = expected_pixels[:, :2] / expected_pixels[:, 2:3]
        direct_pixels = direct_camera @ view.K.T
        direct_pixels = direct_pixels[:, :2] / direct_pixels[:, 2:3]
        assert np.allclose(direct_pixels, expected_pixels)


def _objective(vertices: np.ndarray, faces: np.ndarray) -> SimpleNamespace:
    candidate = SimpleNamespace(vertices=vertices, faces=faces)
    projection = _Projection(
        per_term=tuple(
            _ProjectionTerm(
                semantic_view=view,
                pixel_xy=np.array([[20.0, 15.0], [24.0, 20.0]]),
            )
            for view in ("front", "subject-left", "subject-right")
        )
    )
    return SimpleNamespace(
        total_raw_cost=1.0,
        total_robust_cost=0.8,
        raw_costs={"front_alar_left": 1.0},
        robust_costs={"front_alar_left": 0.8},
        effective_observation_counts={"front_alar_left": 4},
        sample_counts={"front_alar_left": 4},
        effective_confidence_sums={"front_alar_left": 3.5},
        per_view_effective_observation_counts={
            "front": 4,
            "subject-left": 3,
            "subject-right": 3,
        },
        per_view_sample_counts={
            "front": 4,
            "subject-left": 3,
            "subject-right": 3,
        },
        per_view_effective_confidence_sums={
            "front": 3.5,
            "subject-left": 2.5,
            "subject-right": 2.5,
        },
        symmetry_evidence_factors={"alar": 0.2},
        robust_loss="soft_l1",
        robust_f_scale=1.0,
        report_data={"geometry_only": True},
        projection=projection,
        candidate=candidate,
    )


@dataclass(frozen=True)
class _ProjectionTerm:
    semantic_view: str
    pixel_xy: np.ndarray


@dataclass(frozen=True)
class _Projection:
    per_term: tuple[_ProjectionTerm, ...]


def _computed(success: bool) -> runner.ComputedCandidate:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    uv = vertices[:, :2].copy()
    baseline = runner.BaselineState(
        shape_parameters=np.zeros(2, dtype=np.float32),
        expression_parameters=np.zeros(2, dtype=np.float32),
        front_rotation=np.eye(3),
        front_translation=np.array([0.0, 0.0, 1.0]),
        vertices=vertices,
        neutral_vertices=vertices.copy(),
        faces=faces,
        shape_basis=np.zeros((3, 3, 2)),
        landmark_triangles=np.tile(faces, (68, 1)),
        landmark_barycentric=np.tile(
            np.array([[1.0, 0.0, 0.0]]),
            (68, 1),
        ),
        uv_vertices=uv,
        uv_faces=faces.copy(),
    )
    baseline_objective = _objective(vertices.copy(), faces)
    final_objective = (
        _objective(vertices + np.array([0.0, 0.0, 0.01]), faces)
        if success
        else None
    )
    result = SimpleNamespace(
        success=success,
        final_objective=final_objective,
        coefficients=np.array([0.1, -0.05]),
        failure_reason=None if success else "solver_failed",
        solver_status=1 if success else -1,
        solver_message="ok" if success else "synthetic failure",
        nfev=3,
        njev=2,
        objective_evaluation_count=4,
        optimality=0.01,
        jacobian_rank=2,
        active_mask=np.zeros(2, dtype=np.int64),
        iteration_trace=(),
        objective_term_trace={},
        report={"selected_coefficients_source": "solver"},
    )
    context = SimpleNamespace(
        parameter_ordering=("observable_flame_0", "tip_depth"),
        observable_rank=1,
        parameter_count=2,
    )
    semantic_basis = SimpleNamespace(
        region_masks={"nose_tip": np.array([False, False, True])},
        support_mask=np.array([False, False, True]),
        protected_mask=np.array([True, False, False]),
    )
    observable = SimpleNamespace(report_data={"status": "ok"})
    views = (
        SimpleNamespace(
            name="front",
            K=np.eye(3),
            R_model_to_camera=np.eye(3),
            t_model_to_camera=np.array([0.0, 0.0, 1.0]),
        ),
    )
    return runner.ComputedCandidate(
        baseline=baseline,
        views=views,
        semantic_basis=semantic_basis,
        observable_subspace=observable,
        objective_context=context,
        objective_config={"same_for_all_datasets": True},
        optimization_config={"max_nfev": 200},
        baseline_objective=baseline_objective,
        optimization_result=result,
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    captures = tmp_path / "captures_fixture"
    source = tmp_path / "source"
    output = tmp_path / "output"
    rig = tmp_path / "rig.json"
    captures.mkdir()
    (source / "meshes").mkdir(parents=True)
    (source / "textures").mkdir()
    (source / "meshes" / "stable_fit_meta.json").write_text(
        '{"baseline": true}',
        encoding="utf-8",
    )
    (source / "meshes" / "face_same_texture.glb").write_bytes(b"baseline-glb")
    (source / "textures" / "albedo_baseline_locked.png").write_bytes(
        b"baseline-texture"
    )
    rig.write_text('{"rig": true}', encoding="utf-8")
    return captures, source, output, rig


def _patch_lightweight_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    computed: runner.ComputedCandidate,
) -> None:
    observations = _bundle()

    def audit(_captures, _source, output, **_kwargs):
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "nasal_observations.json").write_text(
            '{"fixture": true}',
            encoding="utf-8",
        )
        (Path(output) / "nasal_observation_fields.npz").write_bytes(b"fixture")
        return Path(output) / "debug" / "nasal_observations" / "index.html"

    monkeypatch.setattr(runner, "_run_nasal_observation_audit", audit)
    monkeypatch.setattr(
        runner,
        "load_nasal_observation_bundle",
        lambda _path: observations,
    )
    monkeypatch.setattr(
        runner,
        "_compute_candidate",
        lambda _source, _observations: computed,
    )
    monkeypatch.setattr(
        runner,
        "_load_observation_work_images",
        lambda *_args: {
            view: np.full((40, 60, 3), 32, dtype=np.uint8)
            for view in ("front", "subject-left", "subject-right")
        },
    )


def _fake_export_candidate(**kwargs):
    target = Path(kwargs["output"])
    meshes = target / "meshes"
    textures = target / "textures"
    meshes.mkdir(parents=True)
    textures.mkdir(parents=True)
    obj = meshes / "face_mesh.obj"
    raw = meshes / "face_mesh.glb"
    textured = meshes / "face_same_texture.glb"
    texture = textures / "albedo_baseline_locked.png"
    obj.write_text("fixture", encoding="ascii")
    raw.write_bytes(b"raw-glb")
    textured.write_bytes(b"textured-glb")
    texture.write_bytes(b"texture")
    return {
        "candidate_obj": obj,
        "candidate_geometry_glb": raw,
        "candidate_textured_glb": textured,
        "baseline_locked_texture": texture,
        "baseline_locked_texture_sha256": "fixture",
        "texture_matches_source": True,
        "subdivided_vertex_count": 3,
        "subdivided_face_count": 1,
        "glb_validation": {
            "path": str(textured),
            "embedded_image_count": 1,
        },
    }


def _fake_write_viewer(**kwargs):
    target = Path(kwargs["output"])
    target.write_text(
        "embedded baseline candidate front left right",
        encoding="utf-8",
    )
    return target


def _write_textured_glb(
    path: Path,
    *,
    include_bin: bool = True,
    include_mesh: bool = False,
    texture_index: int = 0,
    image_bytes: bytes | None = None,
    buffer_view_length: int | None = None,
    alpha_mode: str = "OPAQUE",
    transparent_texture: bool = False,
    index_values: tuple[int, int, int] = (0, 1, 2),
    position_component_type: int = 5126,
) -> Path:
    if image_bytes is None:
        encoded = io.BytesIO()
        Image.new(
            "RGBA",
            (2, 2),
            (20, 80, 140, 0 if transparent_texture else 255),
        ).save(encoded, format="PNG")
        image_bytes = encoded.getvalue()
    binary = bytearray()
    buffer_views = []

    def add_buffer_view(values: bytes) -> int:
        binary.extend(b"\0" * ((-len(binary)) % 4))
        offset = len(binary)
        binary.extend(values)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(values),
            }
        )
        return len(buffer_views) - 1

    image_view = add_buffer_view(image_bytes)
    if buffer_view_length is not None:
        buffer_views[image_view]["byteLength"] = buffer_view_length
    payload = {
        "asset": {"version": "2.0"},
        "bufferViews": buffer_views,
        "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "name": "hidden_bottom_white" if alpha_mode == "BLEND" else "face",
                "alphaMode": alpha_mode,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": texture_index}
                }
            }
        ],
    }
    if include_mesh:
        position_view = add_buffer_view(
            struct.pack(
                "<9f",
                -0.8,
                -0.8,
                0.0,
                0.8,
                -0.8,
                0.0,
                0.0,
                0.8,
                0.0,
            )
        )
        texcoord_view = add_buffer_view(
            struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.5, 1.0)
        )
        index_view = add_buffer_view(struct.pack("<3H", *index_values))
        payload["accessors"] = [
            {
                "bufferView": position_view,
                "componentType": position_component_type,
                "count": 3,
                "type": "VEC3",
            },
            {
                "bufferView": texcoord_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
            },
            {
                "bufferView": index_view,
                "componentType": 5123,
                "count": 3,
                "type": "SCALAR",
            },
        ]
        payload["meshes"] = [
            {
                "primitives": [
                    {
                        "mode": 4,
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ]
    payload["buffers"] = [{"byteLength": len(binary)}]
    json_chunk = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    chunks = [
        struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    ]
    if include_bin:
        binary.extend(b"\0" * ((-len(binary)) % 4))
        chunks.append(
            struct.pack("<II", len(binary), 0x004E4942) + bytes(binary)
        )
    body = b"".join(chunks)
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def _assert_no_candidate_outputs(output: Path) -> None:
    assert not (output / "meshes").exists()
    assert not (output / "textures").exists()
    assert not (output / "nasal_shape_compare.html").exists()
    assert not (output / "debug" / "nasal_geometry").exists()
    assert not any(
        path.name.startswith(".nasal-shape-staging-")
        for path in output.iterdir()
    )


def test_embedded_textured_glb_validation_follows_texture_image_chain(
    tmp_path: Path,
) -> None:
    path = _write_textured_glb(tmp_path / "valid.glb", include_mesh=True)

    report = runner._validate_embedded_textured_glb(path)

    assert report["embedded_image_count"] == 1
    assert report["textured_material_count"] == 1
    assert report["buffer_view_count"] == 4
    assert report["valid_triangle_primitive_count"] == 1


def test_embedded_textured_glb_validation_rejects_texture_only_pseudo_glb(
    tmp_path: Path,
) -> None:
    path = _write_textured_glb(tmp_path / "no-mesh.glb")

    with pytest.raises(RuntimeError, match="no accessors|no meshes"):
        runner._validate_embedded_textured_glb(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index_values": (0, 1, 9)}, "index exceeds POSITION count"),
        ({"position_component_type": 5123}, "POSITION accessor contract"),
    ],
)
def test_embedded_textured_glb_validation_rejects_invalid_mesh_accessors(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    path = _write_textured_glb(
        tmp_path / "invalid-mesh.glb",
        include_mesh=True,
        **kwargs,
    )

    with pytest.raises(RuntimeError, match=message):
        runner._validate_embedded_textured_glb(path)


def test_embedded_textured_glb_validation_rejects_missing_bin(
    tmp_path: Path,
) -> None:
    path = _write_textured_glb(tmp_path / "missing-bin.glb", include_bin=False)

    with pytest.raises(RuntimeError, match="BIN chunk"):
        runner._validate_embedded_textured_glb(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"texture_index": 3}, "texture index"),
        ({"buffer_view_length": 100000}, "BIN bounds"),
        ({"image_bytes": b"not-a-png"}, "cannot be decoded"),
    ],
)
def test_embedded_textured_glb_validation_rejects_broken_references_and_images(
    tmp_path: Path,
    kwargs: dict,
    message: str,
) -> None:
    path = _write_textured_glb(tmp_path / "invalid.glb", **kwargs)

    with pytest.raises(RuntimeError, match=message):
        runner._validate_embedded_textured_glb(path)


def test_default_viewer_is_self_contained_without_historical_template(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.glb"
    candidate = tmp_path / "candidate.glb"
    baseline.write_bytes(b"baseline-model")
    candidate.write_bytes(b"candidate-model")

    viewer = runner._write_viewer(
        template=None,
        baseline_glb=baseline,
        candidate_glb=candidate,
        output=tmp_path / "nasal_shape_compare.html",
        dataset_label="captures_fixture",
    )

    text = viewer.read_text(encoding="utf-8")
    assert base64.b64encode(b"baseline-model").decode("ascii") in text
    assert base64.b64encode(b"candidate-model").decode("ascii") in text
    assert "embeddedGlbs" in text
    assert "https://" not in text
    assert 'data-view="front"' in text
    assert 'data-view="left"' in text
    assert 'data-view="right"' in text


def test_default_viewer_parses_gltf_alpha_modes_and_configures_blending(
    tmp_path: Path,
) -> None:
    baseline = _write_textured_glb(
        tmp_path / "baseline.glb",
        include_mesh=True,
    )
    candidate = _write_textured_glb(
        tmp_path / "candidate.glb",
        include_mesh=True,
        alpha_mode="BLEND",
        transparent_texture=True,
    )
    assert runner._validate_embedded_textured_glb(baseline)[
        "valid_triangle_primitive_count"
    ] == 1
    assert runner._validate_embedded_textured_glb(candidate)[
        "valid_triangle_primitive_count"
    ] == 1

    viewer = runner._write_viewer(
        template=None,
        baseline_glb=baseline,
        candidate_glb=candidate,
        output=tmp_path / "alpha-viewer.html",
        dataset_label="alpha_fixture",
    )

    text = viewer.read_text(encoding="utf-8")
    assert '["OPAQUE", "MASK", "BLEND"]' in text
    assert "material.alphaMode" in text
    assert "material.alphaCutoff" in text
    assert "uAlphaMode == 1" in text
    assert "uAlphaMode == 2" in text
    assert "discard" in text
    assert "gl.enable(gl.BLEND)" in text
    assert "gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)" in text
    assert "gl.depthMask(!blended)" in text
    assert "gl.depthMask(true)" in text


def test_default_viewer_discards_transparent_blend_pixels_in_headless_chrome(
    tmp_path: Path,
) -> None:
    chrome = next(
        (
            executable
            for executable in (
                shutil.which("chrome"),
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            )
            if executable is not None and Path(executable).is_file()
        ),
        None,
    )
    if chrome is None:
        pytest.skip("Chrome/Chromium is unavailable for the viewer pixel check")
    baseline = _write_textured_glb(
        tmp_path / "baseline.glb",
        include_mesh=True,
    )
    candidate = _write_textured_glb(
        tmp_path / "candidate.glb",
        include_mesh=True,
        alpha_mode="BLEND",
        transparent_texture=True,
    )
    viewer = runner._write_viewer(
        template=None,
        baseline_glb=baseline,
        candidate_glb=candidate,
        output=tmp_path / "alpha-viewer.html",
        dataset_label="alpha_fixture",
    )
    screenshot = tmp_path / "alpha-viewer.png"
    completed = subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--no-sandbox",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            f"--user-data-dir={tmp_path / 'chrome-profile'}",
            "--window-size=1200,800",
            "--virtual-time-budget=4000",
            f"--screenshot={screenshot}",
            viewer.resolve().as_uri(),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert screenshot.is_file() and screenshot.stat().st_size > 0
    rendered = Image.open(screenshot).convert("RGB")
    background = (15, 20, 28)
    assert rendered.getpixel((300, 450)) != background
    assert rendered.getpixel((900, 450)) == background
    candidate_region = rendered.crop((700, 180, 1100, 700))
    assert all(pixel == background for pixel in candidate_region.getdata())


def test_runner_happy_path_writes_outputs_and_preserves_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, source, output, rig = _inputs(tmp_path)
    before = runner.file_tree_hashes(source)
    computed = _computed(success=True)
    _patch_lightweight_pipeline(monkeypatch, computed)

    def render(**kwargs):
        target = Path(kwargs["output_dir"])
        target.mkdir(parents=True, exist_ok=True)
        result = {"baseline": {}, "candidate": {}}
        for role in result:
            for view in ("front", "subject-left", "subject-right"):
                path = target / f"{role}_{view}.png"
                Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
                result[role][view] = path
        return result

    monkeypatch.setattr(runner, "_export_candidate", _fake_export_candidate)
    monkeypatch.setattr(runner, "render_nasal_geometry_screenshots", render)

    report_path = runner.run_multiview_nasal_shape_experiment(
        captures,
        source,
        output,
        rig_calibration=rig,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["fit"]["parameterization"]["ordering"] == [
        "observable_flame_0",
        "tip_depth",
    ]
    assert "not standalone FLAME shape_params" in report["fit"][
        "parameterization"
    ]["representation_note"]
    assert (output / "meshes" / "face_same_texture.glb").is_file()
    assert (output / "nasal_shape_compare.html").is_file()
    assert "embeddedGlbs" in (
        output / "nasal_shape_compare.html"
    ).read_text(encoding="utf-8")
    assert (output / "debug" / "nasal_geometry" / "index.html").is_file()
    assert not any(
        path.name.startswith(".nasal-shape-staging-")
        for path in output.iterdir()
    )
    assert runner.file_tree_hashes(source) == before


def test_runner_optimizer_failure_reports_before_any_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, source, output, rig = _inputs(tmp_path)
    before = runner.file_tree_hashes(source)
    _patch_lightweight_pipeline(monkeypatch, _computed(success=False))

    def forbidden_export(**_kwargs):
        raise AssertionError("export must not run after optimizer failure")

    monkeypatch.setattr(runner, "_export_candidate", forbidden_export)

    with pytest.raises(RuntimeError, match="optimization failed"):
        runner.run_multiview_nasal_shape_experiment(
            captures,
            source,
            output,
            rig_calibration=rig,
        )

    report = json.loads(
        (output / "nasal_fit_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed_optimization"
    assert report["fit"]["optimizer"]["failure_reason"] == "solver_failed"
    assert not (output / "meshes" / "face_same_texture.glb").exists()
    assert runner.file_tree_hashes(source) == before


def test_runner_export_failure_cleans_staging_and_candidate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, source, output, rig = _inputs(tmp_path)
    before = runner.file_tree_hashes(source)
    _patch_lightweight_pipeline(monkeypatch, _computed(success=True))

    def failing_export(**kwargs):
        partial = Path(kwargs["output"]) / "meshes" / "face_mesh.obj"
        partial.parent.mkdir(parents=True)
        partial.write_text("partial", encoding="ascii")
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(runner, "_export_candidate", failing_export)

    with pytest.raises(RuntimeError, match="synthetic export failure"):
        runner.run_multiview_nasal_shape_experiment(
            captures,
            source,
            output,
            rig_calibration=rig,
        )

    report = json.loads(
        (output / "nasal_fit_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert (output / "nasal_observations.json").is_file()
    assert "candidate_same_texture_glb" not in report["paths"]
    _assert_no_candidate_outputs(output)
    assert runner.file_tree_hashes(source) == before


def test_publish_failure_rolls_back_all_published_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "output" / ".nasal-shape-staging-fixture"
    output = tmp_path / "output"
    source.mkdir()
    (source / "locked.txt").write_text("locked", encoding="utf-8")
    (staging / "meshes").mkdir(parents=True)
    (staging / "textures").mkdir()
    (staging / "debug" / "nasal_geometry").mkdir(parents=True)
    (staging / "meshes" / "face_same_texture.glb").write_bytes(b"candidate")
    (staging / "textures" / "albedo_baseline_locked.png").write_bytes(b"texture")
    (staging / "nasal_shape_compare.html").write_text("viewer", encoding="utf-8")
    (staging / "debug" / "nasal_geometry" / "index.html").write_text(
        "geometry",
        encoding="utf-8",
    )
    (staging / "nasal_fit_report.json").write_text(
        '{"status":"success"}',
        encoding="utf-8",
    )
    original_replace = Path.replace
    failed = False

    def flaky_replace(self, target):
        nonlocal failed
        destination = Path(target)
        if destination.name == "nasal_shape_compare.html" and not failed:
            failed = True
            raise OSError("synthetic publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    with pytest.raises(OSError, match="synthetic publish failure"):
        runner._publish_staging(
            staging,
            output,
            source=source,
            source_hashes=runner.file_tree_hashes(source),
        )

    assert not (output / "meshes").exists()
    assert not (output / "textures").exists()
    assert not (output / "nasal_shape_compare.html").exists()
    assert not (output / "debug" / "nasal_geometry").exists()
    assert not (output / "nasal_fit_report.json").exists()
    assert (staging / "meshes" / "face_same_texture.glb").is_file()
    assert (staging / "textures" / "albedo_baseline_locked.png").is_file()
    assert (staging / "nasal_shape_compare.html").is_file()
    assert (staging / "debug" / "nasal_geometry" / "index.html").is_file()
    assert (staging / "nasal_fit_report.json").is_file()


def test_post_publish_source_check_rolls_back_success_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    staging = output / ".nasal-shape-staging-fixture"
    source.mkdir()
    (source / "locked.txt").write_text("locked", encoding="utf-8")
    (staging / "meshes").mkdir(parents=True)
    (staging / "textures").mkdir()
    (staging / "debug" / "nasal_geometry").mkdir(parents=True)
    (staging / "meshes" / "face_same_texture.glb").write_bytes(b"candidate")
    (staging / "textures" / "albedo_baseline_locked.png").write_bytes(b"texture")
    (staging / "nasal_shape_compare.html").write_text("viewer", encoding="utf-8")
    (staging / "debug" / "nasal_geometry" / "index.html").write_text(
        "geometry",
        encoding="utf-8",
    )
    (staging / "nasal_fit_report.json").write_text(
        '{"status":"success"}',
        encoding="utf-8",
    )
    expected_hashes = runner.file_tree_hashes(source)
    original_replace = Path.replace

    def replace_and_mutate_source(self, target):
        result = original_replace(self, target)
        if Path(target) == output / "nasal_fit_report.json":
            (source / "locked.txt").write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(Path, "replace", replace_and_mutate_source)

    with pytest.raises(RuntimeError, match="source files changed"):
        runner._publish_staging(
            staging,
            output,
            source=source,
            source_hashes=expected_hashes,
        )

    assert not (output / "meshes").exists()
    assert not (output / "textures").exists()
    assert not (output / "nasal_shape_compare.html").exists()
    assert not (output / "debug" / "nasal_geometry").exists()
    assert not (output / "nasal_fit_report.json").exists()
    assert (staging / "meshes" / "face_same_texture.glb").is_file()
    assert (staging / "textures" / "albedo_baseline_locked.png").is_file()
    assert (staging / "nasal_shape_compare.html").is_file()
    assert (staging / "debug" / "nasal_geometry" / "index.html").is_file()
    assert (staging / "nasal_fit_report.json").is_file()


def test_runner_source_mutation_before_publish_leaves_only_failure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, source, output, rig = _inputs(tmp_path)
    _patch_lightweight_pipeline(monkeypatch, _computed(success=True))
    monkeypatch.setattr(runner, "_export_candidate", _fake_export_candidate)
    monkeypatch.setattr(runner, "_write_viewer", _fake_write_viewer)

    def render_and_mutate_source(**kwargs):
        target = Path(kwargs["output_dir"])
        target.mkdir(parents=True, exist_ok=True)
        result = {"baseline": {}, "candidate": {}}
        for role in result:
            for view in ("front", "subject-left", "subject-right"):
                path = target / f"{role}_{view}.png"
                Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
                result[role][view] = path
        (source / "meshes" / "stable_fit_meta.json").write_text(
            '{"mutated": true}',
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(
        runner,
        "render_nasal_geometry_screenshots",
        render_and_mutate_source,
    )

    with pytest.raises(RuntimeError, match="source files changed"):
        runner.run_multiview_nasal_shape_experiment(
            captures,
            source,
            output,
            rig_calibration=rig,
        )

    report = json.loads(
        (output / "nasal_fit_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed_rendering"
    assert report["error"]["type"] == "RuntimeError"
    _assert_no_candidate_outputs(output)


def test_runner_render_failure_cleans_staging_and_candidate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, source, output, rig = _inputs(tmp_path)
    before = runner.file_tree_hashes(source)
    _patch_lightweight_pipeline(monkeypatch, _computed(success=True))
    monkeypatch.setattr(runner, "_export_candidate", _fake_export_candidate)
    monkeypatch.setattr(runner, "_write_viewer", _fake_write_viewer)

    def failing_render(**kwargs):
        partial = Path(kwargs["output_dir"]) / "baseline_front.png"
        partial.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (20, 30, 40)).save(partial)
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(
        runner,
        "render_nasal_geometry_screenshots",
        failing_render,
    )

    with pytest.raises(RuntimeError, match="synthetic render failure"):
        runner.run_multiview_nasal_shape_experiment(
            captures,
            source,
            output,
            rig_calibration=rig,
        )

    report = json.loads(
        (output / "nasal_fit_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed_rendering"
    assert (output / "nasal_observations.json").is_file()
    assert "candidate_same_texture_glb" not in report["paths"]
    _assert_no_candidate_outputs(output)
    assert runner.file_tree_hashes(source) == before
