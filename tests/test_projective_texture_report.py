from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.reports.projective_texture_report as report_module
from src.reports.projective_texture_report import (
    write_projective_texture_truth_report,
)


class RecordingWarp:
    def __init__(self, offset_x: float = 3.0) -> None:
        self.offset_x = float(offset_x)
        self.calls: list[tuple[np.ndarray, tuple[int, int]]] = []

    def apply(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        self.calls.append((np.array(points, copy=True), tuple(image_shape)))
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] += self.offset_x
        return result


class InvalidWarp:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def apply(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        if self.mode == "nan":
            result[0, 0] = np.nan
            return result
        if self.mode == "shape":
            return result[:-1]
        raise AssertionError(f"unexpected mode: {self.mode}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_sha256(image: np.ndarray) -> str:
    array = np.asarray(image)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def force_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report_module, "_create_cuda_visibility_backend", lambda vertices, faces: None
    )


@pytest.fixture
def tiny_scene(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], Path]:
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    # With this K, vertices land exactly on pixel centers (11.5, 11.5),
    # (20.5, 11.5), and (16.5, 20.5) for exact-pixel CPU depth tests.
    (mesh_dir / "face_mesh.obj").write_text(
        "\n".join(
            [
                "v -0.45 -0.45 2.0",
                "v 0.45 -0.45 2.0",
                "v 0.05 0.45 2.0",
                "v 0.0 0.0 -1.0",
                "v 100.0 0.0 2.0",
                "vt 0.0 0.0",
                "vt 1.0 0.0",
                "vt 0.5 1.0",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cameras = {
        "views": {
            "front": {
                "K": [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]],
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [0.0, 0.0, 0.0],
            }
        }
    }
    (mesh_dir / "cameras.json").write_text(
        json.dumps(cameras), encoding="utf-8"
    )
    semantic_path = tmp_path / "semantic_regions.json"
    semantic_path.write_text(
        json.dumps({"regions": {"nose_tip": [2, 999], "empty": [-1, 1000]}}),
        encoding="utf-8",
    )
    image = np.full((32, 32, 3), (28, 42, 61), dtype=np.uint8)
    return mesh_dir, {"front": image}, semantic_path


def test_truth_report_without_warp_records_contract_provenance_and_relative_paths(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, images, semantic_path = tiny_scene
    output_dir = tmp_path / "truth_report"
    mesh_hash_before = _sha256(mesh_dir / "face_mesh.obj")
    cameras_hash_before = _sha256(mesh_dir / "cameras.json")
    semantic_hash_before = _sha256(semantic_path)

    result = write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images=images,
        output_dir=output_dir,
        expected_image_size=(32, 32),
        semantic_regions_path=semantic_path,
    )

    assert result["schema_version"] == 1
    assert result["final_sampling_mode"] == "unwarped_projective"
    assert result["geometry_changed"] is False
    assert result["camera_changed"] is False
    assert result["face_mesh_sha256"] == mesh_hash_before
    assert result["cameras_sha256"] == cameras_hash_before
    assert result["semantic_regions_sha256"] == semantic_hash_before
    assert result["image_sha256"] == {"front": _image_sha256(images["front"])}
    assert _sha256(mesh_dir / "face_mesh.obj") == mesh_hash_before
    assert _sha256(mesh_dir / "cameras.json") == cameras_hash_before

    front = result["views"]["front"]
    assert front["image_size"] == [32, 32]
    assert front["vertex_count"] == 5
    assert front["positive_depth_ratio"] == pytest.approx(0.8)
    assert front["in_image_ratio"] == pytest.approx(0.6)
    assert front["depth_visible_ratio"] == pytest.approx(0.6)
    assert front["visible_vertex_count"] == 3
    assert front["diagnostic_valid"] is True
    assert front["visibility_backend"] == "cpu_fallback"
    assert front["visibility_method"] == "perspective_face_id_vertex_adjacency"
    assert front["camera_canvas_validation"]["valid"] is True
    assert front["camera_canvas_validation"]["expected_image_size"] == [32, 32]
    assert front["camera_canvas_validation"]["actual_image_size"] == [32, 32]
    assert front["legacy_warp_present"] is False
    assert front["legacy_sampling_displacement"]["count"] == 0
    assert front["overlay_path"] == "front_unwarped_overlay.jpg"
    assert not Path(front["overlay_path"]).is_absolute()

    overlay = cv2.imread(str(output_dir / front["overlay_path"]))
    assert overlay is not None
    assert overlay.shape[:2] == (32, 32)
    stored = json.loads((output_dir / "truth_metrics.json").read_text("utf-8"))
    assert stored == result
    json.dumps(stored, allow_nan=False)
    html_text = (output_dir / "index.html").read_text("utf-8")
    assert "front" in html_text
    assert "front_unwarped_overlay.jpg" in html_text
    assert str(output_dir.resolve()) not in html_text


def test_warp_receives_only_raster_visible_in_image_points_and_is_backfilled(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    warp = RecordingWarp(offset_x=3.0)

    result = write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images=images,
        output_dir=tmp_path / "warped_truth_report",
        expected_image_size=32,
        sampling_warps={"front": warp},
    )

    assert result["final_sampling_mode"] == "unwarped_projective"
    front = result["views"]["front"]
    displacement = front["legacy_sampling_displacement"]
    assert front["legacy_warp_present"] is True
    assert displacement["count"] == 3
    assert displacement["mean_displacement_px"] == pytest.approx(3.0)
    assert displacement["p95_displacement_px"] == pytest.approx(3.0)
    assert displacement["max_displacement_px"] == pytest.approx(3.0)
    assert len(warp.calls) == 1
    warped_input, image_shape = warp.calls[0]
    assert warped_input.shape == (3, 2)
    assert image_shape == (32, 32)
    assert np.all((warped_input[:, 0] >= 0) & (warped_input[:, 0] < 32))
    assert np.all((warped_input[:, 1] >= 0) & (warped_input[:, 1] < 32))


def test_truth_report_rejects_view_without_enough_visible_vertices(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    (mesh_dir / "face_mesh.obj").write_text(
        "\n".join(
            [
                "v -0.45 -0.45 -2.0",
                "v 0.45 -0.45 -2.0",
                "v 0.05 0.45 -2.0",
                "vt 0.0 0.0",
                "vt 1.0 0.0",
                "vt 0.5 1.0",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "invalid_truth_report"

    with pytest.raises(RuntimeError, match="not interpretable"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=output_dir,
            expected_image_size=(32, 32),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("mode", ["nan", "shape"])
def test_truth_report_rejects_invalid_warp_output(
    mode: str,
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene

    with pytest.raises(ValueError, match="warp.*(shape|finite)"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=tmp_path / f"bad_warp_{mode}",
            expected_image_size=(32, 32),
            sampling_warps={"front": InvalidWarp(mode)},
        )


@pytest.mark.parametrize("bad_view", ["../escape", "front/../../escape", "camera1"])
def test_truth_report_rejects_unknown_or_traversal_view_names_immediately(
    bad_view: str,
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    output_dir = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="view"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images={bad_view: images["front"]},
            output_dir=output_dir,
            expected_image_size=(32, 32),
        )

    assert not output_dir.exists()


def test_safe_output_path_rejects_escape(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    with pytest.raises(ValueError, match="output path"):
        report_module._safe_output_path(output_dir.resolve(), "../escape.jpg")


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 3), dtype=np.float32),
        np.zeros((32, 32, 4), dtype=np.uint8),
        np.zeros((0, 32, 3), dtype=np.uint8),
    ],
)
def test_truth_report_requires_nonempty_uint8_rgb_images(
    bad_image: np.ndarray,
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
) -> None:
    mesh_dir, _images, _semantic_path = tiny_scene
    with pytest.raises(ValueError, match="uint8 RGB"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images={"front": bad_image},
            output_dir=tmp_path / "bad_image",
            expected_image_size=(32, 32),
        )


def test_truth_report_requires_explicit_expected_image_size(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    with pytest.raises(ValueError, match="expected_image_size"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=tmp_path / "missing_expected_size",
        )


def test_truth_report_rejects_1536_canvas_when_1024_is_required(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, _images, _semantic_path = tiny_scene
    mismatched_image = np.zeros((1536, 1536, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="camera canvas mismatch"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images={"front": mismatched_image},
            output_dir=tmp_path / "bad_canvas",
            expected_image_size=1024,
        )


@pytest.mark.parametrize(
    "mutate_intrinsics",
    [
        lambda K: K.__setitem__((0, 0), 0.0),
        lambda K: K.__setitem__((1, 1), -20.0),
        lambda K: K.__setitem__((2, slice(None)), [0.1, 0.0, 1.0]),
        lambda K: K.__setitem__((0, 1), np.nan),
    ],
)
def test_truth_report_rejects_invalid_camera_intrinsics(
    mutate_intrinsics,
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    cameras_path = mesh_dir / "cameras.json"
    cameras = json.loads(cameras_path.read_text("utf-8"))
    K = np.asarray(cameras["views"]["front"]["K"], dtype=np.float64)
    mutate_intrinsics(K)
    cameras["views"]["front"]["K"] = K.tolist()
    cameras_path.write_text(json.dumps(cameras), "utf-8")

    with pytest.raises(ValueError, match="camera intrinsics"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=tmp_path / "bad_intrinsics",
            expected_image_size=(32, 32),
        )


def test_large_mesh_without_cuda_fails_before_cpu_rasterization(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    vertices = np.array([[-0.45, -0.45, 2], [0.45, -0.45, 2], [0.05, 0.45, 2]], np.float32)
    faces = np.tile(
        np.array([[0, 1, 2]], dtype=np.int32),
        (report_module.CPU_FALLBACK_MAX_FACES + 1, 1),
    )
    monkeypatch.setattr(
        report_module,
        "load_mesh_obj",
        lambda path: (vertices, faces, np.empty((0, 2)), np.empty((0, 3), np.int32)),
    )

    with pytest.raises(RuntimeError, match="CUDA.*large mesh"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=tmp_path / "large_mesh",
            expected_image_size=(32, 32),
        )


def test_visibility_backend_is_created_once_and_reused_across_views(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    cameras_path = mesh_dir / "cameras.json"
    cameras = json.loads(cameras_path.read_text("utf-8"))
    front_camera = cameras["views"]["front"]
    cameras["views"]["left"] = front_camera
    cameras["views"]["right"] = front_camera
    cameras_path.write_text(json.dumps(cameras), "utf-8")
    all_images = {
        "left": images["front"].copy(),
        "front": images["front"].copy(),
        "right": images["front"].copy(),
    }

    class FakeCudaBackend:
        name = "cuda_nvdiffrast"
        method = "nvdiffrast_triangle_id"

        def __init__(self) -> None:
            self.render_calls = 0

        def render(self, K, R, t, image_shape):
            self.render_calls += 1
            return np.ones(image_shape, dtype=np.int32)

    backend = FakeCudaBackend()
    factory_calls = 0

    def create_backend(vertices, faces):
        nonlocal factory_calls
        factory_calls += 1
        return backend

    monkeypatch.setattr(report_module, "_create_cuda_visibility_backend", create_backend)

    result = write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images=all_images,
        output_dir=tmp_path / "three_view_report",
        expected_image_sizes={
            "left": (32, 32),
            "front": (32, 32),
            "right": (32, 32),
        },
    )

    assert factory_calls == 1
    assert backend.render_calls == 3
    assert result["visibility_backend"] == "cuda_nvdiffrast"
    assert result["visibility_method"] == "nvdiffrast_triangle_id"


def test_perspective_cpu_raster_selects_slanted_front_triangle() -> None:
    projected = np.array(
        [
            [4.5, 4.5],
            [11.5, 4.5],
            [8.5, 11.5],
            [4.5, 4.5],
            [11.5, 4.5],
            [8.5, 11.5],
        ],
        dtype=np.float32,
    )
    # At the center, linear z interpolation would put face 2 in front.
    # Perspective-correct 1/z interpolation correctly selects face 1.
    depth = np.array([1.0, 10.0, 10.0, 5.0, 5.0, 5.0], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    triangle_ids = report_module._cpu_perspective_triangle_ids(
        projected, depth, np.ones(6, dtype=bool), faces, (16, 16)
    )

    assert triangle_ids[8, 8] == 1


def test_nvdiffrast_style_triangle_ids_require_a_winning_adjacent_face() -> None:
    triangle_ids = np.zeros((16, 16), dtype=np.int32)
    triangle_ids[4:12, 4:12] = 1
    projected = np.array(
        [[5.5, 5.5], [10.5, 5.5], [8.5, 10.5]] * 2,
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    visible = report_module._visible_vertices_from_triangle_ids(
        triangle_ids,
        projected,
        np.ones(6, dtype=bool),
        faces,
    )

    np.testing.assert_array_equal(visible, [True, True, True, False, False, False])


def test_real_nvdiffrast_overlap_always_selects_near_triangle() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("nvdiffrast.torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real nvdiffrast depth test")

    near = np.array(
        [[-0.225, -0.225, 1.0], [0.225, -0.225, 1.0], [0.025, 0.225, 1.0]],
        dtype=np.float32,
    )
    far = near * 2.0
    behind = near * -1.0
    K = np.array(
        [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)

    for vertices, expected_near_face in (
        (np.vstack([near, far, behind]), 1),
        (np.vstack([far, near, behind]), 2),
    ):
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        backend = report_module._CudaVisibilityBackend(vertices, faces)
        triangle_ids = backend.render(K, R, t, (32, 32))
        assert triangle_ids[16, 16] == expected_near_face
        assert not np.any(triangle_ids == 3)


def test_report_clip_depth_has_ordered_ndc_and_valid_near_far() -> None:
    torch = pytest.importorskip("torch")
    vertices = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, report_module.DEFAULT_POSITIVE_DEPTH_EPSILON * 0.5],
        ],
        dtype=torch.float32,
    )
    K = torch.eye(3, dtype=torch.float32)
    R = torch.eye(3, dtype=torch.float32)
    t = torch.zeros(3, dtype=torch.float32)

    clip, positive, near, far = report_module._camera_vertices_to_perspective_clip(
        vertices, K, R, t, (32, 32)
    )
    ndc_depth = clip[:, 2] / clip[:, 3]

    assert float(near) > 0.0
    assert float(far) > float(near)
    assert float(ndc_depth[0]) < float(ndc_depth[1])
    np.testing.assert_array_equal(positive.numpy(), [True, True, False, False])


def test_overlapped_back_triangle_vertices_are_not_visible_or_sent_to_warp(
    tmp_path: Path,
) -> None:
    mesh_dir = tmp_path / "overlap_mesh"
    mesh_dir.mkdir()
    mesh_lines = [
        "v -0.45 -0.45 2.0",
        "v 0.45 -0.45 2.0",
        "v 0.05 0.45 2.0",
        "v -0.675 -0.675 3.0",
        "v 0.675 -0.675 3.0",
        "v 0.075 0.675 3.0",
    ]
    mesh_lines.extend(["vt 0.0 0.0"] * 6)
    mesh_lines.extend(["f 1/1 2/2 3/3", "f 4/4 5/5 6/6"])
    (mesh_dir / "face_mesh.obj").write_text("\n".join(mesh_lines) + "\n", "utf-8")
    camera = {
        "views": {
            "front": {
                "K": [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]],
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, 0.0],
            }
        }
    }
    (mesh_dir / "cameras.json").write_text(json.dumps(camera), "utf-8")
    warp = RecordingWarp()

    result = write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images={"front": np.zeros((32, 32, 3), dtype=np.uint8)},
        output_dir=tmp_path / "overlap_report",
        expected_image_size=(32, 32),
        sampling_warps={"front": warp},
    )

    assert result["views"]["front"]["depth_visible_ratio"] == pytest.approx(0.5)
    assert result["views"]["front"]["legacy_sampling_displacement"]["count"] == 3
    assert warp.calls[0][0].shape == (3, 2)


def test_second_run_without_warp_replaces_directory_and_removes_old_arrows(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path], tmp_path: Path
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    output_dir = tmp_path / "replace_report"
    write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images=images,
        output_dir=output_dir,
        expected_image_size=(32, 32),
        sampling_warps={"front": RecordingWarp()},
    )
    arrow_path = output_dir / "front_legacy_warp_arrows.jpg"
    assert arrow_path.exists()

    result = write_projective_texture_truth_report(
        mesh_dir=mesh_dir,
        images=images,
        output_dir=output_dir,
        expected_image_size=(32, 32),
    )

    assert result["views"]["front"]["legacy_warp_present"] is False
    assert not arrow_path.exists()


def test_directory_publish_failure_restores_previous_report(
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    output_dir = tmp_path / "transaction_report"
    output_dir.mkdir()
    sentinel = output_dir / "old_report.txt"
    sentinel.write_text("keep me", "utf-8")
    real_replace = report_module.os.replace
    replace_calls = 0

    def fail_new_directory_swap(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated staging publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_new_directory_swap)

    with pytest.raises(OSError, match="simulated"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=output_dir,
            expected_image_size=(32, 32),
        )

    assert sentinel.read_text("utf-8") == "keep me"
    assert not (output_dir / "truth_metrics.json").exists()
    leftovers = [
        path.name
        for path in output_dir.parent.iterdir()
        if path.name.startswith(f".{output_dir.name}.")
    ]
    assert leftovers == []


@pytest.mark.parametrize("changed_input", ["face_mesh.obj", "cameras.json"])
def test_provenance_change_raises_without_publishing_report(
    changed_input: str,
    tiny_scene: tuple[Path, dict[str, np.ndarray], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh_dir, images, _semantic_path = tiny_scene
    output_dir = tmp_path / "provenance_failure"
    original_write_image = report_module._write_image
    mutated = False

    def write_then_mutate(path: Path, image: np.ndarray) -> None:
        nonlocal mutated
        original_write_image(path, image)
        if not mutated:
            with (mesh_dir / changed_input).open("a", encoding="utf-8") as handle:
                handle.write("# concurrent mutation\n")
            mutated = True

    monkeypatch.setattr(report_module, "_write_image", write_then_mutate)

    with pytest.raises(RuntimeError, match="provenance"):
        write_projective_texture_truth_report(
            mesh_dir=mesh_dir,
            images=images,
            output_dir=output_dir,
            expected_image_size=(32, 32),
        )

    assert not (output_dir / "truth_metrics.json").exists()
    assert not (output_dir / "index.html").exists()
