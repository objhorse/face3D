from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_projective_texture_truth_diagnostic as diagnostic


def _touch_capture_set(root: Path) -> None:
    (root / "camera1_subject.JPG").write_bytes(b"left")
    (root / "camera2_subject.jpeg").write_bytes(b"front")
    (root / "camera3_subject.JpEg").write_bytes(b"right")


def test_discover_capture_images_maps_subject_views_and_extensions(tmp_path: Path) -> None:
    _touch_capture_set(tmp_path)
    (tmp_path / "camera4_subject.jpg").write_bytes(b"ignored")
    (tmp_path / "camera1_subject.png").write_bytes(b"ignored")

    discovered = diagnostic.discover_capture_images(tmp_path)

    assert discovered == {
        "left": tmp_path / "camera1_subject.JPG",
        "front": tmp_path / "camera2_subject.jpeg",
        "right": tmp_path / "camera3_subject.JpEg",
    }


def test_discover_capture_images_rejects_missing_camera(tmp_path: Path) -> None:
    (tmp_path / "camera1_subject.jpg").write_bytes(b"left")
    (tmp_path / "camera2_subject.jpg").write_bytes(b"front")

    with pytest.raises(ValueError, match=r"camera3.*found 0"):
        diagnostic.discover_capture_images(tmp_path)


def test_discover_capture_images_rejects_duplicate_camera(tmp_path: Path) -> None:
    _touch_capture_set(tmp_path)
    (tmp_path / "camera1_second.jpeg").write_bytes(b"duplicate")

    with pytest.raises(ValueError, match=r"camera1.*found 2"):
        diagnostic.discover_capture_images(tmp_path)


def test_cli_requires_all_paths_and_defaults_to_no_legacy_warp(tmp_path: Path) -> None:
    args = diagnostic.parse_args(
        [
            "--capture-dir",
            str(tmp_path / "captures"),
            "--mesh-dir",
            str(tmp_path / "meshes"),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert args.capture_dir == tmp_path / "captures"
    assert args.mesh_dir == tmp_path / "meshes"
    assert args.output_dir == tmp_path / "report"
    assert args.with_legacy_warp is False

    with pytest.raises(SystemExit):
        diagnostic.parse_args(["--capture-dir", str(tmp_path)])


def test_cli_accepts_legacy_warp_flag(tmp_path: Path) -> None:
    args = diagnostic.parse_args(
        [
            "--capture-dir",
            str(tmp_path / "captures"),
            "--mesh-dir",
            str(tmp_path / "meshes"),
            "--output-dir",
            str(tmp_path / "report"),
            "--with-legacy-warp",
        ]
    )

    assert args.with_legacy_warp is True


@pytest.mark.parametrize(
    ("capture_relative", "mesh_relative", "output_relative"),
    [
        ("shared", "mesh", "shared"),
        ("captures", "mesh", "captures/output"),
        ("output/captures", "mesh", "output"),
        ("captures", "shared", "shared"),
        ("captures", "mesh", "mesh/output"),
        ("captures", "output/mesh", "output"),
    ],
    ids=[
        "output-equals-capture",
        "output-inside-capture",
        "output-contains-capture",
        "output-equals-mesh",
        "output-inside-mesh",
        "output-contains-mesh",
    ],
)
def test_run_diagnostic_rejects_input_output_path_overlap_before_lock_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_relative: str,
    mesh_relative: str,
    output_relative: str,
) -> None:
    calls: list[str] = []
    capture_path = tmp_path / capture_relative
    mesh_path = tmp_path / mesh_relative
    output_path = tmp_path / output_relative
    capture_path.mkdir(parents=True, exist_ok=True)
    mesh_path.mkdir(parents=True, exist_ok=True)
    capture_marker = capture_path / "capture.keep"
    mesh_marker = mesh_path / "mesh.keep"
    capture_marker.write_text("capture", encoding="utf-8")
    mesh_marker.write_text("mesh", encoding="utf-8")

    class ForbiddenLock:
        @classmethod
        def capture(cls, *args, **kwargs):
            calls.append("lock")
            raise AssertionError("lock must not be created for overlapping paths")

    def forbidden_report(**kwargs):
        calls.append("report")
        raise AssertionError("report must not run for overlapping paths")

    monkeypatch.setattr(diagnostic, "BaselineTextureLock", ForbiddenLock)
    monkeypatch.setattr(
        diagnostic, "write_projective_texture_truth_report", forbidden_report
    )

    with pytest.raises(ValueError, match="must not overlap"):
        diagnostic.run_diagnostic(
            capture_path,
            mesh_path,
            output_path,
        )

    assert calls == []
    assert capture_marker.read_text(encoding="utf-8") == "capture"
    assert mesh_marker.read_text(encoding="utf-8") == "mesh"


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[object] = []
    work_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    hires_image = np.zeros((32, 48, 3), dtype=np.uint8)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text('{"calibration": true}', encoding="utf-8")
    cfg = SimpleNamespace(
        CAMERA_CALIBRATION_PATH=calibration_path,
        DUST3R_DIR=tmp_path / "dust3r",
        MANUAL_INTRINSICS=None,
        UNDISTORT_ALPHA=0.25,
        UNDISTORT_IMAGES=True,
        WORK_IMAGE_SIZE=1024,
    )
    camera_k = np.array(
        [[800.0, 0.0, 512.0], [0.0, 805.0, 512.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    calibration_intrinsics = {
        view: camera_k.copy() for view in ("left", "front", "right")
    }

    class FakeLock:
        @classmethod
        def capture(cls, root, relative_paths):
            events.append(("capture", Path(root), list(relative_paths)))
            return cls()

        def verify(self):
            events.append("verify")
            return {"face_mesh.obj": "mesh", "cameras.json": "camera"}

    def fake_load_images(capture_dir, view_names):
        events.append(("load", Path(capture_dir), dict(view_names)))
        return {view: hires_image.copy() for view in view_names}

    def fake_undistort(images, calibration_path, *, alpha):
        events.append(("undistort", calibration_path, alpha))
        return (
            {view: image.copy() for view, image in images.items()},
            calibration_intrinsics,
        )

    def fake_get_intrinsics(**kwargs):
        events.append(("get_intrinsics", kwargs))
        return {view: camera_k.copy() for view in kwargs["images"]}

    def fake_load_cameras(path):
        events.append(("load_cameras", Path(path)))
        return {
            view: {
                "K": camera_k.copy(),
                "R": np.eye(3),
                "t": np.zeros(3),
            }
            for view in ("left", "front", "right")
        }

    def fake_preprocess(images, *, debug_dir, target_size):
        events.append(("preprocess", Path(debug_dir), target_size))
        return {view: {"image": work_image.copy()} for view in images}

    report_calls: list[dict] = []

    def fake_report(**kwargs):
        events.append("report")
        report_calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return {"final_sampling_mode": "unwarped_projective"}

    monkeypatch.setattr(diagnostic, "cfg", cfg)
    monkeypatch.setattr(diagnostic, "BaselineTextureLock", FakeLock)
    monkeypatch.setattr(diagnostic, "load_images", fake_load_images)
    monkeypatch.setattr(
        diagnostic, "undistort_images_with_calibration", fake_undistort
    )
    monkeypatch.setattr(diagnostic, "get_intrinsics", fake_get_intrinsics)
    monkeypatch.setattr(diagnostic, "load_cameras", fake_load_cameras)
    monkeypatch.setattr(diagnostic, "preprocess_all_views", fake_preprocess)
    monkeypatch.setattr(
        diagnostic, "write_projective_texture_truth_report", fake_report
    )
    return events, report_calls, work_image


def test_run_diagnostic_uses_work_images_and_no_warp_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, report_calls, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    result = diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert result["final_sampling_mode"] == "unwarped_projective"
    assert events[0] == (
        "capture",
        mesh_dir.resolve(),
        ["face_mesh.obj", "cameras.json"],
    )
    assert events[-2:] == ["report", "verify"]
    load_event = next(event for event in events if isinstance(event, tuple) and event[0] == "load")
    assert load_event[2] == {
        "left": "camera1_subject.JPG",
        "front": "camera2_subject.jpeg",
        "right": "camera3_subject.JpEg",
    }
    preprocess_event = next(
        event for event in events if isinstance(event, tuple) and event[0] == "preprocess"
    )
    assert preprocess_event[1].parent == output_dir.resolve().parent
    assert preprocess_event[1] != output_dir.resolve()
    assert preprocess_event[1].name.startswith(output_dir.name)
    assert preprocess_event[2] == 1024
    assert ("undistort", diagnostic.cfg.CAMERA_CALIBRATION_PATH, 0.25) in events

    call = report_calls[0]
    assert call["mesh_dir"] == mesh_dir.resolve()
    assert call["output_dir"].parent == output_dir.resolve().parent
    assert call["output_dir"] != output_dir.resolve()
    assert call["output_dir"].name.startswith(".truth.candidate-")
    assert call["expected_image_size"] == (1024, 1024)
    assert call["sampling_warps"] is None
    assert call["semantic_regions_path"] is None
    assert set(call["images"]) == {"left", "front", "right"}
    assert all(image.shape == (1024, 1024, 3) for image in call["images"].values())

    intrinsics_call = next(
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "get_intrinsics"
    )
    assert all(
        image.shape == (32, 48, 3)
        for image in intrinsics_call["images"].values()
    )
    assert intrinsics_call["manual_intrinsics"] is None
    assert intrinsics_call["calibration_path"] == diagnostic.cfg.CAMERA_CALIBRATION_PATH
    assert set(intrinsics_call["calibration_intrinsics"]) == {
        "left",
        "front",
        "right",
    }
    assert intrinsics_call["work_image_size"] == 1024
    assert intrinsics_call["dust3r_dir"] == diagnostic.cfg.DUST3R_DIR

    provenance_path = output_dir / "preprocess_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(
        diagnostic.cfg.CAMERA_CALIBRATION_PATH.read_bytes()
    ).hexdigest()
    assert provenance["validated"] is True
    assert provenance["work_image_size"] == 1024
    assert provenance["undistort_images"] is True
    assert provenance["undistort_alpha"] == 0.25
    assert provenance["calibration"]["path"] == str(
        diagnostic.cfg.CAMERA_CALIBRATION_PATH.resolve()
    )
    assert provenance["calibration"]["sha256"] == expected_hash
    assert provenance["calibration"]["used_for_undistortion"] is True
    assert provenance["intrinsics_source"]["mode"] == "calibration_or_pipeline_default"
    assert provenance["camera_to_subject_view"] == {
        "camera1": "left",
        "camera2": "front",
        "camera3": "right",
    }
    assert {
        view: source["camera"]
        for view, source in provenance["capture_images"].items()
    } == {"front": "camera2", "left": "camera1", "right": "camera3"}
    assert all(
        Path(source["path"]).is_absolute()
        and len(source["sha256"]) == 64
        for source in provenance["capture_images"].values()
    )
    assert set(provenance["views"]) == {"left", "front", "right"}
    assert all(view["validated"] for view in provenance["views"].values())
    assert all(
        view["k_max_abs_error"] == 0.0
        for view in provenance["views"].values()
    )


def test_run_diagnostic_passes_only_sampling_warps_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    semantic_path = mesh_dir / "stable_semantic_regions.json"
    semantic_path.write_text("{}", encoding="utf-8")
    events, report_calls, _ = _install_fake_pipeline(monkeypatch, tmp_path)
    sampling_warps = {"front": object()}

    def fake_registration(**kwargs):
        events.append(("registration", kwargs))
        return {
            "sampling_warps": sampling_warps,
            "hires_images": {"must_not": "reach report"},
            "feature_masks": {"must_not": "reach report"},
            "report": {"must_not": "change final mode"},
        }

    monkeypatch.setattr(
        diagnostic, "_prepare_legacy_registration", fake_registration
    )

    diagnostic.run_diagnostic(
        capture_dir,
        mesh_dir,
        output_dir,
        with_legacy_warp=True,
    )

    registration = next(event[1] for event in events if isinstance(event, tuple) and event[0] == "registration")
    assert registration["mesh_dir"] == mesh_dir.resolve()
    assert registration["debug_dir"].parent == output_dir.resolve().parent
    assert registration["debug_dir"] != output_dir.resolve()
    assert registration["debug_dir"].name.startswith(output_dir.name)
    assert registration["debug_dir"].name.endswith("_legacy_registration_a1")
    assert set(registration["preprocessed_views"]) == {"left", "front", "right"}
    assert all(image.shape == (32, 48, 3) for image in registration["hires_images"].values())
    assert registration["cfg"] is diagnostic.cfg

    call = report_calls[0]
    assert call["sampling_warps"] is sampling_warps
    assert call["semantic_regions_path"] == semantic_path.resolve()
    assert events[-2:] == ["report", "verify"]


def test_run_diagnostic_rejects_saved_camera_k_mismatch_before_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def mismatched_cameras(path):
        base_k = np.array(
            [[800.0, 0.0, 512.0], [0.0, 805.0, 512.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cameras = {
            view: {"K": base_k.copy(), "R": np.eye(3), "t": np.zeros(3)}
            for view in ("left", "front", "right")
        }
        cameras["front"]["K"][0, 0] += 0.01
        return cameras

    monkeypatch.setattr(diagnostic, "load_cameras", mismatched_cameras)

    with pytest.raises(ValueError, match=r"front.*K.*mismatch"):
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert "report" not in events
    assert events[-1] == "verify"
    assert not (output_dir / "preprocess_provenance.json").exists()


def test_run_diagnostic_writes_provenance_only_after_report_and_lock_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def record_provenance(output_path, provenance):
        events.append(("provenance", Path(output_path), provenance["validated"]))
        return Path(output_path) / "preprocess_provenance.json"

    monkeypatch.setattr(
        diagnostic,
        "_write_preprocess_provenance",
        record_provenance,
    )

    diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert events[-3] == "report"
    assert events[-2] == "verify"
    provenance_event = events[-1]
    assert provenance_event[0] == "provenance"
    assert provenance_event[1].parent == output_dir.resolve().parent
    assert provenance_event[1] != output_dir.resolve()
    assert provenance_event[1].name.startswith(".truth.candidate-")
    assert provenance_event[2] is True


def test_run_diagnostic_propagates_post_report_lock_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    output_dir.mkdir()
    old_marker = output_dir / "previous_report.txt"
    old_marker.write_text("keep", encoding="utf-8")
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    class FailingLock:
        @classmethod
        def capture(cls, root, relative_paths):
            events.append("capture_override")
            return cls()

        def verify(self):
            events.append("verify_failure")
            raise RuntimeError("baseline changed")

    monkeypatch.setattr(diagnostic, "BaselineTextureLock", FailingLock)

    with pytest.raises(RuntimeError, match="baseline changed"):
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert events[-2:] == ["report", "verify_failure"]
    assert old_marker.read_text(encoding="utf-8") == "keep"
    assert not (output_dir / "preprocess_provenance.json").exists()
    assert not any(
        path.name.startswith(".truth.candidate-")
        for path in output_dir.parent.iterdir()
    )


def test_legacy_registration_cannot_mutate_truth_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    _events, report_calls, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def mutating_registration(**kwargs):
        for view_data in kwargs["preprocessed_views"].values():
            view_data["image"][:] = 255
        for image in kwargs["hires_images"].values():
            image[:] = 127
        return {"sampling_warps": None}

    monkeypatch.setattr(
        diagnostic, "_prepare_legacy_registration", mutating_registration
    )

    diagnostic.run_diagnostic(
        capture_dir, mesh_dir, output_dir, with_legacy_warp=True
    )

    truth_images = report_calls[0]["images"]
    assert all(np.count_nonzero(image) == 0 for image in truth_images.values())
    assert all(not image.flags.writeable for image in truth_images.values())


def test_manual_intrinsics_are_identified_in_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    _install_fake_pipeline(monkeypatch, tmp_path)

    diagnostic.run_diagnostic(
        capture_dir,
        mesh_dir,
        output_dir,
        manual_intrinsics={"source": "test"},
    )

    provenance = json.loads(
        (output_dir / "preprocess_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["intrinsics_source"]["mode"] == "manual_intrinsics"


def test_run_diagnostic_verifies_lock_when_report_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def failing_report(**kwargs):
        events.append("report_failure")
        raise ValueError("report failed")

    monkeypatch.setattr(
        diagnostic, "write_projective_texture_truth_report", failing_report
    )

    with pytest.raises(ValueError, match="report failed"):
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert events[-2:] == ["report_failure", "verify"]


def test_run_diagnostic_verifies_lock_when_preprocess_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def failing_preprocess(*args, **kwargs):
        events.append("preprocess_failure")
        raise ValueError("preprocess failed")

    monkeypatch.setattr(diagnostic, "preprocess_all_views", failing_preprocess)

    with pytest.raises(ValueError, match="preprocess failed"):
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert events[-2:] == ["preprocess_failure", "verify"]


def test_run_diagnostic_verifies_lock_when_registration_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    def failing_registration(**kwargs):
        events.append("registration_failure")
        raise ValueError("registration failed")

    monkeypatch.setattr(
        diagnostic, "_prepare_legacy_registration", failing_registration
    )

    with pytest.raises(ValueError, match="registration failed"):
        diagnostic.run_diagnostic(
            capture_dir,
            mesh_dir,
            output_dir,
            with_legacy_warp=True,
        )

    assert events[-2:] == ["registration_failure", "verify"]


def test_run_diagnostic_preserves_both_operation_and_verify_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    class FailingLock:
        @classmethod
        def capture(cls, root, relative_paths):
            return cls()

        def verify(self):
            events.append("verify_failure")
            raise RuntimeError("mesh changed")

    def failing_report(**kwargs):
        events.append("report_failure")
        raise ValueError("report failed")

    monkeypatch.setattr(diagnostic, "BaselineTextureLock", FailingLock)
    monkeypatch.setattr(
        diagnostic, "write_projective_texture_truth_report", failing_report
    )

    with pytest.raises(
        diagnostic.DiagnosticVerificationError,
        match=r"report failed.*mesh changed",
    ) as caught:
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert isinstance(caught.value.operation_error, ValueError)
    assert isinstance(caught.value.verification_error, RuntimeError)
    assert "ValueError: report failed" in caught.value.operation_traceback
    assert "RuntimeError: mesh changed" in caught.value.verification_traceback
    assert caught.value.__cause__ is caught.value.verification_error
    assert events[-2:] == ["report_failure", "verify_failure"]


@pytest.mark.parametrize(
    "interrupt",
    [KeyboardInterrupt("stop"), SystemExit("stop")],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_run_diagnostic_keeps_interrupt_type_when_lock_verification_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: BaseException,
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)

    class FailingLock:
        @classmethod
        def capture(cls, root, relative_paths):
            return cls()

        def verify(self):
            events.append("verify_failure")
            raise RuntimeError("mesh changed")

    def interrupted_report(**kwargs):
        events.append("report_interrupted")
        raise interrupt

    monkeypatch.setattr(diagnostic, "BaselineTextureLock", FailingLock)
    monkeypatch.setattr(
        diagnostic, "write_projective_texture_truth_report", interrupted_report
    )

    with pytest.raises(type(interrupt)) as caught:
        diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert caught.value is interrupt
    assert isinstance(caught.value.verification_error, RuntimeError)
    assert "RuntimeError: mesh changed" in caught.value.verification_traceback
    assert events[-2:] == ["report_interrupted", "verify_failure"]


def test_run_diagnostic_skips_undistortion_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dir = tmp_path / "captures"
    mesh_dir = tmp_path / "meshes"
    output_dir = tmp_path / "truth"
    capture_dir.mkdir()
    mesh_dir.mkdir()
    _touch_capture_set(capture_dir)
    events, _, _ = _install_fake_pipeline(monkeypatch, tmp_path)
    diagnostic.cfg.UNDISTORT_IMAGES = False

    def forbidden_undistort(*args, **kwargs):
        raise AssertionError("undistortion must not run")

    monkeypatch.setattr(
        diagnostic, "undistort_images_with_calibration", forbidden_undistort
    )

    diagnostic.run_diagnostic(capture_dir, mesh_dir, output_dir)

    assert not any(
        isinstance(event, tuple) and event[0] == "undistort" for event in events
    )
    intrinsics_call = next(
        event[1]
        for event in events
        if isinstance(event, tuple) and event[0] == "get_intrinsics"
    )
    assert intrinsics_call["calibration_intrinsics"] is None
    provenance = json.loads(
        (output_dir / "preprocess_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["undistort_images"] is False
    assert events[-2:] == ["report", "verify"]


def test_default_module_import_does_not_require_legacy_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "src.appearance.stable_texture_registration",
        None,
    )

    namespace = runpy.run_path(
        str(Path(diagnostic.__file__).resolve()),
        run_name="projective_truth_without_legacy_registration",
    )

    assert callable(namespace["_prepare_legacy_registration"])
    with pytest.raises(SystemExit) as caught:
        namespace["parse_args"](["--help"])
    assert caught.value.code == 0


def test_main_prints_absolute_report_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "relative" / "truth"

    monkeypatch.setattr(
        diagnostic,
        "run_diagnostic",
        lambda *args, **kwargs: {"final_sampling_mode": "unwarped_projective"},
    )

    diagnostic.main(
        [
            "--capture-dir",
            str(tmp_path / "captures"),
            "--mesh-dir",
            str(tmp_path / "meshes"),
            "--output-dir",
            str(output_dir),
        ]
    )

    stdout = capsys.readouterr().out
    assert str((output_dir / "index.html").resolve()) in stdout
    assert str((output_dir / "truth_metrics.json").resolve()) in stdout
