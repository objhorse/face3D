from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from run_semantic_eyelid_stage_c import _rename_viewer, verify_stage_c_source


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_c_source_is_explicit_and_hash_locked(tmp_path: Path) -> None:
    source = tmp_path / "a2"
    meshes = source / "meshes"
    textures = source / "textures"
    meshes.mkdir(parents=True)
    textures.mkdir(parents=True)
    paths = {
        "obj": meshes / "face_mesh.obj",
        "geometry": meshes / "candidate.glb",
        "textured": meshes / "candidate_textured.glb",
        "texture": textures / "albedo_baseline_locked.png",
    }
    for name, path in paths.items():
        path.write_bytes(f"locked-{name}".encode("ascii"))
    (source / "nasal_base_report.json").write_text(
        json.dumps(
            {
                "schema": "nasal-base-semantic-a2-v1",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )

    result = verify_stage_c_source(
        source,
        expected_obj_sha256=_sha256(paths["obj"]),
        expected_geometry_sha256=_sha256(paths["geometry"]),
        expected_textured_sha256=_sha256(paths["textured"]),
        expected_texture_sha256=_sha256(paths["texture"]),
    )

    assert result["source"] == source.resolve()
    assert result["hashes"]["textured"] == _sha256(paths["textured"])


def test_stage_c_rejects_changed_approved_source(tmp_path: Path) -> None:
    source = tmp_path / "a2"
    meshes = source / "meshes"
    textures = source / "textures"
    meshes.mkdir(parents=True)
    textures.mkdir(parents=True)
    paths = {
        "obj": meshes / "face_mesh.obj",
        "geometry": meshes / "candidate.glb",
        "textured": meshes / "candidate_textured.glb",
        "texture": textures / "albedo_baseline_locked.png",
    }
    for name, path in paths.items():
        path.write_bytes(f"locked-{name}".encode("ascii"))
    (source / "nasal_base_report.json").write_text(
        json.dumps(
            {
                "schema": "nasal-base-semantic-a2-v1",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    wrong_hash = "0" * 64

    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_stage_c_source(
            source,
            expected_obj_sha256=wrong_hash,
            expected_geometry_sha256=_sha256(paths["geometry"]),
            expected_textured_sha256=_sha256(paths["textured"]),
            expected_texture_sha256=_sha256(paths["texture"]),
        )


def test_stage_c_viewer_labels_name_the_accepted_a2_source(
    tmp_path: Path,
) -> None:
    dataset = "captures_example"
    label = f"{dataset} | semantic eyelid Stage C"
    viewer = tmp_path / "viewer.html"
    viewer.write_text(
        "\n".join(
            (
                f"<title>{label} | Baseline vs unified nasal shape</title>",
                (
                    f'<div id="baseline-label" class="label">{label} | '
                    "Baseline: protected expression depth v3</div>"
                ),
                (
                    f'<div id="candidate-label" class="label">{label} | '
                    "New: unified multiview nasal shape</div>"
                ),
            )
        ),
        encoding="utf-8",
    )

    _rename_viewer(viewer, dataset)

    text = viewer.read_text(encoding="utf-8")
    assert "accepted A2 vs fixed-corner eyelids" in text
    assert f"{label} | accepted A2" in text
    assert f"{label} | fixed-corner eyelid fit" in text
    assert "protected expression depth v3" not in text
