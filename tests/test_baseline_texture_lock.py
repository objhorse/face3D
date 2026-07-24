from pathlib import Path

import numpy as np
import pytest

from src.appearance.baseline_texture_lock import (
    BaselineLockError,
    BaselineTextureLock,
    MeshContract,
)


def test_baseline_lock_rejects_changed_artifact() -> None:
    lock = BaselineTextureLock(
        root=Path("."),
        hashes={"face_mesh.obj": "AAA", "cameras.json": "BBB"},
    )
    lock.assert_hashes({"face_mesh.obj": "AAA", "cameras.json": "BBB"})

    with pytest.raises(BaselineLockError, match="cameras.json"):
        lock.assert_hashes({"face_mesh.obj": "AAA", "cameras.json": "CHANGED"})


def test_mesh_contract_detects_geometry_or_uv_change() -> None:
    baseline = MeshContract(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uv=np.zeros((3, 2), dtype=np.float32),
        uv_faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    changed = MeshContract(
        vertices=baseline.vertices.copy(),
        faces=baseline.faces.copy(),
        uv=baseline.uv.copy(),
        uv_faces=baseline.uv_faces.copy(),
    )
    changed.vertices[0, 0] = 0.25

    with pytest.raises(BaselineLockError, match="vertices"):
        baseline.assert_identical(changed)


def test_mesh_contract_uses_exact_topology_and_uv_arrays() -> None:
    baseline = MeshContract(
        vertices=np.zeros((3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        uv=np.zeros((3, 2), dtype=np.float32),
        uv_faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    changed = MeshContract(
        vertices=baseline.vertices.copy(),
        faces=np.array([[0, 2, 1]], dtype=np.int32),
        uv=baseline.uv.copy(),
        uv_faces=baseline.uv_faces.copy(),
    )

    with pytest.raises(BaselineLockError, match="faces"):
        baseline.assert_identical(changed)
