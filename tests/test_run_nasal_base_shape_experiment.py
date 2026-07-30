from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from run_nasal_base_shape_experiment import verify_hash_locked_file


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_hash_locked_file_accepts_exact_artifact(tmp_path):
    artifact = tmp_path / "candidate.glb"
    artifact.write_bytes(b"fixed-v4-candidate")

    assert verify_hash_locked_file(artifact, _digest(artifact)) == _digest(artifact)


def test_verify_hash_locked_file_rejects_changed_artifact(tmp_path):
    artifact = tmp_path / "candidate.glb"
    artifact.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_hash_locked_file(artifact, "0" * 64)


@pytest.mark.parametrize("bad_hash", ("xyz", "A" * 64, "0" * 63))
def test_verify_hash_locked_file_requires_canonical_sha256(tmp_path, bad_hash):
    artifact = tmp_path / "candidate.glb"
    artifact.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="64 lowercase hex"):
        verify_hash_locked_file(artifact, bad_hash)
