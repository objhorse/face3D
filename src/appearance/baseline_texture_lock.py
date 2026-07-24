"""Immutable input checks for texture-only experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


class BaselineLockError(RuntimeError):
    """Raised when a texture-only experiment changes a frozen input."""


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class BaselineTextureLock:
    root: Path
    hashes: Mapping[str, str]

    @classmethod
    def capture(cls, root: Path, relative_paths: Iterable[str]) -> "BaselineTextureLock":
        root = Path(root).resolve()
        hashes: dict[str, str] = {}
        for relative in relative_paths:
            artifact = root / relative
            if not artifact.is_file():
                raise BaselineLockError(f"missing baseline artifact: {relative}")
            hashes[str(relative).replace("\\", "/")] = file_sha256(artifact)
        return cls(root=root, hashes=hashes)

    @classmethod
    def from_manifest(cls, manifest_path: Path, root: Path | None = None) -> "BaselineTextureLock":
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        lock_root = Path(root) if root is not None else Path(data["root"])
        return cls(root=lock_root.resolve(), hashes=dict(data["hashes"]))

    def verify(self) -> dict[str, str]:
        actual: dict[str, str] = {}
        for relative, expected in self.hashes.items():
            artifact = self.root / relative
            if not artifact.is_file():
                continue
            actual_hash = file_sha256(artifact)
            actual[relative] = actual_hash
        self.assert_hashes(actual)
        return actual

    def assert_hashes(self, actual: Mapping[str, str]) -> None:
        mismatches: list[str] = []
        for relative, expected in self.hashes.items():
            actual_hash = actual.get(relative)
            if actual_hash is None:
                mismatches.append(f"{relative}: missing")
            elif str(actual_hash).upper() != str(expected).upper():
                mismatches.append(f"{relative}: expected {expected}, got {actual_hash}")
        if mismatches:
            raise BaselineLockError("baseline lock mismatch: " + "; ".join(mismatches))

    def write_manifest(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"root": str(self.root), "hashes": dict(self.hashes)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


@dataclass(frozen=True)
class MeshContract:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    uv_faces: np.ndarray

    @classmethod
    def from_obj(cls, obj_path: Path) -> "MeshContract":
        from src.module3_texture import load_mesh_obj

        vertices, faces, uv, uv_faces = load_mesh_obj(Path(obj_path))
        return cls(vertices=vertices, faces=faces, uv=uv, uv_faces=uv_faces)

    def assert_identical(self, candidate: "MeshContract") -> None:
        for field_name in ("vertices", "faces", "uv", "uv_faces"):
            expected = np.asarray(getattr(self, field_name))
            actual = np.asarray(getattr(candidate, field_name))
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                raise BaselineLockError(
                    f"mesh contract changed: {field_name} "
                    f"{expected.shape} -> {actual.shape}"
                )
