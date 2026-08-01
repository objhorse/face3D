"""Topology-preserving harmonic deformation bases for semantic face controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from src.geometry.semantic_eyelid_rig import SemanticEyelidRig


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        (faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _select_control_handles(
    rig: SemanticEyelidRig,
    boundary_mask: np.ndarray,
    *,
    handles_per_control: int,
) -> tuple[np.ndarray, np.ndarray]:
    used: set[int] = set()
    vertices = []
    controls = []
    active = np.asarray(rig.active_vertices, dtype=np.int64)
    weights = np.asarray(rig.weights, dtype=np.float64)
    for control_index, name in enumerate(rig.control_names):
        seeds = np.asarray(rig.control_seeds[name], dtype=np.int64)
        candidates = [
            int(vertex)
            for vertex in seeds[np.argsort(weights[seeds, control_index])[::-1]]
            if not boundary_mask[int(vertex)] and int(vertex) not in used
        ]
        if len(candidates) < int(handles_per_control):
            ranked = active[np.argsort(weights[active, control_index])[::-1]]
            candidates.extend(
                int(vertex)
                for vertex in ranked
                if (
                    not boundary_mask[int(vertex)]
                    and int(vertex) not in used
                    and int(vertex) not in candidates
                )
            )
        selected = candidates[: int(handles_per_control)]
        if not selected:
            raise RuntimeError(f"no interior harmonic handles available for {name}")
        for vertex in selected:
            used.add(vertex)
            vertices.append(vertex)
            controls.append(control_index)
    return (
        np.asarray(vertices, dtype=np.int64),
        np.asarray(controls, dtype=np.int64),
    )


@dataclass(frozen=True)
class HarmonicSemanticDeformer:
    baseline_vertices: np.ndarray
    parameter_basis: np.ndarray
    active_vertices: np.ndarray
    diagnostics: dict

    @classmethod
    def build(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        rig: SemanticEyelidRig,
        control_offset_basis: np.ndarray,
        *,
        handles_per_control: int = 2,
    ) -> "HarmonicSemanticDeformer":
        baseline = np.asarray(vertices, dtype=np.float64)
        faces_i = np.asarray(faces, dtype=np.int64)
        control_basis = np.asarray(control_offset_basis, dtype=np.float64)
        expected = (len(rig.control_names), 3)
        if control_basis.ndim != 3 or control_basis.shape[1:] != expected:
            raise ValueError(
                "control_offset_basis must have shape "
                f"(parameters, {expected[0]}, 3)"
            )
        if faces_i.ndim != 2 or faces_i.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        active = np.unique(
            np.asarray(rig.active_vertices, dtype=np.int64)
        )
        if len(active) == 0:
            raise ValueError("semantic rig has no active vertices")

        edges = _unique_edges(faces_i)
        active_mask = np.zeros(len(baseline), dtype=bool)
        active_mask[active] = True
        edge_active_a = active_mask[edges[:, 0]]
        edge_active_b = active_mask[edges[:, 1]]
        crossing = edge_active_a ^ edge_active_b
        boundary = np.unique(
            np.where(
                edge_active_a[crossing],
                edges[crossing, 0],
                edges[crossing, 1],
            )
        ).astype(np.int64)
        boundary_mask = np.zeros(len(baseline), dtype=bool)
        boundary_mask[boundary] = True

        handle_vertices, handle_controls = _select_control_handles(
            rig,
            boundary_mask,
            handles_per_control=int(handles_per_control),
        )
        fixed = np.unique(
            np.concatenate((boundary, handle_vertices))
        ).astype(np.int64)
        fixed_mask = np.zeros(len(baseline), dtype=bool)
        fixed_mask[fixed] = True
        free = active[~fixed_mask[active]]
        free_index = np.full(len(baseline), -1, dtype=np.int64)
        free_index[free] = np.arange(len(free), dtype=np.int64)
        fixed_index = np.full(len(baseline), -1, dtype=np.int64)
        fixed_index[fixed] = np.arange(len(fixed), dtype=np.int64)

        internal_edges = edges[
            active_mask[edges[:, 0]] & active_mask[edges[:, 1]]
        ]
        lengths = np.linalg.norm(
            baseline[internal_edges[:, 1]]
            - baseline[internal_edges[:, 0]],
            axis=1,
        )
        positive = lengths[lengths > 1e-12]
        reference = float(np.median(positive)) if len(positive) else 1.0
        conductance = 1.0 / np.maximum(
            lengths,
            max(reference * 0.05, 1e-12),
        )

        rows = []
        columns = []
        values = []
        diagonal = np.zeros(len(free), dtype=np.float64)
        free_fixed_edges: list[tuple[int, int, float]] = []
        for (vertex_a, vertex_b), weight in zip(
            internal_edges,
            conductance,
        ):
            a = int(vertex_a)
            b = int(vertex_b)
            row_a = int(free_index[a])
            row_b = int(free_index[b])
            if row_a >= 0:
                diagonal[row_a] += float(weight)
                if row_b >= 0:
                    rows.append(row_a)
                    columns.append(row_b)
                    values.append(-float(weight))
                else:
                    free_fixed_edges.append(
                        (row_a, int(fixed_index[b]), float(weight))
                    )
            if row_b >= 0:
                diagonal[row_b] += float(weight)
                if row_a >= 0:
                    rows.append(row_b)
                    columns.append(row_a)
                    values.append(-float(weight))
                else:
                    free_fixed_edges.append(
                        (row_b, int(fixed_index[a]), float(weight))
                    )
        rows.extend(range(len(free)))
        columns.extend(range(len(free)))
        values.extend(diagonal.tolist())
        system = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(free), len(free)),
            dtype=np.float64,
        ).tocsc()
        if len(free):
            regularization = max(float(np.median(diagonal)), 1.0) * 1e-10
            system = system + regularization * sparse.eye(
                len(free),
                format="csc",
            )
            factor = splu(system)
        else:
            factor = None

        parameter_basis = np.zeros(
            (len(control_basis), len(baseline), 3),
            dtype=np.float32,
        )
        handle_control_by_vertex = {
            int(vertex): int(control)
            for vertex, control in zip(handle_vertices, handle_controls)
        }
        for parameter_index in range(len(control_basis)):
            fixed_displacements = np.zeros(
                (len(fixed), 3),
                dtype=np.float64,
            )
            for fixed_vertex in fixed:
                control_index = handle_control_by_vertex.get(int(fixed_vertex))
                if control_index is not None:
                    fixed_displacements[int(fixed_index[fixed_vertex])] = (
                        control_basis[parameter_index, control_index]
                    )
            displacement = np.zeros_like(baseline)
            displacement[fixed] = fixed_displacements
            if len(free):
                rhs = np.zeros((len(free), 3), dtype=np.float64)
                for row, column, weight in free_fixed_edges:
                    rhs[row] += weight * fixed_displacements[column]
                displacement[free] = np.column_stack(
                    [factor.solve(rhs[:, axis]) for axis in range(3)]
                )
            parameter_basis[parameter_index] = displacement.astype(np.float32)

        return cls(
            baseline_vertices=baseline.copy(),
            parameter_basis=parameter_basis,
            active_vertices=active,
            diagnostics={
                "mode": "harmonic_dirichlet_parameter_basis",
                "parameter_count": int(len(control_basis)),
                "active_vertex_count": int(len(active)),
                "boundary_vertex_count": int(len(boundary)),
                "handle_vertex_count": int(len(handle_vertices)),
                "free_vertex_count": int(len(free)),
                "handles_per_control": int(handles_per_control),
            },
        )

    def apply(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if values.shape != (len(self.parameter_basis),):
            raise ValueError(
                "harmonic coefficients must match the parameter basis"
            )
        displacement = np.tensordot(
            values,
            np.asarray(self.parameter_basis, dtype=np.float64),
            axes=(0, 0),
        )
        candidate = self.baseline_vertices + displacement
        outside = np.setdiff1d(
            np.arange(len(candidate), dtype=np.int64),
            self.active_vertices,
        )
        candidate[outside] = self.baseline_vertices[outside]
        return candidate
