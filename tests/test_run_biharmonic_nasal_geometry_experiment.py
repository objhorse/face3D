from __future__ import annotations

import numpy as np

from run_biharmonic_nasal_geometry_experiment import _distance_metrics


def test_identity_distance_is_measured_inside_the_nasal_support() -> None:
    baseline = np.zeros((100, 3), dtype=np.float64)
    candidate = baseline.copy()
    candidate[:4, 0] = 0.002
    nasal_support = np.zeros(100, dtype=bool)
    nasal_support[:4] = True

    full = _distance_metrics(baseline, candidate)
    nasal = _distance_metrics(
        baseline,
        candidate,
        vertex_mask=nasal_support,
    )

    assert full["p95_mm"] == 0.0
    assert nasal["p95_mm"] == 2.0
