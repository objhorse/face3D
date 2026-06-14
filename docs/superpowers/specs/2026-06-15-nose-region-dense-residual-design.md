# Nose Region Dense Residual Experiment

## Goal

Improve full nose likeness beyond sparse 68-point nose-base alignment. The current `nose_mouth_local_residual` pass safely improves landmarks 31-35, but it cannot learn nose bridge width, tip volume, nostril/wing contour, or side-view protrusion. This experiment adds a conservative nose-region residual pass before depth displacement.

## Approach

Add `nose_region_dense_residual`, disabled or independently gated by config if needed. The pass reuses the optimized FLAME mesh, per-view cameras, preprocessed RGB images, and 68-point landmarks.

The first version should:

1. Build a 2D nose ROI from landmarks 27-35, expanded around nose bridge, nose tip, and nose wings.
2. Select mesh vertices whose front-view projection falls inside this ROI, with side-view visibility used as supporting evidence.
3. Apply bounded local residual offsets from a small set of semantic handles: bridge center, tip/base center, left wing, right wing, and nose base.
4. Keep eyes, mouth, face contour, and non-nose vertices protected.
5. Gate acceptance on nose-region improvement, no protected-region worsening, no profile-contour worsening, bounded moved-vertex ratio, and bounded maximum displacement.

## Metrics

The debug report should include:

- Nose landmark mean error before/after.
- Nose-base error before/after.
- Front nose-width delta before/after.
- Side-view nose-tip/protrusion proxy before/after if available.
- Protected landmark worsening.
- Profile contour worsening.
- Moved vertices, moved ratio, mean/p95/max offset.

## Debug Output

Write outputs to `output/debug/nose_region_dense_residual/`:

- `summary.json`
- `index.html`
- per-view before/after overlays
- front-view ROI/selected-vertex visualization
- optional residual heatmap

## Safety

This is an experiment, not a replacement for the stable v6 path. If the new gates reject the deformation, the pipeline should continue with the previous mesh. The report must make rejection reasons visible so the next iteration can tune the signal without silently degrading the face.
