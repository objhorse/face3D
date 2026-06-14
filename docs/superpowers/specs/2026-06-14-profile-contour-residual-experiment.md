# Profile Contour Residual Experiment

Date: 2026-06-14

## Background

The semantic face-contour residual pass improved the face-shape metric safely, but the right-side view remained weak:

- Left semantic contour improved by 4.943 px.
- Front semantic contour improved by 9.657 px.
- Right semantic contour improved by only 1.835 px.
- Stable facial anchors did not worsen.

This suggests the residual solver is stable, but the side-view constraint is too broad. A side view should primarily follow the visible profile line: forehead, nose/front facial outline, lips, chin, jawline, and cheek transition. The opposite boundary often includes ear-side silhouette, inner projection, or ambiguous face width cues and should not have equal weight.

## Goal

Add an experimental profile-only mode for side views in the personal residual deformation stage:

- Side views use one primary profile boundary instead of both left and right mask boundaries.
- The front view still uses both boundaries for face width and symmetry.
- The experiment records profile metrics separately from the existing semantic contour metrics.
- Existing mesh safety gates stay active.

## Design

### Boundary Selection

For named side views:

- `left` view targets the left boundary as the main profile side.
- `right` view targets the right boundary as the main profile side.

This matches the current capture setup, where the visible nose/profile direction points toward the named view side.

If future data has different naming or pose conventions, this should become pose-derived from projected nose/chin direction instead of name-derived.

### Metrics

Each personal residual view record should include:

- `target_sides`
- `profile_target_side`
- `profile_before_px`
- `profile_after_px`
- `profile_improve_px`

The existing `before_dense_contour_px` / `after_dense_contour_px` fields will represent the acceptance target. In profile-only mode, side-view acceptance uses the profile boundary; front-view acceptance remains two-sided semantic contour.

### Debugging

Keep existing before/after contour overlays, but the metrics make clear which side was optimized. If needed later, add a dedicated right-side profile diagnostic page with target candidates, editable vertices, protected anchors, and residual heatmap.

## Acceptance

First experiment acceptance should remain conservative:

- At least one side profile improves by 4 px.
- Mean side profile improvement is at least 2 px.
- No side profile worsens by more than 1 px.
- Stable facial landmarks worsen by at most 0.75 px.
- Global landmark mean worsens by at most 1 px.
- Existing moved-ratio and edge-jump safety gates must pass.

## Risk

Profile-only can improve silhouette while losing information about cheek thickness. This is acceptable for this experiment because the current failure is visual face outline mismatch, not a validated 3D scan error.

## Decision

Proceed with profile-only side residual as an experiment layered on top of the semantic face contour target.
