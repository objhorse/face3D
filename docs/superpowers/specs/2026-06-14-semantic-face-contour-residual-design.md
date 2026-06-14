# Semantic Face Contour Residual Design

Date: 2026-06-14

## Background

The first personal residual deformation experiment proved that bounded per-vertex residual offsets can be inserted after FLAME shape/pose refinement without destabilizing eyes, nose, and mouth. It was accepted by the safety gates, but the visual gain was modest:

- Overall full-mask dense contour improved from 74.963 px to 68.617 px.
- Left side improved by 6.824 px.
- Front improved by 10.766 px.
- Right side improved by only 1.449 px.
- Sparse landmark contour slightly worsened, while stable facial landmarks stayed unchanged.

Visual review showed the main cause: the current contour target follows the full face mask. On side views, that mask includes ear, hair, neck, and background-related boundaries. These regions should not drive face-shape deformation. The user specifically identified cheek, jaw, and forehead shape as the inaccurate areas.

## Goal

Replace the personal residual contour target with a semantic face-shape contour target:

- Strongly optimize cheek, jawline, and chin.
- Lightly optimize forehead / lower hairline region.
- Protect eyes, nose, and mouth.
- Exclude ears, hair outer boundary, neck, shoulders, and background edges.

The next experiment should measure face-shape improvement on this semantic contour, not only on the older full-mask contour.

## Non-Goals

- Do not introduce a new neural model or training step.
- Do not change FLAME topology or identity parameter dimensions.
- Do not solve texture holes, mouth interior artifacts, nostril texture, or eyeball rendering.
- Do not freely deform ears or hair; those areas are explicitly excluded from this stage.

## Design

### 1. Semantic Contour Data

Add a semantic contour builder for the personal residual stage. It will still use the existing mask row-bound machinery, but it will restrict valid rows and boundary targets using landmarks:

- Vertical range:
  - Top: forehead / upper face oval percentile, with a conservative offset.
  - Bottom: chin landmark plus a small margin.
- Side-view target:
  - Primary target side follows the visible face outline from forehead through cheek to chin.
  - The ear-side mask boundary is excluded or strongly down-weighted.
- Front-view target:
  - Both left and right facial outlines are used.
  - Nose, mouth, and eye regions are protected anchors, not deformation targets.

The builder returns the same structure used by the current residual code, plus metadata:

- `semantic_mode`
- `excluded_regions`
- `region_weights`
- `metric_name`

This keeps downstream residual accumulation, smoothing, safety gates, and debug rendering mostly reusable.

### 2. Region Weights

Rows receive semantic weights:

- Jaw and lower cheek: high weight.
- Mid cheek: high weight.
- Forehead: lower weight because hair and hairline are ambiguous.
- Rows near mouth/nose/eyes: low target weight or protected.

This prevents the optimizer from spending residual capacity on unstable boundaries.

### 3. Metrics

The report should distinguish:

- `full_mask_dense_contour_px`: old metric, useful for continuity.
- `semantic_face_contour_px`: new metric, used for acceptance.

Acceptance should primarily use semantic side-view improvement:

- Side semantic mean improvement >= 3 px for the first implementation.
- At least one side semantic improvement >= 5 px.
- No side semantic worsening > 1 px.
- Stable landmarks worsening <= 0.75 px.
- Global landmark worsening <= 1 px.
- Existing mesh safety limits remain active.

The thresholds are intentionally conservative for the first semantic pass. If the first run is stable but under-aggressive, tune weights and row ranges rather than relaxing safety gates first.

### 4. Debug Outputs

Extend `output/debug/personal_residual_deform/` with semantic-specific visual checks:

- Before/after semantic contour overlay.
- Optional full-mask overlay for comparison.
- Residual heatmap projection.
- Summary fields for semantic metrics and excluded target regions.

The main evaluation HTML should show whether the accepted deformation was driven by semantic face contour or the older full-mask contour.

## Risks

- Landmark-based semantic cropping can miss the true forehead if hair covers the upper face.
- Side-view MediaPipe landmarks may be less reliable around the far cheek and jaw.
- Excluding ear-side boundaries too aggressively may reduce constraints on face width.
- The residual field may improve 2D silhouette while still being approximate in true 3D volume.

## Decision

Proceed with a semantic face-contour target for personal residual deformation. The first pass should prioritize cheek and jawline, keep forehead soft, and use the new semantic metric as the acceptance signal while retaining the original mesh safety gates.
