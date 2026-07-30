# Balanced Nasal Then Eyelid Optimization Design

## Goal

Improve identity likeness in two strictly sequential stages:

1. **Stage A:** correct alar width and nose-tip shape.
2. **Stage C:** correct the eye and eyelid region after Stage A is visually accepted.

The work must preserve the current preferred model as an immutable reference and must not reintroduce patch-like quality gates as the optimization strategy.

## Canonical Baseline

The only canonical baseline for this work is:

- Dataset: `captures_20260612_135253`
- Artifact: `output/experiments/captures_20260612_135253_protected_expression_depth_v3/meshes/face_same_texture.glb`
- SHA-256: `BF12A6C78DC0F24679814F6973123807DE3D0976D1CD703306AE49AB6AE2E71B`

Every experiment records this hash and verifies it before running. The baseline GLB is read-only input. Candidate geometry, texture baking, viewer framing, camera state, and lighting must not overwrite or silently replace it.

The older model with hash `B17558815F862DA9EE86999551294D082C316BBF8CEE1CD3219638B8AE311860` is historical evidence only and is not the baseline for Stage A or Stage C.

## Root Cause Being Addressed

The previous nasal v3 objective did not represent visual likeness evenly:

- Side-profile terms contributed about `830` raw loss.
- Frontal alar-width terms contributed about `29` raw loss.
- The resulting influence was approximately `29:1` in favor of side profiles.
- Two broad FLAME directions and eight semantic nose modes were optimized together, so the solver could improve the numeric objective by altering unrelated or weakly identifiable shape directions.

This is an objective-design problem, not a reason to add more rejection gates. The new design separates observations by what they can identify, normalizes their influence, and exposes only semantic degrees of freedom required by the task.

## Stage A: Balanced Nasal Shape

### Parameterization

Stage A optimizes only these semantic parameters:

1. `alar_width_shared`
2. `alar_width_asymmetry`
3. `alar_depth_shared`
4. `alar_depth_asymmetry`
5. `tip_depth`
6. `tip_vertical`
7. `tip_roundness`
8. `tip_alar_fullness`

The two previous generic `observable_flame_*` directions are excluded from Stage A. The rest of the face is fixed to the canonical baseline.

### Evidence Ownership

Each observation group controls only parameters it can reliably identify:

- Frontal view:
  - alar width
  - left-right width asymmetry
  - limited tip vertical position
- Left and right oblique/profile views:
  - tip depth
  - tip roundness
  - alar depth
  - left-right depth asymmetry
- Three-view joint refinement:
  - all eight semantic parameters
  - starts from the staged solution
  - uses normalized group weights

No texture pixels participate in the Stage A geometric objective. Geometry observations come from detected boundaries, silhouettes, landmarks that have compatible semantics, and calibrated projections.

### Objective Normalization

Losses are normalized in two steps:

1. Divide each observation group by its number of valid effective observations.
2. Normalize frontal and profile group influence so that each contributes approximately half of the data term at initialization.

Robust loss is applied per residual to limit detector outliers. Regularization is expressed continuously in the objective:

- semantic parameter prior
- symmetry prior only for the shared/asymmetric decomposition
- local smoothness of the generated deformation
- preservation of non-nasal vertices

These are optimization terms, not post-hoc accept/reject patches.

### Stage A Outputs

Stage A writes a new experiment directory and never replaces the canonical baseline:

- `output/experiments/captures_20260612_135253_balanced_nasal_v4/meshes/baseline.glb`
- `output/experiments/captures_20260612_135253_balanced_nasal_v4/meshes/candidate.glb`
- `output/experiments/captures_20260612_135253_balanced_nasal_v4/meshes/candidate_textured.glb`
- `output/experiments/captures_20260612_135253_balanced_nasal_v4/balanced_nasal_compare.html`
- `output/experiments/captures_20260612_135253_balanced_nasal_v4/fit_report.json`

The viewer provides synchronized geometry-only and textured comparisons from front, subject-left, and subject-right views. Baseline and candidate use identical camera, lighting, background, material mode, and independent baseline-derived framing.

### Stage A Acceptance

Stage A is accepted by the user only after visual review. Supporting diagnostics must show:

- alar width is closer to the source front image;
- nose-tip depth and curvature are closer in both side views;
- nose bridge, mouth, chin, eyes, ears, and outer face contour remain unchanged within numerical tolerance;
- candidate mesh has no new non-finite, degenerate, or non-manifold geometry;
- baseline hash remains exactly `BF12A6C...2E71B`.

Numeric loss reduction alone is not acceptance.

## Stage C: Semantic Eyelid Shape

Stage C starts only from the user-approved Stage A candidate. It must not reopen nasal, mouth, jaw, face-contour, or global identity parameters.

### Eye-State Detection

The pipeline determines open or closed eye state independently for each eye and view using normalized eyelid aperture and detection confidence. Low-confidence views are down-weighted instead of forcing an eye-state decision.

### Parameterization

Stage C optimizes only:

- upper-lid vertical contour
- lower-lid vertical contour
- eyelid aperture
- local eyelid/ocular bulge
- left-right eyelid asymmetry

The eye-region deformation uses a localized semantic basis with smooth falloff. Eyebrows, nose bridge, cheeks, and temples are fixed outside the falloff support.

### Evidence and Objective

Compatible upper- and lower-eyelid contours drive the fit. Texture pixels are excluded from the geometric objective. Terms are normalized per eye and per view, then combined with local smoothness and small-deformation priors.

The existing semantic-eyelid implementation is generalized to accept an explicit baseline path and hash. It may not hard-code the historical `controlled_identity_v1` model.

### Stage C Outputs

- `output/experiments/captures_20260612_135253_semantic_eyelid_v2/meshes/baseline.glb`
- `output/experiments/captures_20260612_135253_semantic_eyelid_v2/meshes/candidate.glb`
- `output/experiments/captures_20260612_135253_semantic_eyelid_v2/meshes/candidate_textured.glb`
- `output/experiments/captures_20260612_135253_semantic_eyelid_v2/semantic_eyelid_compare.html`
- `output/experiments/captures_20260612_135253_semantic_eyelid_v2/fit_report.json`

Stage C uses the same viewer parity rules as Stage A.

## Architecture Changes

### Optimizer

Refactor the multiview nasal optimizer so observation groups expose:

- residual vector
- valid-observation count
- initial robust scale
- owned semantic parameters

The optimizer composes normalized group losses without knowing detector internals.

### Experiment Runner

Add an explicit Stage A runner configuration containing:

- canonical baseline path and expected hash
- eight-parameter semantic mode list
- staged and joint optimization settings
- output directory

Generalize the eyelid runner to accept the approved Stage A artifact as input.

### Reporting

Each report records:

- input artifact path and full SHA-256
- parameter names and values
- raw and normalized loss by view and observation group
- before/after projections
- non-target vertex displacement statistics
- mesh-quality comparison
- output artifact hashes

## Testing

### Unit Tests

- Loss normalization prevents observation count or raw scale from dominating a group.
- Parameter ownership masks gradients for non-owned parameters.
- Stage A exposes exactly eight semantic parameters.
- Generic FLAME directions are absent from Stage A.
- Baseline hash mismatch fails before optimization.
- Eyelid runner accepts an explicit baseline and does not use a hard-coded historical artifact.

### Regression Tests

- Non-nasal vertices remain unchanged within tolerance after Stage A.
- Non-eye vertices remain unchanged within tolerance after Stage C.
- Viewer baseline framing is derived from the baseline alone and is unaffected by candidate bounds.
- Geometry and textured GLBs refer to the same candidate vertex positions.
- Existing mesh-quality and viewer tests continue to pass.

### Experiment Validation

Run Stage A first on `captures_20260612_135253`. Do not start Stage C until the user approves the Stage A viewer. After Stage C, compare the final model against the same source images and the accepted Stage A model.

## Out of Scope

- Free per-vertex deformation
- Depth-Anything displacement
- Global identity refitting
- Mouth, jaw, chin, ear, or outer-contour optimization
- Medical-grade measurement claims
- Automatic promotion of a lower-loss candidate without visual approval
