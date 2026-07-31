# Multi-View Alar Surface Optimization Design

## Status

Approved for implementation planning on 2026-07-31.

## Problem

The accepted A2 geometry and v5 texture registration produce a stable face,
but the outer nasal wings still do not match the subject.

This is not primarily an optimizer-strength problem. The current nasal-base
model prevents the requested change by construction:

- `NasalBaseSemanticBasis` protects landmarks 31 and 35 and excludes outer
  alar width from its editable support.
- Its metadata declares the outer alar policy as `bit-exactly fixed`.
- The frontal objective uses the two outer alar anchors to remove local
  translation, rotation, and scale before measuring residuals.
- The outer anchors are therefore not shape residuals. A true alar-width
  error is normalized away instead of optimized.

The next stage must add a general, observation-driven outer-alar model rather
than increase the freedom of the compact nostril model or add per-subject
vertex corrections.

## Goal

Starting from the accepted A2 geometry, improve the subject's outer nasal-wing
width, flare, depth, and lower-rim contour using the three calibrated images.
The result must preserve fixed topology and leave the bridge, broad tip, upper
lip, eyes, cheeks, and the rest of the face unchanged.

After geometry acceptance, rebake only the nasal UV region with the accepted
pixel-locked texture pipeline.

## Non-Goals

- Do not refit global FLAME identity or expression.
- Do not use texture color or photometric residuals to move geometry.
- Do not enable free per-vertex residual deformation.
- Do not change camera calibration or camera pose.
- Do not alter the accepted v5 baseline files.
- Do not claim medical measurement accuracy.

## Baseline Contract

The experiment uses the accepted artifacts as immutable inputs:

- Geometry: A2 `face_mesh.obj`, SHA-256
  `50051dd20c973ed43f6511eca7b2a5ab4592438901e31da9a74a4ad3bc60201b`.
- Textured comparison baseline: v5 source v2 GLB, SHA-256
  `219fde410ddc99abf2367315ffe7407969f17b355e654b74f2a21f0e9fd3c7da`.
- Texture baseline: v2 `albedo_white.png`, SHA-256
  `b4402401937e5df8a5dc7fe1755be75c02482424ee807463b613485ca6f8e620`.
- Cameras: the hash-locked three-camera rig already recorded by the A2/v5
  reports.

Every experiment verifies these hashes before and after execution. It writes
to a new output directory and never replaces an accepted artifact.

## Architecture

### 1. Alar Observation Extraction

Add `src/geometry/alar_surface_observations.py`.

For each semantic view (`front`, `subject-left`, `subject-right`):

1. Read the existing preprocessed nose semantic mask.
2. Restrict it to a subject-relative nasal-wing region derived from detected
   landmarks and face scale.
3. Extract ordered left and right outer-alar boundary curves.
4. Build a signed Euclidean distance transform for each curve.
5. Estimate confidence from segmentation continuity, edge agreement, and
   visibility.

The front view contributes bilateral width and lower-wing curvature. Each
oblique view contributes the visible wing's depth, flare, and profile
curvature. Occluded or discontinuous samples receive zero weight instead of
being forced into the fit.

The optimization target is a curve or distance field, not a fixed landmark
index. This avoids the known mismatch between MediaPipe, FLAME, and visible
skin-boundary semantics.

### 2. Compact Outer-Alar Basis

Add `src/geometry/alar_surface_basis.py`.

The basis is rebuilt from the current subject mesh and uses mesh-geodesic
support around the outer alar landmarks. It exposes six semantic modes:

1. `alar_width_shared`
2. `alar_width_asymmetry`
3. `alar_flare_shared`
4. `alar_flare_asymmetry`
5. `alar_vertical_shared`
6. `alar_rim_curvature_shared`

The shared modes handle the dominant identity shape. Asymmetry is enabled only
for width and flare because those quantities have independent bilateral
evidence in all three views.

The basis has exact zero displacement outside its support. The inner nostril
rim and columella may move only through a smooth transition band so the outer
wing cannot detach from the accepted A2 nasal base. The bridge, broad tip,
philtrum, upper lip, cheeks, and every non-nasal vertex remain bit-exact.

This is a low-dimensional semantic model, not a collection of subject-specific
vertex offsets.

### 3. Multi-View Alar Objective

Add `src/geometry/alar_surface_optimizer.py`.

For a candidate coefficient vector:

1. Apply the six semantic basis fields to the immutable A2 vertices.
2. Project the candidate into all three fixed cameras.
3. Select visible semantic alar-surface samples.
4. Evaluate each projected sample against the corresponding observed signed
   distance field.
5. Add a reverse-coverage term from observed contour samples to the projected
   semantic curve.

The objective contains:

- Front left/right alar distance residuals.
- Subject-left visible alar profile residuals.
- Subject-right visible alar profile residuals.
- Weak coefficient prior.
- Local membrane/Laplacian regularity on the affected surface.
- Triangle orientation and minimum-area barrier.
- Weak bilateral symmetry prior scaled down when bilateral evidence supports
  real asymmetry.

No per-view similarity alignment is allowed inside the alar objective. Camera
coordinates and face scale are already fixed. Only a small robust common
2-D translation nuisance term, bounded to 2 px and shared by all nasal terms
within a view, may absorb segmentation quantization. It cannot change width,
depth, or scale.

Use bounded robust least squares. Fit in three stages:

1. `front`: width, width asymmetry, and vertical position.
2. `profile`: flare, flare asymmetry, and rim curvature.
3. `joint`: all six parameters with all views and regularization.

### 4. Candidate Validation

Add geometry validation before texture baking:

- Same vertex, face, UV-vertex, and UV-face counts as A2.
- No NaN or Inf.
- No new degenerate, flipped, or non-manifold faces.
- No vertex outside the semantic alar support moves.
- Bridge, tip, philtrum, and lip protected anchors are bit-exact.
- Front and both oblique alar distance scores must improve or remain within
  0.25 px of baseline.
- At least two of the three views must improve by 10 percent or more.
- The joint robust objective must improve by at least 10 percent.

Metric gates select invalid candidates; they do not choose visual identity.
The final decision remains an A/B review against the source photographs.

### 5. Texture Reuse

The geometry experiment first exports clay and untextured candidates.

For a geometry candidate that passes validation:

1. Run the accepted ordered nasal texture registration against the new
   geometry.
2. Recompute nasal UV ownership from the candidate surface.
3. Composite only owned nasal UV pixels onto the accepted v2 texture.
4. Preserve every non-owned texture byte exactly.
5. Export the final textured GLB without smoothing geometry.

This keeps geometry and texture diagnosis separate while producing a useful
visual comparison.

### 6. Experiment Runner and Reports

Add `run_multiview_alar_surface_experiment.py`.

Default output:

`output/experiments/captures_20260612_135253_alar_surface_v1`

Artifacts:

- `meshes/baseline_geometry.glb`
- `meshes/candidate_geometry.glb`
- `meshes/candidate_textured.glb`
- `textures/albedo_white.png`
- `debug/alar_observations/index.html`
- `debug/alar_projection/front.png`
- `debug/alar_projection/subject_left.png`
- `debug/alar_projection/subject_right.png`
- `alar_surface_report.json`
- `alar_surface_compare.html`

The viewer embeds both GLBs, supports unrestricted orbit and zoom, and labels
baseline and candidate explicitly.

## Error Handling

- Missing or hash-mismatched accepted inputs abort before creating a candidate.
- A missing view, invalid camera, or invalid semantic mask aborts observation
  construction.
- Low-confidence evidence disables the affected view term and is reported.
- Fewer than two usable views abort the geometry fit.
- Solver failure, bound saturation in more than three parameters, or geometry
  validation failure produces a diagnostic report but no success GLB.
- Texture failure does not delete a valid geometry candidate; it marks the
  textured stage as failed.

## Tests

### Unit Tests

- Ordered alar boundary extraction on synthetic masks.
- Correct subject-left/subject-right semantic mapping.
- Signed-distance sign and subpixel sampling.
- Each basis mode moves only its intended semantic region.
- Shared and asymmetric modes have the expected bilateral directions.
- Protected vertices remain bit-exact.
- Objective residuals decrease for known synthetic perturbations.
- Width errors cannot be removed by nuisance translation.
- Face-flip and minimum-area barriers detect unsafe candidates.

### Integration Tests

- Run geometry-only fitting on `captures_20260612_135253`.
- Verify all accepted source hashes remain unchanged.
- Verify fixed topology and protected-region invariants.
- Generate three calibrated projection diagnostics.
- Rebake nasal UV pixels and verify non-nasal texture bytes are unchanged.
- Load both final GLBs in the existing static viewer.

## Acceptance Criteria

The first implementation is accepted when:

- The outer nasal-wing contour is visibly closer to the subject in the front
  and at least one oblique view.
- The candidate does not make the bridge, tip, nostrils, mouth, or overall face
  less recognizable.
- No tearing, folding, holes, or new shading discontinuities appear.
- The geometry and UV topology remain fixed.
- Non-nasal vertices and non-nasal texture pixels remain unchanged.
- The result is produced from the same algorithm and subject-relative evidence,
  without constants tuned to one subject's pixel coordinates.
