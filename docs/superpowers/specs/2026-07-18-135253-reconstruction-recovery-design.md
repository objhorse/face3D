# 135253 Reconstruction Recovery Design

## Goal

Make the three-view stable pipeline produce a recognizable, closed-eye,
closed-mouth reconstruction for `captures_20260612_135253` without hiding
failures behind topology-only quality gates.

## Confirmed Failures

1. The final geometry has open eyelids and an open mouth even though all three
   source images show closed eyes and closed lips.
2. `face_stable_neutral.glb` is copied from the expression mesh and is therefore
   byte-for-byte equivalent in vertex positions instead of being neutral.
3. Texture colors are inpainted after the observation alpha is computed. The
   inpainted feature pixels remain transparent in the GLB, which creates black
   holes around the eyelids, nostrils, lips, and chin.
4. Candidate selection treats distance from MICA as identity truth. On the
   failing sample, the strongest MICA anchor has worse photo observation error
   than weaker candidates but is selected because it alone passes the MICA
   drift thresholds.
5. Side-view fixed contour landmarks are correctly excluded, but the current
   silhouette target compares a face-skin mask with a mesh that includes scalp,
   ears, and neck. This makes the silhouette score unsuitable as a hard gate.

## Design

### Expression Fidelity

- Add dedicated eye-gap and inner-lip-gap losses derived from the observed 68
  landmarks so closed and open states are preserved without a dataset-specific
  global switch.
- Keep shared expression for synchronized captures, but record per-feature gap
  diagnostics and reject clearly inconsistent final expression geometry.
- Export the expression mesh from the optimized expression parameters and a
  real neutral mesh from the same shape parameters with zero expression.

### Texture Visibility

- Preserve the original observed-alpha mask as confidence evidence.
- After inpainting, make all valid UV pixels opaque except explicitly hidden
  lower geometry. Inpainted pixels remain low confidence but are visible.
- Add appearance gates for feature-region transparency and atlas near-black
  content. Topology passing alone must not mark the textured result successful.

### Identity Selection

- Rename the existing identity check conceptually to MICA drift: it is a prior,
  not ground truth.
- Reject only runaway deformation with relaxed absolute drift limits.
- Among mesh-safe candidates, choose using independent photo evidence first and
  use MICA drift as a soft tie-breaker.
- Restore frontal contour evidence at low weight while continuing to exclude
  fixed side-view contour correspondences.
- Do not use the current full-head silhouette score as a hard acceptance gate
  until a face-only render mask excludes scalp, ears, and neck.

## Verification

- Unit tests prove neutral and expression meshes differ when expression is
  nonzero, inpainted valid UV pixels are opaque, observed confidence remains
  available, and MICA candidate selection does not prefer worse photo fit only
  because it is closer to MICA.
- Existing mesh, texture-registration, identity, and silhouette tests remain
  green.
- A full rerun on `captures_20260612_135253` must show closed eyes and lips in
  clay geometry before texture is judged.
- The textured GLB must not contain black transparent holes in the eyes, nose,
  mouth, or chin. The quality report must expose expression, appearance, and
  MICA-drift results separately.

