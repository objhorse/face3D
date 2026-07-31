# Cross-View Texture-Guided Nasal Surface Design

## Status

Approved for detailed implementation planning on 2026-07-31.

## Problem

The committed multi-view alar experiment improves the outer nasal-wing
contour, but the lateral nasal-tip surface still differs from the subject.
The remaining mismatch is concentrated in the soft-triangle, alar-dome, and
alar-groove transition rather than the global nose width.

The v10 experiment exposes two separate limitations:

- The six semantic modes describe only shared/asymmetric width, flare,
  vertical motion, and one lower-rim curvature field. They cannot represent
  independent curvature changes along the tip-to-wing transition.
- The objective observes only the projected outer and lower alar curves. It
  contains no interior surface evidence that can determine whether a nasal
  side-wall point should move forward or backward.

Increasing the existing coefficient bound would amplify the same coarse
fields. Adding unconstrained vertices would introduce directions that the
current contour objective cannot observe. The next stage therefore adds
surface degrees of freedom and cross-view evidence together.

## Goal

Starting from the committed v10 candidate, improve the three-dimensional
shape of both nasal-tip side walls and their transition into the alae. Use
trusted cross-view skin appearance only as weak correspondence evidence, while
keeping semantic boundaries as the strong geometric evidence.

The method must remain subject-independent: controls are generated from mesh
topology and semantic nasal regions, not from hand-picked vertex indices or
pixel constants for this subject.

## Non-Goals

- Do not optimize geometry directly from image brightness or shading.
- Do not treat the baked GLB texture as an observation.
- Do not use monocular depth or Depth-Anything vertex displacement.
- Do not enable unrestricted per-vertex deformation.
- Do not change the fixed camera rig or independently refit camera poses.
- Do not change the bridge, nasal-tip center, columella, philtrum, lips, eyes,
  cheeks, or other facial regions.
- Do not claim medical or metric reconstruction accuracy.

## Immutable Baseline

The experiment starts from commit `8b2a796` and hash-locks these v10 outputs:

- Candidate OBJ: `053e4495c7c18240f27c44d67fa338ac2775c4a2a886156bc703a8f5fcea0376`
- Geometry GLB: `73d51fa7f8538ffbf2598f8d5e44794c36f285c1ed7da31908a7fe531ede5816`
- Textured GLB: `ee4d1812d36f20591a30064710ffd51e88f10382c48e5741fcb4ca529df13098`
- Texture: `dc6bb4cbf55ce6364a20879257d7eb5fc442ef5bab6bf4c647c24176e4f39894`
- Report: `9ee414b86d47ce8960671b1c4335e5d04d02dc8a15ed506e083d7dc7bf1f8527`

The runner verifies these hashes before and after execution and writes to a
new experiment directory. It never replaces v10 artifacts.

## Architecture

### 1. Fixed View Registration

The v10 solver allowed a two-pixel nuisance translation per view, and five of
six translation components reached their limits. Shape and view alignment
must not continue compensating for each other.

Before any surface optimization, estimate one fixed two-dimensional offset
for each view from protected nasal bridge and peri-nasal skin evidence. The
offset is reported, then frozen for every contour and texture term. Camera
intrinsics, rotation, translation, and scale remain unchanged.

If a stable offset cannot be estimated, the view keeps zero offset and its
appearance confidence is reduced. The surface solver receives no camera or
view-alignment variables.

### 2. Trusted Cross-View Appearance Observations

Add a nasal-region observation builder that operates on original preprocessed
camera images, not the current texture atlas.

Use the front-to-subject-left and front-to-subject-right pairs independently;
direct left-to-right matching is optional because their shared visible area is
small. Reuse the existing LoFTR, reciprocal matching, epipolar filtering, and
fixed-rig triangulation infrastructure where it is reliable.

Global LoFTR matches alone are known to be sparse on the nose. Supplement them
with model-guided local matching:

1. Select visible baseline mesh samples in the two nasal side-wall patches.
2. Project each sample through the fixed rig into a camera pair.
3. Search only a short segment along the corresponding epipolar line.
4. Compare normalized gradient/census-style patches rather than raw color.
5. Require reciprocal agreement, local patch distinctiveness, positive depth,
   sufficient ray angle, and low reprojection error.

Reject observations affected by:

- saturated specular highlights;
- nostril or deep shadow masks;
- occlusion or grazing view angles;
- weak local texture or repeated patterns;
- inconsistent front-left/front-right geometry;
- large epipolar or reprojection error.

Each accepted observation stores both image coordinates, triangulated 3-D
position, covariance/confidence, semantic region, and rejection provenance.
Low-confidence evidence is omitted rather than forced into the fit.

### 3. Hierarchical Nasal Side-Surface Basis

Retain the six v10 alar modes as the coarse level. Add a low-frequency residual
basis only on the bilateral tip-side/alar transition patches.

The residual basis is generated automatically:

1. Build a geodesic patch bounded by the protected tip center, bridge,
   columella, philtrum, and outer nasal support boundary.
2. Select three topology-relative control sites per side using geodesic
   farthest-point sampling, seeded by semantic soft-triangle, alar-dome, and
   alar-groove regions.
3. Construct smooth biharmonic influence fields with zero displacement on the
   patch boundary.
4. Give each site two meaningful directions: surface normal and
   subject-lateral/depth tangent.

This produces at most twelve residual coefficients, in addition to the six
coarse coefficients. It can change local bulge, groove depth, and the
tip-to-wing roll without exposing individual vertices.

The residual is low frequency by construction. Protected and out-of-support
vertices remain bit-exact. The optimizer may use a larger trust region than
v10, but displacement is limited by continuous surface strain, bending, and
orientation penalties rather than a single small coefficient ceiling.

### 4. Joint Objective

Optimize the coarse and residual coefficients against:

```text
E = E_contour
  + lambda_feature * E_cross_view_3d
  + lambda_reprojection * E_feature_reprojection
  + lambda_membrane * E_membrane
  + lambda_bending * E_bending
  + lambda_strain * E_local_strain
  + lambda_prior * E_coefficient_prior
  + E_orientation_barrier
```

- `E_contour` keeps the v10 front and profile alar boundaries as strong terms.
- `E_cross_view_3d` measures point-to-surface distance from trusted
  triangulated observations to the candidate nasal patches.
- `E_feature_reprojection` verifies that a fitted surface sample still
  projects to its matched pixels in both cameras.
- The regularizers preserve smooth curvature and prevent local stretching,
  folding, or detachment.

Appearance evidence is confidence-weighted and capped so it cannot overpower
the contour objective. No raw RGB residual or shading-derived normal term is
included.

Fit in four stages:

1. Freeze geometry and estimate fixed per-view offsets.
2. Re-evaluate the six coarse v10 parameters with the fixed offsets.
3. Enable the low-frequency residual basis using trusted 3-D observations.
4. Jointly refine all geometry coefficients with every regularizer active.

### 5. Validation and Selection

The candidate is invalid when any of these conditions fail:

- Baseline hashes changed.
- Vertex, face, UV-vertex, or UV-face counts changed.
- A protected or out-of-support vertex moved.
- A face flipped, became newly degenerate, or crossed the minimum orientation
  ratio.
- Local area or edge-length strain exceeds the configured continuous limit.
- Fewer than six trusted appearance observations survive on either side.
- Median held-out reprojection error or point-to-surface distance worsens.
- Any strong contour view regresses by more than 0.5 pixels.

Training observations and held-out observations are separated spatially so
that additional freedom is not accepted merely because it interpolates its
own control points.

Metrics reject unsafe or unsupported candidates; they do not decide identity.
The final decision remains a textured and clay A/B review against all three
source photographs.

### 6. Texture and Viewer

Geometry fitting uses original camera images and never samples the existing
texture atlas. For the first A/B experiment, apply the same accepted v5 texture
to baseline and candidate so the viewer isolates geometry.

Only after geometry acceptance, rerun the existing pixel-locked local nasal
texture registration on the accepted candidate. Preserve all non-nasal texture
pixels exactly.

The viewer must provide:

- synchronized baseline/candidate orbit and unrestricted zoom;
- textured and clay modes;
- fixed front, subject-left, and subject-right camera presets;
- optional trusted-observation markers;
- separate labels for geometry-only and texture-rebaked candidates.

## Components

- `src/geometry/nasal_texture_observations.py`: appearance masks, local
  epipolar matching, confidence, and triangulated records.
- `src/geometry/nasal_residual_basis.py`: bilateral geodesic patch and smooth
  low-frequency control fields.
- `src/geometry/texture_guided_nasal_optimizer.py`: staged joint objective.
- `run_texture_guided_nasal_surface_experiment.py`: hash-locked experiment,
  reports, exports, and A/B viewer.

Existing fixed-rig projection, LoFTR matching, reciprocal filters,
triangulation, mesh-quality checks, GLB export, and viewer helpers should be
reused rather than duplicated.

## Tests

### Unit Tests

- Model-guided matching remains on the epipolar line.
- Reciprocal and reprojection filters reject synthetic outliers.
- Highlight, shadow, nostril, and low-texture masks suppress observations.
- Triangulated observations have positive depth in both cameras.
- Residual basis has exact zero support outside its patch and at protected
  vertices.
- Each local mode is smooth and moves only its intended side.
- Synthetic known deformations reduce 3-D and reprojection residuals.
- Raw intensity changes alone cannot move geometry.
- Orientation, strain, and bending barriers reject unsafe candidates.

### Integration Tests

- Run the observation audit without changing v10 geometry.
- Report accepted/rejected matches per side and their spatial coverage.
- Run geometry optimization only when both sides have sufficient evidence.
- Verify v10 hashes remain unchanged.
- Verify fixed topology, support, protected vertices, and non-nasal texture.
- Generate calibrated clay and same-texture A/B comparisons.

## Acceptance Criteria

- Both nasal side-wall patches contain spatially distributed trusted
  observations, not one small match cluster.
- The candidate improves held-out cross-view 3-D/reprojection evidence.
- Front and profile outer-alar contours do not materially regress.
- The soft-triangle, alar-dome, and alar-groove transitions look closer to the
  subject in the textured and clay views.
- The tip center, bridge, nostril base, philtrum, lips, and whole-face identity
  remain unchanged.
- No tearing, folding, holes, or new shading discontinuities appear.
- The same pipeline can run on another three-view dataset without changing
  vertex IDs, pixel coordinates, or subject-specific constants.

## First Milestone

The first implementation milestone is observation-only. It generates the
trusted nasal correspondence/triangulation audit and does not deform the mesh.
Geometry work begins only after the audit demonstrates sufficient bilateral
coverage and acceptable reprojection quality. This prevents adding new freedom
before proving that the new evidence can constrain it.
