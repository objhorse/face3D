# Absolute Eyelid Geometry And Texture Registration Design

## Context

The current closed-eye result contains two visibly different eye lines:

- the geometric eyelid slit and crease produced by the mesh;
- the dark eyelash and closed-eye line baked from the source photograph.

The geometric line is above the photographed eye line, while the texture extends lower and farther toward the outer corner. This is not a lighting-only artifact. The Stage C projection overlays also show the geometric FLAME eye contour above the observed closed-eye contour.

Two implementation choices allowed the defect to survive:

1. Stage C compared each predicted and observed eye in separately normalized local frames. This removed absolute position, roll, and width from the objective.
2. Stable texture registration used eyes and mouth only for one bounded global face similarity transform. Its local non-rigid correction was nose-only.

Consequently, the Stage C score could improve while the eye remained absolutely displaced, and texture baking had no per-eye mechanism to place the eyelash line on the final mesh slit.

## Goal

Starting from the committed A2 baseline, align the actual mesh eyelid boundary with the observed eye line in all usable views, then rebake the texture against that accepted geometry with independent left-eye and right-eye registration.

The result must:

- remove the double-eye-line appearance;
- preserve the accepted nose, mouth, face contour, ears, UV layout, and camera rig;
- work for both open-eye and closed-eye captures without subject-specific vertex edits;
- keep geometry fitting and texture fitting separately measurable;
- never accept a candidate based only on a normalized local eye-shape score.

## Options Considered

### A. Absolute geometry alignment followed by geometry-conditioned eye rebake

This is the selected approach. It corrects the mesh observation model first and then corrects texture sampling against the final mesh.

Advantages:

- addresses both underlying causes;
- produces geometry that remains meaningful without texture;
- generalizes through semantic eye boundaries and state detection;
- preserves a clear geometry/appearance separation.

Cost:

- requires extracting and projecting the actual mesh eyelid boundary;
- requires a new per-eye texture warp and a fresh bake.

### B. Texture-only per-eye warp

Keep A2 geometry unchanged and warp each source eye texture onto the existing slit.

Advantages:

- faster;
- likely removes the most obvious double line.

Limitations:

- leaves the geometric eye line incorrect;
- can hide a model defect rather than fix it;
- is unsuitable as the primary route for future medical-beauty editing.

### C. Full FLAME expression refit

Re-optimize the global expression vector and cameras using the eye observations.

Advantages:

- stays within the parameterized face model.

Limitations:

- expression coefficients couple eyes, mouth, cheeks, and nose;
- previous experiments showed that a local mouth or eye correction can change unrelated facial regions;
- carries a larger regression risk than the selected semantic-eye model.

## Baseline And Artifact Contract

- Use the committed A2 geometry and its accepted texture as immutable baseline inputs.
- Do not use the uncommitted Stage C candidate as a new baseline.
- Record SHA-256 hashes for baseline OBJ, texture, camera metadata, and final baseline GLB.
- Run all experiments in a new output directory.
- Never overwrite `face.glb`, the A2 experiment, or previous viewers during development.
- Candidate geometry must preserve vertex count, face count, face indices, UV coordinates, and UV-face indices.
- Texture baking must not smooth geometry or delete faces.

## Observation Layer

### Actual mesh eyelid curves

The face mesh has no topological eye-hole boundary. The optimizer must therefore
represent the visible eye closure using ordered upper and lower semantic surface
curves embedded in the fixed mesh topology. The curves are initialized by the
FLAME eye landmarks, densified over the eyelid surface, and projected directly
from the current candidate mesh.

FLAME and MediaPipe landmarks remain useful for:

- identifying inner and outer corners;
- choosing the eye semantic region;
- initializing correspondence direction;
- detecting open-eye or closed-eye state.

They are not the final geometric boundary used for alignment.

### Observed eye line

For each view and each eye:

1. detect eye state using MediaPipe dense landmarks;
2. initialize an ordered eyelid curve from dense eye landmarks;
3. refine the curve toward the strongest plausible eyelash or lid boundary within a narrow semantic search band;
4. reject hair, eyebrow, specular, and shadow edges using orientation, distance, and semantic-region constraints;
5. store confidence per curve sample.

For a closed eye, the primary observed target is the single closed-eye line. For an open eye, upper and lower lid boundaries remain separate.

Views with low confidence contribute only weak landmark anchors and cannot drive local deformation.

## Geometry Model

Each eye uses a shared low-dimensional 3D semantic deformation model. It provides:

- eye-frame vertical translation;
- eye-frame roll adjustment;
- eye width adjustment around the eye center;
- upper-lid and lower-lid aperture;
- upper-lid and lower-lid curvature;
- controlled surface-normal bulge for the eyelid transition.

The controls deform the eye core and decay smoothly through an orbital transition band. All vertices outside that support remain exactly equal to the baseline.

Inner and outer corners are soft anchors rather than frozen points. Their movement is allowed when supported by absolute multiview evidence, but is regularized and bounded. This avoids the previous failure where fixed corners prevented the whole slit from reaching the observed line.

The same 3D controls are projected into all three fixed calibrated cameras. Camera parameters are not optimized in this experiment.

## Geometry Objective

For each view, a single stable face frame is estimated from projected and
observed mid-face anchors outside the eyes. The transform is frozen before eye
optimization. All eye residuals are then evaluated in that shared frame. This
retains absolute eye position relative to the face while remaining insensitive
to a translation of the whole image.

The primary objective uses image-space quantities that retain absolute
eye-to-face position:

- symmetric distance between the projected mesh boundary and observed eye curve;
- corner position residual;
- eye-center vertical and horizontal residual;
- eye-line roll residual;
- eye width residual;
- open-eye upper/lower aperture residual or closed-eye single-line residual.

Auxiliary terms include:

- local normalized curve-shape residual;
- multiview consistency;
- left/right soft symmetry without forcing identical eyes;
- Laplacian or ARAP-like transition smoothness;
- displacement magnitude regularization;
- normal-flip, collapsed-face, and degeneracy penalties.

The normalized local curve score is diagnostic only. It cannot independently select or accept a candidate.

## Geometry Quality Selection

Candidate selection uses a Pareto-style rule instead of a single weighted scalar:

1. absolute eye-line alignment must improve in the front view;
2. the visible side-view alignment must not materially worsen;
3. eye width, center, and roll must remain within observation confidence;
4. non-eye geometry must remain byte-for-byte unchanged;
5. topology and local normal checks must pass;
6. the user-facing pure-geometry comparison must not show pinching, a double slit, or unnatural eyelid bulges.

If these conditions conflict, the experiment reports the conflict and retains A2. It does not silently fall back while labeling the candidate successful.

## Per-Eye Texture Registration

Texture registration runs only after geometry selection.

For each view:

1. project the final actual mesh eyelid boundary;
2. pair it with the observed eye curve using ordered arc-length correspondence;
3. build independent left-eye and right-eye local sampling fields;
4. add zero-displacement anchors around the orbital boundary so the warp decays before the eyebrow, nose, and cheek;
5. compose eye fields with the existing bounded global transform and nasal field;
6. validate Jacobian and displacement limits;
7. rebake the entire texture from the final candidate geometry.

The eye warp changes source sampling coordinates, not mesh vertices or UV topology. It must preserve eyelashes and eyelid coloration while placing the photographed eye line on the geometric slit.

If a per-eye warp fails its Jacobian or confidence checks, that view-eye contribution is reduced or excluded. The system must not replace it with an unconstrained broad facial warp.

## Outputs

The new experiment directory will contain:

- immutable A2 baseline OBJ and GLB references;
- pure-geometry eyelid candidate OBJ and GLB;
- freshly textured candidate GLB;
- absolute projection overlays for all views;
- eye crops showing source, geometry-only projection, baseline texture, and candidate texture;
- per-eye displacement-field visualizations;
- `eye_observations.json`;
- `geometry_metrics.json`;
- `texture_registration_metrics.json`;
- an offline A/B viewer with identical camera, lighting, controls, and framing.

The viewer must allow unrestricted orbit and zoom and must not change framing when switching models.

## Tests

### Unit tests

- ordered actual eyelid boundary extraction from fixed topology;
- closed/open eye observation extraction and confidence;
- absolute center, roll, width, and curve residuals;
- proof that translating or rotating the entire eye changes the primary score;
- proof that the old normalized local metric alone cannot pass acceptance;
- deformation support is zero outside the eye region;
- per-eye texture warp decays to zero outside the orbital mask;
- Jacobian and displacement validation.

### Regression tests

- A2 source artifacts remain unchanged;
- nose, mouth, face contour, ears, topology, and UVs remain unchanged after eye geometry optimization;
- texture export does not alter geometry;
- baseline and candidate viewer sides load the intended distinct artifacts;
- both `captures_20260612_135253` and `captures_20260612_140210` complete.

## Acceptance Criteria

- No visible second eyelash or closed-eye line above or below the geometric slit.
- The projected semantic mesh eyelid curves overlap the observed eye line in the shared face image frame.
- Front-view eye center, roll, and width improve rather than only normalized local curve shape.
- Pure geometry remains natural from front and side views.
- Nose, mouth, face contour, ears, and accepted A2 texture outside the eye registration support do not regress.
- Candidate texture is freshly baked from candidate geometry.
- Both test datasets demonstrate the same behavior without subject-specific vertex indices or manual per-subject parameters.

## Non-Goals

- changing the accepted nasal geometry;
- optimizing camera calibration;
- adding free per-vertex eye residuals;
- using a broad whole-face optical-flow warp;
- claiming medical-grade eyelid measurement accuracy.
