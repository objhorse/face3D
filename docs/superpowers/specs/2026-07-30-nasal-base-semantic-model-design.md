# Nasal Base Semantic Model Design

## Goal

Add an A2 stage between balanced nasal fitting and eyelid Stage C. A2 improves the geometry of the nasal base without reopening the rest of the face.

The pure-geometry review of `balanced_nasal_v4` shows that the remaining mismatch is geometric:

- the columella is insufficiently defined;
- the two nostril apertures have the wrong relative opening;
- the inner alar rim is too flat;
- the current eight broad nasal modes cannot control these structures independently.

Texture rebaking alone cannot correct these errors.

## Immutable Input

A2 starts from the accepted direction represented by `balanced_nasal_v4`:

- Dataset: `captures_20260612_135253`
- Geometry: `output/experiments/captures_20260612_135253_balanced_nasal_v4/meshes/candidate.glb`
- Geometry SHA-256: `B3CA8484D33A3C453FD975DE1B09ADAD6B782E6686BD965DDB78DA58AD95F6F6`
- Textured model: `output/experiments/captures_20260612_135253_balanced_nasal_v4/meshes/candidate_textured.glb`
- Textured SHA-256: `BC610FC635BBCE9F12D90C87F60C5A5D97D04217016E29EDD6FF13B24D1ADE69`

The A2 runner verifies these hashes before optimization. It must not overwrite the v4 experiment.

## Scope

A2 changes only the nasal-base support:

- columella;
- left and right nostril aperture boundaries;
- left and right inner alar rims;
- the immediate transition between nostril rim and nose tip.

The following regions remain fixed:

- nose bridge and nasal root;
- broad nose-tip position and overall tip depth established by v4;
- outer alar width established by v4;
- philtrum and mouth;
- cheeks, eyes, chin, jaw, ears, and outer face contour.

## Semantic Parameterization

A2 adds eight local semantic parameters:

1. `columella_vertical`
2. `columella_depth`
3. `nostril_width_shared`
4. `nostril_width_asymmetry`
5. `nostril_height_shared`
6. `nostril_height_asymmetry`
7. `alar_rim_curvature_shared`
8. `alar_rim_curvature_asymmetry`

These modes are constructed in a local anatomical frame derived from the v4 nose tip, alar seeds, and front-camera orientation.

Every mode has compact geodesic support and a smooth falloff. Mode amplitudes use a fixed fraction of face width so the same parameterization can be used on another subject.

No generic FLAME identity directions and no free per-vertex residuals participate in A2.

## Observation Extraction

### Nasal Aperture Evidence

For each view, A2 projects the v4 nasal-base support into the calibrated image and creates a constrained search region. Within that region it extracts candidate aperture curves from:

- local dark-region likelihood;
- image-gradient ridges;
- FaceMesh nose and alar anchors;
- continuity of the inner alar boundary;
- agreement with the projected v4 geometry.

The result is an ordered curve with per-sample confidence, not a binary mask.

Dark pixels alone are not treated as nostril evidence because shadows, facial hair, and texture can produce similar intensity.

### Columella Evidence

The front view provides the lower columella point and the separation between the two nostril apertures. Oblique views provide columella depth and the transition into each alar rim.

### Confidence

Confidence decreases when:

- gradient and dark-region evidence disagree;
- the curve leaves the geometry-guided search region;
- the curve is discontinuous;
- left-right evidence is inconsistent without supporting image asymmetry;
- a view observes the structure at a grazing angle.

Low-confidence samples remain reported but contribute less to the objective.

## Optimization

### Stage A2.1: Frontal Nasal Base

The front view optimizes:

- `columella_vertical`;
- `nostril_width_shared`;
- `nostril_width_asymmetry`;
- `nostril_height_shared`;
- `nostril_height_asymmetry`.

All other A2 parameters are frozen.

### Stage A2.2: Oblique Depth And Rim Curvature

The two oblique views optimize:

- `columella_depth`;
- `alar_rim_curvature_shared`;
- `alar_rim_curvature_asymmetry`.

The frontal result remains fixed during this stage.

### Stage A2.3: Balanced Joint Refinement

All eight A2 parameters are refined together. Observation groups are normalized by effective confidence and calibrated at the immutable v4 input so that front and oblique evidence have comparable initial robust cost.

The objective includes continuous terms for:

- aperture-curve reprojection;
- columella reprojection;
- local Laplacian smoothness;
- preservation of the outer alar boundary;
- preservation of the v4 nose-tip centroid and depth;
- left-right symmetry weighted by bilateral evidence;
- triangle-orientation and positive-depth feasibility.

These are objective terms, not post-hoc shape patches.

## Topology Limitation

A2 uses the existing fixed topology. It can reshape the nasal sill and rim but cannot create a true nostril cavity if the template has no vertices capable of representing it.

The report therefore measures whether the semantic modes saturate at their bounds while aperture residual remains high. Saturation is reported as a representation limitation. It does not trigger automatic free deformation.

If this limitation is reached, the next architectural option is a reusable fixed-topology high-detail nasal patch for all subjects. That option is outside A2 and requires separate approval.

## Outputs

A2 writes:

- `output/experiments/captures_20260612_135253_nasal_base_v1/meshes/baseline.glb`
- `output/experiments/captures_20260612_135253_nasal_base_v1/meshes/candidate.glb`
- `output/experiments/captures_20260612_135253_nasal_base_v1/meshes/candidate_textured.glb`
- `output/experiments/captures_20260612_135253_nasal_base_v1/nasal_base_compare.html`
- `output/experiments/captures_20260612_135253_nasal_base_v1/fit_report.json`
- `output/experiments/captures_20260612_135253_nasal_base_v1/debug/nasal_base_observations/index.html`

The comparison viewer uses the same baseline-derived framing, camera, lighting, material controls, and synchronized interaction as v4.

## Reporting

The report records:

- full input hashes;
- extracted aperture and columella curves;
- confidence by sample and view;
- raw and normalized cost by evidence group;
- all eight coefficients by stage;
- parameter-bound saturation;
- displacement inside and outside nasal-base support;
- mesh-quality comparison;
- geometry and textured output hashes.

## Tests

### Unit Tests

- A2 exposes exactly eight canonical nasal-base parameters.
- Every mode has compact support and zero displacement outside the nasal base.
- Shared and asymmetric modes follow subject-relative left/right semantics.
- Parameter ownership freezes non-stage coefficients exactly.
- View balancing equalizes initial front and oblique robust costs.
- Baseline hash mismatch stops before observation extraction.
- Low-confidence aperture samples have proportionally lower influence.

### Regression Tests

- Nose bridge and root vertices remain unchanged.
- Outer alar width remains unchanged within numerical tolerance.
- V4 nose-tip centroid and depth remain unchanged within tolerance.
- Mouth, philtrum, cheeks, and all non-nasal vertices remain unchanged.
- No new non-finite, degenerate, or non-manifold geometry appears.
- Geometry-only and textured GLBs use identical candidate vertex positions.
- Existing balanced-v4 and legacy-v3 tests remain green.

## Acceptance

A2 is accepted only after the user reviews geometry-only and textured models from front, below-front, subject-left, and subject-right angles.

Required visual improvements:

- a clearer central columella;
- more plausible bilateral nostril apertures;
- smoother and more anatomical inner alar rims;
- no regression in the v4 nose-tip and outer alar shape;
- no change to the rest of the face.

Stage C starts only after A2 is accepted.
