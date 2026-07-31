# Cross-View Texture-Guided Nasal Surface Implementation Plan

> Design source: `docs/superpowers/specs/2026-07-31-cross-view-texture-guided-nasal-surface-design.md`

## Objective

Use trusted front-to-side appearance correspondences to constrain the
three-dimensional nasal side-wall surface, then fit a smooth low-frequency
residual basis on top of the committed v10 alar geometry.

The work is split into two releases:

- **Release A: observation audit only.** It must prove bilateral cross-view
  coverage without modifying geometry.
- **Release B: geometry optimization.** It is enabled only after Release A
  passes its evidence gate.

All work stays on `codex/restore-7-12-full`. Do not stage or modify the
unrelated semantic-eyelid files already present in the worktree.

## Baseline Contract

- Commit: `8b2a796`
- v10 candidate OBJ SHA-256:
  `053e4495c7c18240f27c44d67fa338ac2775c4a2a886156bc703a8f5fcea0376`
- v10 geometry GLB SHA-256:
  `73d51fa7f8538ffbf2598f8d5e44794c36f285c1ed7da31908a7fe531ede5816`
- v10 textured GLB SHA-256:
  `ee4d1812d36f20591a30064710ffd51e88f10382c48e5741fcb4ca529df13098`
- v10 texture SHA-256:
  `dc6bb4cbf55ce6364a20879257d7eb5fc442ef5bab6bf4c647c24176e4f39894`

The experiment must verify all hashes before and after every real-data run.

## Release A: Observation Audit

### Task 1: Define Nasal Appearance Observation Contracts

**Files**

- Add `src/geometry/nasal_texture_observations.py`
- Add `tests/test_nasal_texture_observations.py`

**Steps**

1. Add failing tests for immutable records:
   - `NasalTextureObservationConfig`
   - `NasalPairMatch`
   - `TrustedNasalObservation`
   - `NasalTextureObservationBundle`
2. Require explicit coordinate provenance for every pixel:
   - semantic view name;
   - source image size;
   - work image size;
   - source-to-work transform;
   - distortion/undistortion state.
3. Require each trusted observation to store:
   - pixels in both views;
   - triangulated reference-camera point;
   - confidence and covariance proxy;
   - semantic region;
   - ray angle and reprojection errors;
   - source matcher and rejection provenance.
4. Implement validation and read-only NumPy storage.
5. Run:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_nasal_texture_observations.py -q
```

**Done when**

- Invalid coordinates, view names, confidence, depth, or provenance fail
  explicitly.
- Records cannot be mutated after construction.

### Task 2: Build Trusted Nasal Image Masks

**Files**

- Modify `src/geometry/nasal_texture_observations.py`
- Modify `tests/test_nasal_texture_observations.py`

**Steps**

1. Add synthetic-image tests that separately identify:
   - valid nasal skin;
   - saturated highlight;
   - nostril/deep shadow;
   - low-gradient skin;
   - face-parser leakage outside the nose.
2. Reuse semantic and face masks from `nasal_observations.py` and
   `cross_view_surface_observations.py`.
3. Build separate maps rather than one opaque binary mask:
   - `semantic_support`;
   - `specular_reject`;
   - `shadow_reject`;
   - `texture_strength`;
   - `final_confidence`.
4. Use subject-relative nasal regions derived from the projected v10 mesh.
   Do not introduce subject-specific pixel boxes.
5. Save every intermediate map in the later report.
6. Run the focused tests.

**Done when**

- Lighting-only highlights and nostril shadows cannot become geometry
  observations.
- The mask retains visible skin on both tip-side/alar transition patches.

### Task 3: Estimate and Freeze Per-View Image Offsets

**Files**

- Add `src/geometry/nasal_view_registration.py`
- Add `tests/test_nasal_view_registration.py`

**Steps**

1. Add failing synthetic tests with known subpixel offsets and outliers.
2. Reuse protected bridge/peri-nasal surface samples from the fixed v10 mesh.
3. Associate only high-confidence cross-view observations in protected regions
   with their closest compatible baseline surface points.
4. Solve robust per-view two-dimensional offsets with a zero-mean gauge and a
   weak zero prior.
5. Reject scale, rotation, shear, and camera-pose variables.
6. Report confidence intervals, support counts, and residual distributions.
7. Freeze accepted offsets before any alar or residual optimization.
8. Run:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_nasal_view_registration.py -q
```

**Done when**

- Synthetic offsets are recovered within 0.25 px.
- Shape coefficients cannot alter the registered view offsets.
- Low-support views fall back to zero offset with reduced confidence.

### Task 4: Add Model-Guided Epipolar Local Matching

**Files**

- Modify `src/geometry/nasal_texture_observations.py`
- Modify `tests/test_nasal_texture_observations.py`
- Reuse `src/learned_cross_view_geometry.py`
- Reuse `src/geometry/cross_view_surface_observations.py`
- Reuse `src/geometry/profile_triangulation.py`

**Steps**

1. Add synthetic stereo tests for a known slanted textured surface.
2. Use LoFTR matches inside trusted nasal masks as seed observations.
3. Sample visible baseline vertices in each tip-side/alar patch.
4. For each front-view sample, search a short fixed-rig epipolar segment in
   the corresponding side image.
5. Score patches with illumination-resistant descriptors:
   - zero-mean normalized gradients;
   - census-style local ordering;
   - optional LoFTR confidence when a seed is nearby.
6. Require reciprocal matching and a unique best match separated from the
   second-best candidate.
7. Triangulate with `triangulate_profile_point` and retain only observations
   with positive depth, sufficient ray angle, and bounded reprojection error.
8. Preserve every rejection reason for the audit report.
9. Run focused tests plus:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_profile_triangulation.py tests\test_semantic_epipolar_refinement.py -q
```

**Done when**

- Matching cannot leave the epipolar search interval.
- Uniform, specular, shadowed, ambiguous, or occluded patches are rejected.
- Known synthetic 3-D points are recovered without using raw brightness as a
  shape residual.

### Task 5: Build the Observation-Only Audit Runner

**Files**

- Add `run_nasal_texture_observation_audit.py`
- Add `src/reports/nasal_texture_observation_report.py`
- Add `tests/test_run_nasal_texture_observation_audit.py`

**Steps**

1. Add runner tests for hash mismatch, missing views, insufficient evidence,
   successful report generation, and unchanged geometry.
2. Hash-lock v10 and the fixed rig before loading learned matchers.
3. Run Tasks 2-4 on `captures_20260612_135253`.
4. Generate:
   - raw/accepted/rejected match overlays for each front-side pair;
   - epipolar search diagnostics;
   - a triangulated nasal point cloud;
   - model-to-observation distance maps;
   - per-region spatial coverage heatmaps;
   - fixed-offset diagnostics;
   - JSON metrics and a static HTML report.
5. Use an explicit Release A gate:
   - at least six trusted observations per side;
   - observations cover at least two of the three semantic subregions per side;
   - median reprojection error at most 1.5 px;
   - 90th percentile reprojection error at most 2.5 px;
   - no concentrated cluster contains more than 60% of one side's evidence.
6. If the gate fails, report `insufficient_texture_evidence` and stop without
   creating a deformed mesh.
7. Run focused and related regression tests.

**Release A review**

Open the audit report and verify that accepted points lie on actual shared skin
features, not highlights, nostril shadows, eyelashes, or texture seams. Release
B starts only after this visual review is accepted.

## Release B: Geometry Optimization

### Task 6: Generate the Low-Frequency Nasal Residual Basis

**Files**

- Add `src/geometry/nasal_residual_basis.py`
- Add `tests/test_nasal_residual_basis.py`

**Steps**

1. Add tests for bilateral semantic patch construction and exact protection.
2. Build the patch from the v10 semantic frame and mesh adjacency.
3. Select three control sites per side by semantic seed plus geodesic
   farthest-point sampling.
4. Construct biharmonic influence fields with zero boundary values.
5. Expose normal and lateral/depth-tangent displacement at each control site.
6. Orthonormalize nearly dependent fields and discard numerically weak modes.
7. Verify:
   - no support outside the intended patch;
   - protected vertices are bit-exact;
   - fields are smooth and topology-relative;
   - total residual mode count is no greater than twelve.
8. Run focused tests.

**Done when**

- The basis can independently represent side-wall bulge, groove depth, and
  tip-to-wing roll without moving the tip center or outer face.

### Task 7: Implement the Texture-Guided Joint Objective

**Files**

- Add `src/geometry/texture_guided_nasal_optimizer.py`
- Add `tests/test_texture_guided_nasal_optimizer.py`
- Reuse `src/geometry/alar_surface_optimizer.py`

**Steps**

1. Add synthetic failing tests for known coarse and local deformations.
2. Optimize the six coarse v10 modes plus accepted residual modes.
3. Remove all per-view translation variables; use frozen Task 3 offsets.
4. Add confidence-weighted terms:
   - strong alar contour distance;
   - triangulated point-to-surface distance;
   - matched-pixel reprojection;
   - membrane, bending, and local strain;
   - coefficient prior and orientation barrier.
5. Use robust losses and held-out observations.
6. Replace the single v10 coefficient bound with:
   - broad numerical trust bounds;
   - continuous strain/orientation barriers;
   - line-search rejection of unsafe steps.
7. Ensure appearance evidence is capped below the strong contour contribution.
8. Run focused tests plus existing alar tests.

**Done when**

- Synthetic deformations are recovered.
- Raw intensity changes with fixed correspondences produce no shape change.
- Unsafe large moves fail through geometry energy, not subject-specific gates.

### Task 8: Add the Full Experiment Runner and A/B Viewer

**Files**

- Add `run_texture_guided_nasal_surface_experiment.py`
- Add `tests/test_run_texture_guided_nasal_surface_experiment.py`
- Reuse existing GLB, report, and static viewer helpers.

**Steps**

1. Hash-lock v10 and load the accepted Release A observation bundle.
2. Run the four optimization stages from the design.
3. Validate topology, protected support, strain, orientation, held-out
   reprojection, and strong contour metrics.
4. Export baseline and candidate geometry GLBs.
5. Apply the exact same v5 texture to both GLBs for the first comparison.
6. Generate synchronized textured/clay A/B panels with calibrated view presets
   and unrestricted zoom.
7. Do not rebake texture in this task.
8. Run focused and related regression tests.

**Done when**

- The viewer isolates geometry changes.
- No geometry outside the intended nasal side patches changes.
- Candidate status distinguishes `passed`, `review_required`, and
  `insufficient_texture_evidence`.

### Task 9: Real-Data Acceptance and Generalization

**Files**

- Update experiment reports only if required by discovered evidence.
- Do not change model constants per dataset.

**Steps**

1. Run Release A and B on `captures_20260612_135253`.
2. Compare textured and clay front/left/right renders against source photos.
3. Run the same unchanged pipeline on `captures_20260612_140210`.
4. Verify both datasets use identical configuration and semantic logic.
5. If the first dataset passes but the second lacks evidence, report that lack
   honestly; do not loosen thresholds or add pixel constants.
6. After geometry is visually accepted, run the existing pixel-locked local
   nasal texture registration and export a separate rebaked candidate.
7. Run the full relevant test set and `git diff --check`.

## Verification Commands

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_nasal_texture_observations.py tests\test_nasal_view_registration.py tests\test_profile_triangulation.py tests\test_semantic_epipolar_refinement.py -q

D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_nasal_residual_basis.py tests\test_texture_guided_nasal_optimizer.py tests\test_alar_surface_optimization.py -q

D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_run_nasal_texture_observation_audit.py tests\test_run_texture_guided_nasal_surface_experiment.py -q
```

When sandbox ACLs block pytest temporary directories, rerun the identical test
command outside the sandbox with an explicit unique `--basetemp`; do not treat
an ACL setup error as a code failure.

## Stop Conditions

Stop and return to evidence diagnosis instead of adding geometry freedom when:

- either nasal side lacks six trusted observations;
- accepted matches cluster in one highlight or nostril-shadow region;
- view offsets cannot be estimated independently of shape;
- held-out evidence worsens while training evidence improves;
- three or more implementation hypotheses fail in different components.

These conditions indicate insufficient observations or a wrong correspondence
model, not a need for more deformation parameters.
