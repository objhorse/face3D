# Controlled Low-Frequency Identity Deformation Design

## Goal

Recover face width, cheek, jaw, and chin identity cues that the current FLAME/MICA
identity space cannot express, while preserving the fixed topology and stable nose,
eye, and mouth geometry. In plain terms, this stage adds a small set of smooth face
shape controls instead of allowing every vertex to move independently.

The first experiment targets `captures_20260612_135253`. It builds on the calibrated
three-view cameras and the SDF silhouette evidence, but does not change camera pose,
expression, texture generation, or the selected FLAME identity coefficients.

## Diagnosed Limitation

The SDF experiment proved that the frontal silhouette contains a useful correction
signal: its final candidate reduced the frontal trusted-boundary error from about
32.55 px to 24.04 px without creating a topology failure. The candidate then
plateaued near a 26% improvement, while both profile views worsened slightly and the
FLAME coefficient update approached its drift limit.

This indicates that the observation direction is useful but the available FLAME
identity basis is too restrictive for the required face outline. Increasing the SDF
weight or coefficient limit would keep pushing inside the same limited basis and is
more likely to change the wrong facial regions than to recover the missing identity.

## Considered Approaches

### Semantic low-frequency deformation basis (selected)

Build a small deformation basis over semantic face regions and optimize only its
coefficients. Each coefficient moves a broad, smoothly weighted area of the mesh.
This provides enough freedom to recover the observed face shape while keeping the
problem low-dimensional and inspectable.

### Learned extended identity basis

Train or import a richer registered-face identity model such as a FaceScape-derived
basis. This is a strong future direction, but it requires compatible registered data,
model licensing review, and a separate training and validation pipeline.

### Free per-vertex residual

Allow direct vertex offsets with Laplacian regularization. This has the most freedom,
but previous experiments produced nose, philtrum, and mouth distortions. It is
explicitly rejected for this experiment.

## Architecture

Add `src/geometry/controlled_identity_deformation.py` as an isolated geometry unit.
It receives the low-resolution fitted mesh, fixed faces, calibrated view data,
interior landmark observations, and silhouette targets. It returns a candidate mesh,
the shared identity displacement field, optimized control coefficients, and a
structured audit report.

The module must not own camera fitting, expression fitting, subdivision, texture
baking, or GLB export. `module2_geometry.py` calls it after stable FLAME fitting and
before mesh subdivision. The same low-resolution identity displacement is applied by
vertex index to the neutral and final expression meshes before subdivision.

The existing SDF renderer and candidate metrics remain observation providers. The
old `ENABLE_FREE_IDENTITY_DEFORM`, personal residual, nose/mouth residual, and dense
nose residual paths are not enabled or called by this stage. Generic adjacency,
projection, and mesh-quality helpers may be reused only when their behavior is
covered by tests.

## Deformation Parameterization

Use 12 to 18 configurable degrees of freedom rather than one offset per vertex. The
initial basis covers:

- temple and upper-face width;
- upper-cheek and cheekbone width;
- mid-cheek width and anterior depth;
- jaw-angle and lower-jaw width;
- chin width, vertical length, and anterior depth.

Paired left and right controls have independent coefficients with a soft symmetry
penalty. This permits real asymmetry while preventing one noisy view from pulling a
single side out of proportion. Directions are defined in the canonical FLAME frame,
not in image coordinates, so the identity field is shared consistently by all views.

Each control uses graph-distance or geodesic Gaussian weights on the low-resolution
mesh. Paired weights and directions are mirrored from the fixed FLAME topology. The
eye sockets, nose, philtrum, and lips have protected cores where low-frequency basis
weights are zero. A smooth transition annulus avoids a hard seam at protection
boundaries. The ears and neck are outside the editable support.

The combined displacement is the weighted sum of all basis vectors. Coefficients are
bounded, and the final per-vertex displacement is clamped by a face-size-relative
limit rather than a dataset-specific distance in meters.

## Optimization Evidence

Freeze the calibrated cameras, per-view expression, FLAME identity coefficients,
and texture. Optimize only the low-frequency control coefficients using:

- frontal trusted-boundary signed-distance loss;
- subject-facing profile boundary loss in both side views;
- interior eye, nose, and mouth landmark anchor loss;
- differential-coordinate/Laplacian preservation from the fitted baseline;
- edge-length and local area distortion penalties;
- paired-control symmetry and coefficient magnitude priors.

The frontal SDF supplies the main correction direction. The two profile views are
active evidence and preservation constraints, not merely post-hoc checks. Hairline,
ears, neck, and view-dependent fixed contour landmark correspondences remain excluded.

Save the baseline, best valid checkpoint, and final optimizer state separately. Rank
checkpoints by the full acceptance score; never publish the final iteration merely
because it was last.

## Data Flow

1. Load the stable fitted low-resolution neutral and expression meshes.
2. Build or load the deterministic semantic deformation basis for the FLAME topology.
3. Render the current mesh through the fixed calibrated cameras.
4. Optimize only the low-frequency coefficients from multi-view SDF and anchor losses.
5. Evaluate every checkpoint with full-resolution observation and mesh-quality gates.
6. Apply the selected shared identity displacement to neutral and expression meshes.
7. Subdivide, export geometry-only artifacts, and generate matched-camera comparisons.
8. Bake texture only when the candidate passes all gates and the visual comparison is
   accepted.

If no checkpoint passes, the pipeline keeps the pre-deformation baseline and reports
the rejected candidate. Failure to improve is not a pipeline error.

## Quality And Acceptance Gates

A candidate is eligible for publication only when all of the following hold:

- frontal trusted-boundary mean error improves by at least 30% from the baseline;
- neither profile trusted-boundary mean error worsens by more than 10%;
- mean interior landmark error worsens by no more than 10%;
- protected nose, eye, philtrum, and mouth core displacement stays below its strict
  configured limit;
- maximum and mean editable-region displacement stay within face-size-relative limits;
- no new NaN, Inf, degenerate face, non-manifold edge, boundary edge, or normal flip
  appears;
- topology, face indices, vertex count, and UV correspondence remain unchanged;
- matched-camera geometry does not show an obvious identity regression.

The visual gate is mandatory because the current project has already shown that a
lower aggregate pixel score can produce a less recognizable person.

## Outputs And Reporting

Write experiment artifacts under
`output/experiments/<dataset>_controlled_identity_v1/` and include:

- `meshes/baseline_geometry.glb`;
- `meshes/candidate_geometry.glb`;
- `debug/controlled_identity_deformation/summary.json`;
- `debug/controlled_identity_deformation/coefficients.json`;
- `debug/controlled_identity_deformation/basis_support.png`;
- calibrated before/after silhouette and interior-landmark overlays for all views;
- geometry-only matched-camera renders for front, left profile, and right profile;
- textured output only for a candidate that passes the geometry gates.

The report must show baseline and candidate metrics side by side and state whether the
candidate was selected, rejected, or retained only for diagnosis.

## Configuration

The stage is disabled by default until the experiment passes. Configuration exposes:

- stage enable flag and deterministic seed;
- control coefficient and displacement limits;
- SDF, profile, landmark, smoothness, distortion, and symmetry weights;
- checkpoint interval and optimization step count;
- all numerical acceptance thresholds.

No configuration option may silently enable the legacy free-residual stages.

## Tests

Unit tests cover:

- deterministic basis construction and stable parameter ordering;
- mirrored support and direction for paired controls;
- zero support in protected cores and smooth support transitions;
- bounded coefficients and face-size-relative displacement clamps;
- deformation locality, fixed topology, and unchanged face indices;
- application of the same identity field to neutral and expression meshes;
- gradients from front and profile SDF evidence;
- candidate acceptance and fallback behavior;
- rejection of normal flips, degeneracy, non-manifold changes, and protected-region
  motion.

The experiment also runs the existing silhouette, mesh-quality, stable reconstruction,
expression, and texture regression tests.

## Experiment And Acceptance

Run a geometry-first A/B comparison on `captures_20260612_135253`:

1. the current stable baseline selected before low-frequency deformation;
2. the best gated controlled-deformation candidate with identical cameras and
   expression;
3. matched-camera source/model comparisons without texture;
4. texture baking only after the geometry candidate is accepted.

Success means the face width, cheek, jaw, and chin are visibly closer to the source,
the frontal boundary reaches the numerical target without sacrificing either profile,
and the stable central facial geometry remains intact. This is an identity-shape
experiment, not a medical measurement claim.

## Rollback

Commit `d547868` remains the pre-SDF stable reconstruction checkpoint. The current
SDF work remains an observation-layer experiment. If controlled deformation fails,
disable its stage and retain the selected stable baseline; do not restore any legacy
free-residual deformation path.

## Non-Goals

- No camera recalibration or pose refinement.
- No texture registration or seam correction.
- No local nose, eyelid, lip, or philtrum sculpting.
- No learned identity model training.
- No medical-grade dimensional accuracy claim.
