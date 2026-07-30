# SDF Silhouette Identity Recovery Design

## Goal

Correct the shared FLAME identity shape when the MICA initialization has the wrong
face width or jaw profile. The experiment uses the existing three calibrated views,
keeps topology fixed, and does not change texture generation.

## Diagnosed Failure

For `captures_20260612_135253`, the selected model is about 22% narrower between
the frontal jaw endpoints and its chin projects about 61 px above the observation.
The current joint fit excludes face-oval landmarks 0-16 because they are
view-dependent, while the existing silhouette loss has useful gradients mainly near
the target boundary. With a roughly 32 px boundary gap, all 22 shape checkpoints
were rejected and the narrow MICA identity was retained.

## Design

### Long-range boundary evidence

Extend `SilhouetteTarget` with a normalized signed-distance field derived from the
trusted semantic face boundary. The optimization loss samples this field with a
differentiable boundary extracted from the rendered mesh silhouette. A boundary far
inside the observed face therefore still receives an outward gradient instead of
waiting until it reaches a narrow target band.

Use the distance term together with the existing Dice/L1 coverage terms. The SDF
term drives long-range alignment; Dice/L1 prevents boundary collapse and preserves
the occupied face region.

### View semantics

- Front view: use both skin boundaries from below the unreliable hairline through
  the chin.
- Side views: use only the subject-facing profile boundary and lower jaw.
- Do not restore fixed correspondence loss for landmarks 0-16.
- Freeze camera pose and per-view expression during identity-shape refinement.
- Keep eye, nose, and mouth landmarks as interior anchors.

### Candidate selection

Save checkpoints during shape optimization and rank them using observation evidence,
not distance from the MICA prediction. A candidate must:

- materially reduce full-resolution trusted-boundary error;
- materially improve the front view while preserving both profile views;
- keep interior landmark error within the configured tolerance;
- pass finite, degeneracy, non-manifold, and fixed-topology quality gates.

MICA coefficient drift remains a runaway-deformation guard, not an identity-quality
score. Reports must state this distinction.

## Experiment Scope

Run one geometry-only ablation on `captures_20260612_135253`:

1. Current checkpoint `d547868` as baseline.
2. SDF silhouette candidate with identical cameras, expressions, and texture inputs.
3. Texture the accepted geometry only after the geometric comparison is generated.

The comparison page must show the source images, calibrated silhouette overlays,
interior landmark overlays, and matched-camera textured renders. A generic viewer
camera is supplementary and cannot decide acceptance.

## Acceptance

- Frontal trusted-boundary mean error improves by at least 30% from 32.55 px.
- The front view measurably improves; neither profile view worsens by more than 10%.
- Mean interior landmark error worsens by no more than 10%.
- Face count is unchanged and no degeneracy or non-manifold regression appears.
- The matched-camera model no longer presents the clearly narrow jaw seen in the
  baseline. Numerical gates cannot override an obvious identity regression.

## Tests

- Unit-test signed-distance construction, scale normalization, and far-boundary
  gradients.
- Unit-test that side-view reliability selects only the profile-facing boundary.
- Unit-test candidate selection and topology rejection.
- Run the existing reconstruction, identity-quality, expression, and texture tests.

## Rollback

The pre-experiment code is permanently recoverable at commit `d547868`. If the SDF
candidate fails acceptance, retain that commit's geometry and report the failed
ablation instead of silently publishing the candidate.
