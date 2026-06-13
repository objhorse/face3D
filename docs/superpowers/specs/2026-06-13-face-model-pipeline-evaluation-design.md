# 3D Face Pipeline Evaluation Design

## Objective

Run the complete reconstruction pipeline on the configured three-view capture set
`captures_20260519_102448`, independently inspect the resulting face model, and
produce a self-contained HTML report that explains whether the result is excellent.

## Scope

- Use the current working tree and current configuration without altering algorithm
  parameters solely to improve the reported result.
- Run geometry reconstruction through `run_phase1.py` and texture generation and GLB
  packaging through `run_phase3.py`.
- Preserve the repository's existing uncommitted algorithm changes.
- Store evaluation outputs in a dedicated directory under `output/evaluation/`.
- Evaluate the current subject only. Synthetic ground-truth benchmarking and
  cross-subject comparisons are outside this run.

## Evidence Collected

### Pipeline execution

Record the interpreter, device, start and end times, stage status, warnings, errors,
and final artifact paths. A failed or incomplete pipeline cannot receive an excellent
rating.

### Geometry and projection

Measure mesh vertex and face counts, invalid or non-finite values, degenerate faces,
connected components, boundary and non-manifold edges where available, bounding-box
dimensions, and landmark or contour reprojection errors exposed by pipeline artifacts.
Render textured and neutral-material views from the front, both 45-degree angles, and
both sides so geometry can be judged without relying on texture alone.

### Texture

Measure texture dimensions, valid coverage, near-black and near-white pixel ratios,
large suspicious regions, and available view-assignment or texture-audit statistics.
Inspect the forehead, eyes, nostrils, mouth, cheeks, jaw, ears, and side transitions for
holes, ghosting, stretching, seams, color discontinuities, and inpainting artifacts.

### Source alignment

Show the three source photographs beside corresponding reconstruction views and include
debug landmark, mask, contour, and projection images when generated. The report must
distinguish directly measured evidence from visual judgment.

## Rating Model

The report uses a 100-point score:

| Dimension | Weight |
| --- | ---: |
| Geometry and identity resemblance | 30 |
| Multi-view projection and contour fit | 25 |
| Texture clarity and continuity | 25 |
| Mesh and artifact integrity | 10 |
| Pipeline completeness and reproducibility | 10 |

Final labels are:

- **Excellent**: score at least 85, the full pipeline succeeds, and no critical defect
  is found.
- **Good**: score from 75 to 84 with no critical defect.
- **Acceptable**: score from 60 to 74, or a notable defect limits practical use.
- **Needs improvement**: score below 60 or any critical defect is present.

Critical defects include an unusable or missing GLB, severe facial collapse or
asymmetry, large black texture holes on central facial regions, gross source-view
misalignment, invalid mesh data, or a pipeline stage that does not complete. The final
label is therefore not determined by the weighted score alone.

## HTML Report

Generate a self-contained report at `output/evaluation/face_model_evaluation.html`.
Local project assets may be embedded as data URLs so the report remains viewable when
opened directly from disk. The report contains:

1. Executive verdict, score, rating label, and concise rationale.
2. Pipeline timeline and artifact inventory.
3. Metric cards and threshold explanations.
4. Source-to-reconstruction visual comparisons.
5. Geometry-only and textured multi-angle render galleries.
6. Texture atlas and diagnostic overlays.
7. Strengths, defects, severity, and recommended next fixes.
8. An interactive 3D viewer when local browser support permits, with static renders as
   the guaranteed fallback.

## Error Handling

If a stage fails, retain logs and partial artifacts, diagnose the root cause, and only
make a narrowly scoped fix when required to execute the existing intended pipeline.
Any fix must be reported separately from the model-quality assessment. If rendering or
interactive viewing is unavailable, complete the metric and static-image report and
state the limitation explicitly.

## Verification

- Confirm the two pipeline entry points exit successfully.
- Confirm required OBJ, camera, texture, and GLB artifacts are newly generated.
- Validate metric JSON for finite values and expected fields.
- Open the HTML report in a browser, check console errors, image loading, layout,
  interaction, and representative desktop viewport screenshots.
- Cross-check the written verdict against both measured thresholds and visual evidence.
