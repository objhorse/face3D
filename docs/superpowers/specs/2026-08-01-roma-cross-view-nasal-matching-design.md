# RoMa Cross-View Nasal Matching Design

Date: 2026-08-01

## Goal

Upgrade the current cross-view nasal texture evidence stage from sparse LoFTR
seeds plus handcrafted patch scoring to dense learned correspondence proposals.
The upgraded stage must recover enough trustworthy bilateral observations to
drive nasal geometry fitting while preserving the fixed camera rig and the
accepted `alar_surface_v10` result.

The run is allowed to proceed automatically from evidence extraction to a new
candidate mesh and comparison viewer. It must never overwrite the v10 mesh,
`face.glb`, an existing texture, or an existing experiment directory.

## Current Problem

The Release A audit is geometrically conservative but produces almost no
trusted nasal evidence on `captures_20260612_135253`:

- subject-left: zero trusted observations;
- subject-right: one trusted observation;
- most candidates fail because the source pixel is unreliable, no unique local
  match is found, or reciprocal matching fails.

The limiting factor is therefore the correspondence proposal stage. Expanding
the handcrafted search window would increase ambiguity around smooth skin,
specular highlights, nostril shadows, and repeated gradients without adding
new information.

## Chosen Approach

Use RoMa as a dense learned proposal generator on cropped nasal regions, then
let the calibrated three-camera rig decide which proposals are valid 3D
measurements. RoMa does not directly deform the mesh and does not replace the
existing geometry checks.

Alternatives considered:

- Multi-scale LoFTR: least integration work, but it remains sparse and the
  current LoFTR-assisted pipeline already lacks bilateral nasal coverage.
- DINOv2 feature-grid matching: robust at semantic recognition but too coarse
  for precise alar and nostril localization; it remains a possible future
  coarse prior.
- RoMa dense matching: selected because it returns dense warps and certainty,
  allowing the rig validator to choose well-distributed evidence rather than
  depending on a few detected keypoints.

Reference implementations:

- RoMa: https://github.com/Parskatt/RoMa
- LoFTR in Kornia: https://kornia.readthedocs.io/en/latest/models/loftr.html
- DINOv2: https://github.com/facebookresearch/dinov2

## Architecture

### 1. Dense Matcher Adapter

Add a small adapter that owns RoMa model loading and inference. Its public
output is independent of the RoMa package and contains:

- front pixel;
- side-view pixel;
- learned certainty;
- source and destination crop transforms;
- matcher identity and model configuration.

Inference runs one image pair at a time on the nasal crops. The default mode is
CUDA half precision where supported, with bounded resolution suitable for the
local RTX 4060 8 GB GPU. CPU execution is diagnostic fallback only.

### 2. Canonical Nasal Crops

The existing locked v10 mesh, projection matrices, semantic nasal regions, and
coordinate provenance define each crop. Every dense-match coordinate is mapped
back to the existing work-image coordinate system before validation. Crop
offsets, resizing, and any padding are explicit metadata; no implicit image
coordinate conversion is permitted.

### 3. Rig-Constrained Validation

RoMa proposes candidates but fixed-rig geometry validates them. The validator
applies the existing semantic and source-pixel reliability masks, followed by:

- learned-certainty filtering;
- fixed-rig epipolar residual;
- forward/backward consistency;
- one-to-one correspondence selection;
- positive camera depth and minimum triangulation angle;
- per-view reprojection residual;
- distance from the triangulated point to the exact locked v10 triangle
  surface;
- bilateral spatial coverage across alar, nasal-tip, and peri-nasal support.

The handcrafted affine-gradient and census score remains an independent weak
check, not the primary proposal mechanism. Highlight, saturation, deep nostril
shadow, and mask-boundary pixels receive low observation confidence instead of
being treated as shape evidence.

### 4. Confidence and Evidence Contract

Each accepted observation stores both learned and geometric confidence. Final
weight is derived from learned certainty, image reliability, reciprocal
consistency, triangulation conditioning, and reprojection quality. Rejection
reasons remain explicit in the audit output.

Geometry fitting is permitted only when both subject sides satisfy the existing
Release A coverage contract. The contract must require multiple distinct
surface anchors and more than one nasal semantic subregion per side; a dense
cluster on one highlight or nostril edge is not sufficient.

### 5. Automatic Candidate Generation

When bilateral evidence passes, the runner automatically invokes the existing
nasal surface fitting path with the accepted triangulated observations. Camera
parameters, texture coordinates, and non-nasal geometry remain fixed. The
candidate receives the same accepted texture and viewer settings as the v10
baseline so that the comparison reflects geometry rather than presentation
changes.

The candidate is written to a new experiment directory such as:

`output/experiments/captures_20260612_135253_roma_nasal_v1/`

Expected artifacts include:

- `baseline.glb` copied or referenced from the locked v10 artifact;
- `candidate.glb` containing only the new nasal geometry result;
- correspondence and triangulation diagnostics;
- a machine-readable quality summary;
- a local-file-compatible baseline/candidate viewer.

If the evidence contract fails, no candidate geometry is emitted. The run still
produces a diagnostic report explaining the limiting rejection categories.

## Safety and Isolation

- The v10 baseline is immutable input.
- Existing `face.glb`, textures, and experiment directories are not modified.
- Camera intrinsics and extrinsics are fixed during matching and fitting.
- RoMa output alone never moves a vertex.
- The output directory is created with a unique version and refuses to replace
  an existing run.
- Existing uncommitted eyelid experiments are outside this work's scope.

## Testing

Unit tests cover crop-coordinate round trips, matcher-output validation,
confidence fusion, geometric rejection reasons, duplicate removal, bilateral
coverage, and fail-closed behavior.

Integration tests use deterministic synthetic correspondences to verify that:

- correct fixed-rig matches triangulate and reach the fitter;
- epipolar-inconsistent or reciprocal-inconsistent matches cannot deform the
  mesh;
- insufficient unilateral evidence emits diagnostics but no candidate;
- successful bilateral evidence creates a new candidate and viewer without
  changing the baseline files.

The real-data smoke test runs `captures_20260612_135253` end to end, records GPU
memory behavior, and compares baseline and candidate with identical texture,
camera framing, lighting, and controls.

## Acceptance Criteria

- RoMa inference completes on the local RTX 4060 8 GB GPU without out-of-memory
  failure.
- Trusted evidence exists on both subject sides with distributed nasal-region
  coverage.
- Accepted observations satisfy the fixed-rig triangulation and reprojection
  checks.
- A new textured baseline/candidate viewer is generated automatically when the
  bilateral evidence contract passes.
- The v10 baseline and all existing outputs remain byte-for-byte unchanged.
- The candidate has no new holes, inverted triangles, degenerate faces, or
  non-nasal vertex movement.
- The report clearly separates learned match certainty from geometric validity.

## Out of Scope

- Re-estimating camera extrinsics or intrinsics;
- changing the global face identity or expression fit;
- replacing the texture-baking pipeline;
- medical-grade accuracy claims;
- training or fine-tuning a new correspondence network.
