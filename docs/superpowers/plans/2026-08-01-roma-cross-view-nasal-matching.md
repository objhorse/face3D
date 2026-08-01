# RoMa Cross-View Nasal Matching Implementation Plan

Date: 2026-08-01
Design: `docs/superpowers/specs/2026-08-01-roma-cross-view-nasal-matching-design.md`

## Scope

Upgrade the uncommitted Release A nasal observation pipeline to use RoMa dense
correspondence proposals, preserve fixed-rig geometric validation, and
automatically create a non-overwriting textured candidate viewer when bilateral
evidence passes.

## Task 1: Dependency and GPU Smoke Test

1. Install `romatch` into the existing `gaussian` environment through the
   configured local proxy.
2. Load the official pretrained matcher on CUDA.
3. Run one bounded-resolution pair and record peak GPU memory.
4. Add a clear dependency/model-loading error when RoMa is unavailable.

## Task 2: Dense Matcher Adapter

Files:

- create `src/geometry/roma_nasal_matcher.py`;
- create `tests/test_roma_nasal_matcher.py`.

Implement immutable crop transforms, package-independent match records, crop to
work-image coordinate round trips, one-pair-at-a-time inference, deterministic
sampling, and explicit model metadata. Test all coordinate conversions without
requiring model weights.

## Task 3: Release A Integration

Files:

- update `run_nasal_texture_observation_audit.py`;
- update `src/geometry/nasal_texture_observations.py`;
- update corresponding tests.

Generate canonical nasal crops from locked v10 projections. Use RoMa proposals
as the primary candidate source and keep handcrafted gradient/census evidence as
an independent weak verifier. Preserve fixed camera parameters, coordinate
provenance, semantic masks, reciprocal consistency, one-to-one selection,
triangulation, reprojection, exact surface-distance checks, and explicit
rejection reasons.

## Task 4: Bilateral Evidence Contract

Require trusted observations on both subject sides with distinct anchors and
multiple nasal semantic subregions. Record learned certainty separately from
geometric validity and produce a fail-closed report if coverage is insufficient.
Add tests for unilateral, clustered, duplicated, and geometrically inconsistent
evidence.

## Task 5: Automatic Candidate and Viewer

Add a runner that consumes successful Release A evidence and invokes the
existing nasal surface fitter. Keep non-nasal vertices, cameras, UVs, texture,
lighting, framing, and viewer controls identical to the locked v10 baseline.
Create only a fresh versioned output directory and refuse replacement.

Expected real-data output:

`output/experiments/captures_20260612_135253_roma_nasal_v1/`

The directory contains baseline and candidate GLBs, quality JSON, match
diagnostics, and a local-file-compatible comparison viewer.

## Task 6: Verification

1. Run focused adapter, observation, triangulation, fitting, and runner tests.
2. Run the existing related regression suite.
3. Run `captures_20260612_135253` end to end on CUDA.
4. Verify byte hashes of locked v10 inputs before and after the run.
5. Inspect the generated viewer and diagnostic images for correspondence
   coverage, nasal likeness, holes, folds, texture parity, camera parity, and
   usable zoom/rotation controls.
6. If bilateral evidence still fails, stop before geometry and report the
   measured limiting cause rather than relaxing geometry constraints blindly.
