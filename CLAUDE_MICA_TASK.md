# Claude Task Brief: Upgrade Face3D Initialization for Higher-Fidelity Reconstruction

## Goal

Upgrade the current geometry initialization pipeline so the project no longer relies primarily on `face_alignment` / weak landmark-based initialization.

The immediate target is:

- replace the current shape initialization with a stronger identity model
- keep the existing FLAME-based multi-view optimization pipeline
- improve realism of the reconstructed face, especially:
  - side profile
  - jawline
  - lower face contour
  - nose projection / nasal profile

This project is **not** aiming for a cartoonized or approximate face. The priority is **maximum realism** within the current FLAME-based framework.

## High-Level Direction

Please implement a new initialization backend:

- `MICA` for identity / shape initialization
- `DECA` or `EMOCA` for per-view expression and pose initialization
- then feed these results into the existing multi-view FLAME optimization

The existing FLAME fitting and export pipeline should remain the main downstream path unless refactoring is clearly necessary.

## Current Situation

The current codebase uses a pipeline centered around:

- preprocessing in `src/module1_preprocess.py`
- geometry fitting in `src/module2_geometry.py`
- FLAME optimization with sparse landmark constraints

Current issues:

- sparse landmarks dominate optimization too much
- side-view landmarks are unstable
- current initialization is not strong enough for identity-faithful reconstruction
- output can be "usable" but is not realistic enough for high-fidelity / medical-aesthetic expectations

We have already confirmed that:

- mask quality is not the primary bottleneck
- depth displacement is not the primary bottleneck
- the main weakness is geometry initialization + sparse landmark-driven fitting

## Required Implementation

### 1. Add a new initialization backend

Add a new initialization mode that uses:

- `MICA` to estimate identity shape
- `DECA` or `EMOCA` to estimate per-view expression / pose

This backend should become available alongside existing initialization methods.

Suggested config values:

- `INIT_BACKEND = "face_alignment"`
- `INIT_BACKEND = "deca"`
- `INIT_BACKEND = "mica_deca"`
- optionally `INIT_BACKEND = "mica_emoca"`

The new backend should produce:

- shared identity shape init across views
- per-view expression init
- per-view pose init

### 2. Preserve current FLAME fitting pipeline

Do not replace the whole geometry pipeline unless necessary.

The intended design is:

1. collect stronger initial parameters from MICA + DECA/EMOCA
2. feed them into the current FLAME multi-view optimization
3. preserve existing output formats and downstream texture/export steps

### 3. Add a clean initializer abstraction

Do not pile all new logic directly into the current `module2_geometry.py` if avoidable.

Please introduce a cleaner structure, for example:

- `src/initializers/mica_initializer.py`
- `src/initializers/deca_initializer.py`
- `src/initializers/emoca_initializer.py`
- `src/initializers/__init__.py`

Then `module2_geometry.py` should call into those helpers.

### 4. Shared shape fusion across multi-view inputs

The new initialization stage should treat identity shape as shared across views.

Expected behavior:

- each view may yield its own identity-related estimate
- these should be merged into one shared shape init
- expression and pose should remain per-view

Please use a reasonable fusion strategy, such as:

- mean over available MICA identity embeddings / shape parameters
- confidence-aware averaging if confidence can be estimated

If MICA provides identity features rather than direct FLAME shape coefficients, implement the cleanest practical mapping available in that model's standard workflow.

### 5. Backward compatibility

The project should still be runnable without the new backend enabled.

If MICA or EMOCA is unavailable:

- existing `face_alignment` / DECA fallback path should still work
- code should fail gracefully with actionable error messages

## Configuration Requirements

Please update `src/config.py` to support the new backend clearly.

Add config entries such as:

- `INIT_BACKEND`
- `MICA_DIR`
- `EMOCA_DIR`
- `DECA_DIR`
- model checkpoint paths if needed
- optional flags for CPU/GPU behavior

If additional setup is needed, document it in comments or a short README section.

## Debug / Diagnostics Requirements

This part is important. Do not only wire the model and stop there.

For each view, add debug outputs that let us compare old vs new initialization quality.

Required outputs:

- initial mesh projection image before optimization
- initial landmark reprojection image before optimization
- saved initial parameters per view
- saved shared shape init
- a short text or JSON summary of:
  - initialization backend used
  - which views succeeded
  - which model produced shape / exp / pose

Suggested debug output files:

- `output/debug/init_mica_shape.json`
- `output/debug/init_view_left.json`
- `output/debug/init_view_front.json`
- `output/debug/init_view_right.json`
- `output/debug/init_reproj_left.png`
- `output/debug/init_reproj_front.png`
- `output/debug/init_reproj_right.png`

## Evaluation / A-B Comparison

Please make it easy to compare:

- current baseline initialization
- new MICA-based initialization

At minimum, there should be a clear switch in config so we can run both modes on the same input set.

Please ensure the following are directly comparable:

- final reprojection errors
- initial reprojection errors
- resulting mesh visual quality
- side-view stability

## Acceptance Criteria

The implementation is considered successful only if:

1. The project can run with the new backend enabled.
2. The pipeline still produces the normal geometry outputs.
3. The debug outputs clearly show what initialization was used.
4. On the same three-view test set, initialization quality is visibly stronger than the current baseline.
5. Side profile / lower-face geometry should improve, or at minimum the initialization should give a better starting point than `face_alignment`-only setup.

## Constraints

- Preserve the current repo structure where possible.
- Avoid destructive rewrites unless necessary.
- Keep changes modular.
- Prefer explicit config-driven behavior over hardcoded branching.
- Maintain compatibility with existing outputs and downstream steps.

## Suggested Files To Modify

Primary:

- `src/module2_geometry.py`
- `src/config.py`

Likely additions:

- `src/initializers/mica_initializer.py`
- `src/initializers/deca_initializer.py`
- `src/initializers/emoca_initializer.py`

Optional:

- docs describing setup for checkpoints / dependencies

## Deliverables

Please provide:

1. Code changes implementing the new initialization backend
2. Notes on dependency / checkpoint requirements
3. A short summary of what was changed
4. Any assumptions or limitations
5. A comparison note between baseline and new initialization

## Important Clarification

The objective is not just "better landmark fitting."

The objective is:

- more realistic identity shape
- better geometric initialization
- better downstream reconstruction realism

If any part of the current sparse-landmark fitting has to be slightly reorganized to support that goal cleanly, that is acceptable.
