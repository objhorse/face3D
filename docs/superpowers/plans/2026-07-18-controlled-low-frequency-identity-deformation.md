# Controlled Low-Frequency Identity Deformation Implementation Plan

## Context

The repository already contains a legacy module named
`controlled_identity_deformation.py`. Its active deformation graph exposes roughly
660 translation values, permits central-feature motion, and ranks candidates from
triangulated attachment error. The new stage keeps the module boundary but does not
use that graph as its identity parameterization.

## Batch 1: Low-Frequency Basis

Files:

- `src/geometry/controlled_identity_deformation.py`
- `tests/test_controlled_identity_deformation.py`

Tasks:

1. Add a deterministic 12-18 parameter semantic basis on the low-resolution FLAME
   mesh.
2. Define paired width/depth controls and chin length/depth controls in canonical
   FLAME coordinates.
3. Build smooth face support and protected eye/nose/philtrum/mouth cores from fitted
   3D landmarks.
4. Add bounded NumPy and Torch application functions.
5. Test basis size, mirror behavior, protected support, locality, fixed topology, and
   displacement bounds on a synthetic symmetric face.

Verification:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_controlled_identity_deformation.py -q
```

## Batch 2: Multi-View Optimizer

Files:

- `src/geometry/controlled_identity_deformation.py`
- `src/geometry/differentiable_silhouette.py`
- `tests/test_controlled_identity_deformation.py`

Tasks:

1. Optimize tanh-bounded control coefficients while cameras, expressions, and FLAME
   shape stay frozen.
2. Reuse front and subject-facing profile SDF losses.
3. Add interior landmark anchors, symmetry, coefficient, edge, and differential
   coordinate losses.
4. Save deterministic checkpoints and select the best candidate rather than the last
   iteration.
5. Implement gates for front improvement, profile preservation, interior landmarks,
   protected motion, displacement, and topology.

Verification:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests\test_controlled_identity_deformation.py tests\test_differentiable_silhouette.py tests\test_mesh_quality.py -q
```

## Batch 3: Stable Pipeline Integration

Files:

- `src/config.py`
- `src/module2_geometry.py`
- `src/pipeline/stable_three_view.py`
- `tests/test_reconstruction_recovery.py`

Tasks:

1. Add disabled-by-default controlled-identity configuration.
2. Run the stage after FLAME shape fitting and before subdivision.
3. Apply the selected low-resolution displacement to neutral and expression meshes.
4. Leave all legacy free-residual stages disabled in the stable path.
5. Write baseline/candidate meshes, coefficients, summary, basis support, and
   calibrated overlays.
6. Fall back to the exact baseline when no checkpoint passes.

Verification:

```powershell
D:\Anaconda\envs\gaussian\python.exe -m pytest tests -q
```

## Batch 4: Geometry-First Experiment

Dataset:

- `D:\face3D\captures_20260612_135253`

Tasks:

1. Run the stable pipeline with controlled identity enabled into a new experiment
   directory.
2. Compare baseline and candidate through calibrated front/left/right geometry
   renders before texture.
3. Inspect numerical gates and local mesh quality.
4. Bake texture only if the geometry candidate passes.
5. Generate a file-based viewer and report that work with the user's proxy setup.

Acceptance:

- front trusted-boundary mean improves by at least 30%;
- each profile worsens by no more than 10%;
- interior landmarks worsen by no more than 10%;
- protected central features remain stable;
- no topology or normal-flip regression;
- matched-camera appearance is visibly closer to the source.

## Rollback

The stage remains disabled by default until Batch 4 passes. A rejected or failed
candidate never replaces the baseline geometry. Commit `d547868` remains the stable
pre-SDF recovery point.
