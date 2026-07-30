# Nasal A/B Viewer Parity Design

## Context

The v3 A/B page displays the unchanged baseline GLB on the left, but that model
looks different from the same GLB in the earlier expression-depth viewer. The
baseline artifact is byte-identical (`BF12A6C...E2E71B`), so the regression is
in presentation rather than reconstruction.

The earlier viewer uses Three.js `PerspectiveCamera`, `GLTFLoader`,
`OrbitControls`, sRGB output, and hemisphere/directional lighting. The v3 page
uses a custom orthographic WebGL shader, ignores normals and lighting, and
clamps zoom to `0.55-2.2`. Those differences invalidate visual A/B judgments.

## Goal

Restore the nasal A/B viewer to the established Three.js presentation path so
the baseline and candidate can be judged under the same camera, material,
lighting, color, framing, and interaction behavior as previous accepted
viewers.

## Non-Goals

- Do not modify either GLB, mesh vertices, faces, normals, UVs, or materials.
- Do not rebake, recolor, inpaint, or otherwise modify texture files.
- Do not rerun nasal optimization or change any objective or quality gate.
- Do not promote the v3 candidate over the baseline.
- Do not change the default reconstruction pipeline.

## Viewer Architecture

`run_multiview_nasal_shape_experiment.py` will generate the A/B page using the
same local Three.js stack as the earlier accepted expression-depth viewer:

- `frontend/vendor/three.module.js`
- `frontend/vendor/three/controls/OrbitControls.js`
- `frontend/vendor/three/loaders/GLTFLoader.js`

Both GLBs remain embedded as base64 data so the model payloads are unchanged.
The page may reference only the repository-local viewer modules; it must work
through `file://` without localhost, a CDN, or network access.

## Rendering Parity

The restored page uses:

- `THREE.PerspectiveCamera(38, ...)`, matching the accepted viewer.
- `THREE.WebGLRenderer` with antialiasing and sRGB output.
- The same hemisphere, key, and fill lighting values as the accepted viewer.
- Native glTF scene traversal and materials through `GLTFLoader`.
- A shared model center, scale, and camera distance for both A/B halves.
- Scissor rendering so both models use one synchronized camera.

No custom orthographic vertex projection or raw unlit texture shader
participates in the comparison.

## Interaction

- Mouse drag rotates both models synchronously.
- Wheel zoom uses `OrbitControls` without the custom `0.55-2.2` clamp.
- Front, subject-left, and subject-right presets remain available.
- Texture, pure-geometry, and wireframe modes remain available.
- Loading progress and visible failure text remain available.
- `window.viewerReady` and `window.viewerError` remain available for automated
  browser verification.

## Data Integrity

The viewer generator records or verifies the SHA-256 of the embedded baseline
and candidate GLBs. Tests decode both embedded payloads and require byte-for-byte
equality with the corresponding source files. Viewer generation is therefore
not allowed to rewrite a model while preparing the page.

## Verification

### Automated

- Generated HTML imports only repository-local Three.js modules.
- Embedded baseline and candidate bytes match their source GLBs.
- The page contains perspective camera, sRGB, lights, and `OrbitControls`.
- The custom orthographic shader and zoom clamp are absent.
- Headless Chrome waits for `window.viewerReady` and confirms both halves render
  non-background pixels.
- Existing runner, GLB validation, and nasal geometry tests remain green.

### Visual

Regenerate A/B pages for both `captures_20260612_135253` and
`captures_20260612_140210`. Compare the left baseline against the prior
expression-depth viewer at front and side presets. Baseline facial proportions,
material response, color, and interaction must match before judging candidate
likeness or starting any texture repair.

## Acceptance

This phase is complete when:

1. The unchanged baseline again looks and behaves like it does in the accepted
   Three.js viewer.
2. Both halves support synchronized free rotation and practical wheel zoom.
3. The two embedded GLB hashes are unchanged.
4. No geometry, texture, optimization, or pipeline code changes are included.
5. The user can make a meaningful baseline-versus-candidate judgment from the
   regenerated `file://` pages.
