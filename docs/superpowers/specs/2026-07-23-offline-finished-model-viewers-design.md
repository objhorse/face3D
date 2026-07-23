# Offline Finished-Model Viewers Design

## Goal

Create one directly openable 3D viewer for each finished reconstruction:

- `captures_20260612_135253_controlled_identity_v1`
- `captures_20260612_140210_full_rerun_20260719`

Each viewer displays that experiment's textured `meshes/face.glb` and is intended for judging the finished modeling result, not projection diagnostics.

## Output

Each experiment root receives `model_viewer.html`. The viewer loads `meshes/face.glb` using the repository's local Three.js, OrbitControls, and GLTFLoader modules under `frontend/vendor`. It does not use a CDN, localhost, or external network resources.

## Interaction

- Textured, clay, and wireframe display modes.
- Front, subject-left, and subject-right preset views.
- Mouse drag to rotate and wheel to zoom.
- Reset view and fullscreen controls.
- Clear loading progress and a visible error message if the GLB cannot be loaded.

## Presentation

The two pages use the same compact, work-focused layout and correct UTF-8 Chinese labels. The canvas fills the viewport, controls stay at the top, and the model is automatically centered and framed without changing its geometry or material data.

## Verification

- Confirm both HTML files reference the correct relative GLB path and only local JavaScript modules.
- Open both pages through `file://`.
- Capture desktop screenshots and verify that the textured face is nonblank, centered, and responds to preset views.
- Confirm textured, clay, and wireframe modes do not mutate the GLB files.
