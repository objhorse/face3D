# Semantic Eyelid Candidate Texture Rebake Design

## Goal

Create a textured A/B comparison for the accepted semantic-eyelid candidate. The baseline remains the user-approved textured model. The candidate receives a fresh three-view texture bake so its texture registration is consistent with its modified eyelid geometry.

## Geometry Contract

- Treat `semantic_eyelid_v1/meshes/eyelid_candidate.obj` as immutable input.
- Record its SHA-256 hash and vertex, face, UV, and UV-face counts before baking.
- Disable geometry smoothing and face deletion during GLB export.
- Verify the candidate OBJ hash and topology contract again after baking.
- Never overwrite the approved baseline experiment or the pure-geometry eyelid experiment.

## Texture Flow

1. Copy the candidate OBJ and baseline camera metadata into a separate textured output directory.
2. Reuse the calibrated three-view preprocessing and stable texture-registration pipeline.
3. Recompute semantic feature warps, photometric normalization, view blending, and texture confidence for the candidate geometry.
4. Bake a new `albedo_white.png` and export a textured candidate GLB with no geometry smoothing.
5. Keep low-confidence regions as texture-confidence metadata; do not remove mesh faces.

## Comparison Output

- Left model: approved baseline `face.glb` with its original accepted texture.
- Right model: accepted semantic-eyelid candidate with its newly baked texture.
- Generate an offline embedded A/B viewer in the semantic-eyelid experiment directory.
- Label the comparison as a combined geometry-plus-texture result, not an isolated geometry test.

## Acceptance

- Candidate source OBJ hash is unchanged before and after texture baking.
- Candidate GLB loads and contains the full face topology.
- No texture-stage geometry smoothing or face deletion is enabled.
- Baseline artifacts remain byte-for-byte unchanged.
- Viewer works from a `file:///` URL without a local server.
