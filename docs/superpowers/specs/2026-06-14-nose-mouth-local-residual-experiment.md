# Nose/Mouth Local Residual Experiment

## Goal

Test whether a small, local per-vertex residual deformation around the nose base and lips can improve the current face identity without damaging the successful cheek/jaw profile correction.

The previous profile contour residual improved the outer face shape, but the nose and mouth region stayed almost unchanged because the stable anchor mask protected nose/eye/mouth vertices. The diagnosis page at `output/debug/nose_mouth_direction_check/index.html` shows roughly 10 px nose/mouth landmark error and no mouth metric improvement.

## Scope

This experiment is intentionally narrow:

- Keep the existing FLAME identity and profile contour residual active.
- Add a second local residual pass after the profile residual and before depth displacement.
- Only target the nose base, nostril-side region, and outer/inner lip landmarks.
- Keep eye landmarks and the nose bridge centerline protected.
- Keep offsets bounded to a smaller range than the global profile residual.

## Editable And Protected Regions

Editable landmarks:

- Nose base and wings: 31-35.
- Outer mouth: 48-59.
- Inner mouth: 60-67 with lower weight.

Protected landmarks:

- Eyes: 36-47.
- Nose bridge: 27-30.
- Face contour residual anchors remain protected outside the local nose/mouth patch.

The local patch is built from projected landmark neighborhoods in the front view, then constrained by multi-view visibility so it does not spread into cheek, jaw, neck, or ear regions.

## Acceptance Criteria

Accept the deformation only when all conditions hold:

- Nose base and outer mouth mean landmark error improves by at least 1 px overall.
- Front nose/mouth landmark error improves by at least 1.5 px.
- Eye and nose-bridge stable error worsens by no more than 0.35 px.
- Existing profile contour side mean does not worsen by more than 1 px.
- Maximum local offset stays below 10 mm.
- Moved vertex ratio stays below 5%.

## Debug Output

Write a dedicated audit page to:

- `output/debug/nose_mouth_local_residual/index.html`

It should include before/after landmark overlays, residual heatmaps, a JSON summary, and clear accept/reject reasons.

## Expected Result

If the direction is right, the front view should show smaller correction vectors around nostrils and lip boundaries, while eyes and nose bridge remain visually stable. If rejected, the experiment should leave the mesh unchanged and still write the diagnostic page.
