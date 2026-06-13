# Free Face Repair Plan

## Goal

Make the reconstructed face match the real subject's face shape and expression more closely than the FLAME template can provide by itself.

## Steps

1. **Real expression mode**
   - Preserve the expression from the source photos.
   - Disable forced closed-eye landmark weighting and forced closed-eye mesh deformation by default.
   - Regenerate the HTML effect report after the change.

2. **Free identity deformation layer**
   - Keep FLAME as the initial topology and pose scaffold.
   - Add a non-parametric mesh deformation layer driven by dense landmarks and face masks.
   - Let cheeks, jaw, temples, and mouth corners move outside the limited FLAME shape space.
   - Status: first landmark-handle version implemented on 2026-05-22, then disabled after over-deforming the surface.
   - Guardrail: reject any version that moves too much of the mesh before stable anchors are added.

3. **Stable facial anchors**
   - Keep nose bridge, eye centers, and inner mouth structure stable.
   - Let outline and soft tissue areas deform more freely.
   - Status: implemented as a shared safety layer on 2026-05-22.
   - Guardrail: free-face and future free-identity deformation must pass anchor-motion, moved-ratio, and mesh offset-jump checks before being applied.

4. **Multi-view validation**
   - Use front and side views to prevent a change that improves one view while breaking another.
   - Continue writing before/after metrics and images into `output/debug/code_change_effect_report.html`.
   - Status: implemented as a deformation gate on 2026-05-22.
   - Guardrail: front view must improve, overall contour must improve, side-view worsening must stay below a strict threshold, and a JSON validation report is written under `output/debug/multiview_validation/`.
   - Regression note: after a visual failure with chin fragmentation and temple holes, the gate was tightened on 2026-05-22 and the visible-face crop now keeps one boundary ring.
   - Regression note: after a remaining floating lower-face strip was spotted on 2026-05-23, visible-face crop now removes detached components below 1000 faces.
   - Regression note: a connected side-ear trim was tried on 2026-05-23, then reverted because it removed valid facial surface instead of repairing the texture defect.

5. **Iterate toward likeness**
   - Tune deformation strength and accepted regions by looking at the HTML report.
   - Promote only changes that visibly reduce face-shape and expression mismatch.
