from pathlib import Path

from src.reports.offline_glb_compare import write_offline_glb_compare_viewer


def test_offline_compare_viewer_uses_relative_local_assets(tmp_path: Path):
    output = tmp_path / "experiment" / "compare.html"
    left = tmp_path / "experiment" / "meshes" / "baseline.glb"
    right = tmp_path / "experiment" / "meshes" / "strict.glb"
    vendor = tmp_path / "frontend" / "vendor"
    left.parent.mkdir(parents=True)
    vendor.mkdir(parents=True)
    left.write_bytes(b"left")
    right.write_bytes(b"right")

    write_offline_glb_compare_viewer(
        output_path=output,
        left_model=left,
        right_model=right,
        vendor_root=vendor,
        title="A2 comparison",
        left_label="Baseline",
        right_label="Strict",
    )

    html = output.read_text(encoding="utf-8")
    assert "meshes/baseline.glb" in html
    assert "meshes/strict.glb" in html
    assert "../frontend/vendor/three.module.js" in html
    assert 'data-view="underside"' in html
    assert "https://" not in html
    assert "__LEFT_MODEL__" not in html
