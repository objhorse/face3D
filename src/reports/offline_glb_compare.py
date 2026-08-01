"""Generate an offline side-by-side GLB comparison viewer."""

from __future__ import annotations

import os
from pathlib import Path


def _relative_url(source: Path, destination_dir: Path) -> str:
    return os.path.relpath(source.resolve(), destination_dir.resolve()).replace("\\", "/")


def write_offline_glb_compare_viewer(
    *,
    output_path: Path,
    left_model: Path,
    right_model: Path,
    vendor_root: Path,
    title: str,
    left_label: str,
    right_label: str,
) -> Path:
    """Write a file:// compatible viewer backed only by local assets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = output_path.parent
    values = {
        "three": _relative_url(vendor_root / "three.module.js", destination),
        "controls": _relative_url(
            vendor_root / "three" / "controls" / "OrbitControls.js", destination
        ),
        "loader": _relative_url(
            vendor_root / "three" / "loaders" / "GLTFLoader.js", destination
        ),
        "left_model": _relative_url(left_model, destination),
        "right_model": _relative_url(right_model, destination),
        "title": title,
        "left_label": left_label,
        "right_label": right_label,
    }
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
    body { background: #10151b; color: #eef3f6; font: 14px "Segoe UI", sans-serif; }
    header {
      position: fixed; z-index: 4; inset: 0 0 auto 0; min-height: 54px;
      display: flex; align-items: center; gap: 8px; padding: 9px 12px;
      border-bottom: 1px solid #35434e; background: rgba(15, 21, 27, 0.96);
    }
    h1 { min-width: 0; margin: 0 auto 0 0; font-size: 14px; font-weight: 600; }
    .controls { display: flex; gap: 3px; padding: 3px; border: 1px solid #35434e; border-radius: 6px; }
    button {
      min-height: 32px; padding: 5px 10px; border: 0; border-radius: 4px;
      background: transparent; color: #c9d4dc; cursor: pointer; font: inherit;
    }
    button:hover { background: #26323b; }
    button.active { background: #d8e7ef; color: #11181d; }
    #compare { position: fixed; inset: 54px 0 0; display: grid; grid-template-columns: 1fr 1fr; }
    .panel { position: relative; min-width: 0; min-height: 0; overflow: hidden; }
    .panel + .panel { border-left: 1px solid #35434e; }
    .canvas { position: absolute; inset: 0; }
    .label, .status {
      position: absolute; z-index: 2; left: 10px; padding: 7px 10px;
      border: 1px solid #35434e; border-radius: 5px; background: rgba(15, 21, 27, 0.9);
    }
    .label { top: 10px; color: #eef3f6; }
    .status { bottom: 10px; color: #9fc3d4; font-size: 12px; }
    .status.error { border-color: #a95757; color: #ffc0c0; }
    @media (max-width: 760px) {
      header { flex-wrap: wrap; }
      h1 { width: 100%; }
      #compare { inset-block-start: 96px; grid-template-columns: 1fr; grid-template-rows: 1fr 1fr; }
      .panel + .panel { border-left: 0; border-top: 1px solid #35434e; }
      button { padding-inline: 7px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="controls" id="modes">
      <button type="button" data-mode="texture" class="active">Texture</button>
      <button type="button" data-mode="clay">Clay</button>
      <button type="button" data-mode="wire">Wire</button>
    </div>
    <div class="controls" id="views">
      <button type="button" data-view="front" class="active">Front</button>
      <button type="button" data-view="left">Left</button>
      <button type="button" data-view="right">Right</button>
      <button type="button" data-view="underside">Underside</button>
    </div>
    <div class="controls"><button type="button" id="reset">Reset</button></div>
  </header>
  <main id="compare">
    <section class="panel">
      <div class="canvas" id="left"></div>
      <div class="label">__LEFT_LABEL__</div>
      <div class="status" id="left-status">Loading model</div>
    </section>
    <section class="panel">
      <div class="canvas" id="right"></div>
      <div class="label">__RIGHT_LABEL__</div>
      <div class="status" id="right-status">Loading model</div>
    </section>
  </main>
  <script type="module">
    import * as THREE from '__THREE__';
    import { OrbitControls } from '__CONTROLS__';
    import { GLTFLoader } from '__LOADER__';

    const clay = new THREE.MeshStandardMaterial({
      color: 0xb8b0a3, roughness: 0.82, metalness: 0, side: THREE.DoubleSide
    });
    const wire = new THREE.MeshBasicMaterial({
      color: 0xc9e4ef, wireframe: true, transparent: true, opacity: 0.42
    });

    class ModelPanel {
      constructor(hostId, statusId, modelUrl) {
        this.host = document.getElementById(hostId);
        this.status = document.getElementById(statusId);
        this.modelUrl = modelUrl;
        this.root = null;
        this.distance = 1;
        this.activeMode = 'texture';
        this.activeView = 'front';
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x10151b);
        this.camera = new THREE.PerspectiveCamera(38, 1, 0.0001, 100);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
        this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.05;
        this.host.appendChild(this.renderer.domElement);
        this.scene.add(new THREE.HemisphereLight(0xffffff, 0x303944, 1.8));
        const key = new THREE.DirectionalLight(0xffffff, 2.2);
        key.position.set(1.5, 1.4, 2.2);
        this.scene.add(key);
        const fill = new THREE.DirectionalLight(0xc9e0ed, 0.9);
        fill.position.set(-2, 0.5, 1);
        this.scene.add(fill);
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.enablePan = false;
        this.resize();
        new GLTFLoader().load(modelUrl, (gltf) => this.loaded(gltf.scene), undefined, (error) => {
          console.error(error);
          this.status.textContent = `Load failed: ${error.message || error}`;
          this.status.classList.add('error');
        });
      }
      loaded(root) {
        this.root = root;
        this.scene.add(root);
        const box = new THREE.Box3().setFromObject(root);
        if (box.isEmpty()) throw new Error('Model contains no visible geometry');
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        root.position.sub(center);
        const verticalFov = THREE.MathUtils.degToRad(this.camera.fov);
        const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
        this.distance = Math.max(
          size.y / (2 * Math.tan(verticalFov / 2)),
          Math.max(size.x, size.z) / (2 * Math.tan(horizontalFov / 2))
        );
        this.camera.near = Math.max(this.distance / 1000, 0.0001);
        this.camera.far = this.distance * 100;
        this.camera.updateProjectionMatrix();
        this.controls.minDistance = this.distance * 0.25;
        this.controls.maxDistance = this.distance * 4;
        this.setMode(this.activeMode);
        this.setView(this.activeView);
        this.status.textContent = 'Ready';
      }
      setMode(mode) {
        this.activeMode = mode;
        if (!this.root) return;
        this.root.traverse((object) => {
          if (!object.isMesh) return;
          if (!object.userData.textureMaterial) object.userData.textureMaterial = object.material;
          object.material = mode === 'texture' ? object.userData.textureMaterial : mode === 'wire' ? wire : clay;
        });
      }
      setView(view) {
        this.activeView = view;
        if (!this.root) return;
        const portrait = this.host.clientWidth <= 600;
        const d = this.distance * (portrait ? 1.28 : 1.15);
        const targetY = portrait ? this.distance * 0.06 : 0;
        if (view === 'left') this.camera.position.set(-d, targetY, 0);
        else if (view === 'right') this.camera.position.set(d, targetY, 0);
        else if (view === 'underside') this.camera.position.set(0, -d * 0.78, d * 0.62);
        else this.camera.position.set(0, targetY + this.distance * 0.03, d);
        this.camera.lookAt(0, targetY, 0);
        this.controls.target.set(0, targetY, 0);
        this.controls.update();
      }
      resize() {
        const width = Math.max(this.host.clientWidth, 1);
        const height = Math.max(this.host.clientHeight, 1);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
      }
      render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
      }
    }

    const panels = [
      new ModelPanel('left', 'left-status', '__LEFT_MODEL__'),
      new ModelPanel('right', 'right-status', '__RIGHT_MODEL__')
    ];
    document.getElementById('modes').addEventListener('click', (event) => {
      const button = event.target.closest('[data-mode]');
      if (!button) return;
      panels.forEach((panel) => panel.setMode(button.dataset.mode));
      document.querySelectorAll('[data-mode]').forEach((item) => item.classList.toggle('active', item === button));
    });
    document.getElementById('views').addEventListener('click', (event) => {
      const button = event.target.closest('[data-view]');
      if (!button) return;
      panels.forEach((panel) => panel.setView(button.dataset.view));
      document.querySelectorAll('[data-view]').forEach((item) => item.classList.toggle('active', item === button));
    });
    document.getElementById('reset').addEventListener('click', () => {
      panels.forEach((panel) => panel.setView('front'));
    });
    addEventListener('resize', () => panels.forEach((panel) => panel.resize()));
    rendererLoop();
    function rendererLoop() {
      panels.forEach((panel) => panel.render());
      requestAnimationFrame(rendererLoop);
    }
  </script>
</body>
</html>
"""
    replacements = {
        "__TITLE__": values["title"],
        "__LEFT_LABEL__": values["left_label"],
        "__RIGHT_LABEL__": values["right_label"],
        "__THREE__": values["three"],
        "__CONTROLS__": values["controls"],
        "__LOADER__": values["loader"],
        "__LEFT_MODEL__": values["left_model"],
        "__RIGHT_MODEL__": values["right_model"],
    }
    for marker, value in replacements.items():
        html = html.replace(marker, value)
    output_path.write_text(html, encoding="utf-8")
    return output_path
