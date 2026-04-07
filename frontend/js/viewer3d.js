/**
 * viewer3d.js — Three.js GLB 查看器
 *
 * 支持：
 *   - GLB 加载（GLTF 2.0）
 *   - OrbitControls 旋转 / 缩放
 *   - 分屏同步旋转
 */

import * as THREE from '../vendor/three.module.js';
import { GLTFLoader } from '../vendor/three/loaders/GLTFLoader.js';
import { OrbitControls } from '../vendor/three/controls/OrbitControls.js';
import { DRACOLoader } from '../vendor/three/loaders/DRACOLoader.js';


export class Viewer3D {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts
     * @param {boolean} [opts.syncTarget]  — 是否作为同步目标（接收旋转，不发出）
     */
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.syncTarget = opts.syncTarget || false;
        this._syncListeners = [];

        this._initRenderer();
        this._initScene();
        this._initCamera();
        this._initControls();
        this._initLights();
        this._startLoop();

        // 响应 canvas 父容器大小变化
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(canvas.parentElement || canvas);
    }

    // ── 初始化 ────────────────────────────────────────────────────────────────

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        const p = this.canvas.parentElement;
        if (p) {
            this.renderer.setSize(p.clientWidth, p.clientHeight);
        }
    }

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0d0d0d);
    }

    _initCamera() {
        const w = this.canvas.parentElement?.clientWidth || 800;
        const h = this.canvas.parentElement?.clientHeight || 600;
        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.001, 100);
        this.camera.position.set(0, 0, 0.35);
    }

    _initControls() {
        this.controls = new OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.05;
        this.controls.maxDistance = 2.0;
        this.controls.target.set(0, 0, 0);
        this.controls.update();

        // 当旋转/缩放时，通知同步目标
        this.controls.addEventListener('change', () => {
            if (!this.syncTarget) {
                this._syncListeners.forEach(fn => fn(this.camera, this.controls.target));
            }
        });
    }

    _initLights() {
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);

        const key = new THREE.DirectionalLight(0xffffff, 1.4);
        key.position.set(0.5, 1, 1);
        this.scene.add(key);

        const fill = new THREE.DirectionalLight(0xffffff, 0.5);
        fill.position.set(-1, 0.5, -0.5);
        this.scene.add(fill);
    }

    _startLoop() {
        this._animId = requestAnimationFrame(this._loop.bind(this));
    }

    _loop() {
        this._animId = requestAnimationFrame(this._loop.bind(this));
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    _onResize() {
        const p = this.canvas.parentElement;
        if (!p) return;
        const w = p.clientWidth, h = p.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    // ── 公开 API ──────────────────────────────────────────────────────────────

    /**
     * 加载 GLB URL。返回 Promise<void>。
     */
    async loadGLB(url) {
        // 清除现有模型
        this._clearModel();

        const loader = new GLTFLoader();
        const draco = new DRACOLoader();
        draco.setDecoderPath('/ui/vendor/draco/');
        loader.setDRACOLoader(draco);

        const gltf = await new Promise((resolve, reject) => {
            loader.load(url, resolve, undefined, reject);
        });

        const model = gltf.scene;

        // 重算顶点法线，消除平面着色的块状感
        model.traverse(child => {
            if (child.isMesh) {
                child.geometry.computeVertexNormals();
            }
        });

        // 居中 + 归一化大小
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 0.25 / maxDim;
        model.position.sub(center.multiplyScalar(scale));
        model.scale.setScalar(scale);

        this.scene.add(model);
        this._model = model;

        // 重置相机视角
        this.camera.position.set(0, 0, 0.35);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    _clearModel() {
        if (this._model) {
            this.scene.remove(this._model);
            this._model = null;
        }
    }

    /**
     * 注册相机同步回调（当 this 的相机变化时调用）
     */
    onCameraChange(fn) {
        this._syncListeners.push(fn);
    }

    /**
     * 从外部同步相机状态（用于分屏同步）
     */
    syncCamera(srcCamera, srcTarget) {
        // 只同步 spherical 坐标（方向+距离）和 target，不同步 position 绝对值
        const offset = srcCamera.position.clone().sub(srcTarget);
        this.controls.target.copy(srcTarget);
        this.camera.position.copy(srcTarget.clone().add(offset));
        this.camera.quaternion.copy(srcCamera.quaternion);
        this.controls.update();
    }

    dispose() {
        cancelAnimationFrame(this._animId);
        this._resizeObserver.disconnect();
        this.controls.dispose();
        this.renderer.dispose();
    }
}
