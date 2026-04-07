/**
 * app.js — 主应用逻辑
 *
 * 功能：
 *   1. 会话列表（侧边栏）
 *   2. 上传 + 重建
 *   3. 单视图 3D 查看
 *   4. 分屏对比（选择两个会话，ICP 对齐后同步旋转）
 *   5. WebSocket 进度显示
 */

import { Viewer3D } from './viewer3d.js';
import { api } from './api.js';

// ── DOM 引用 ──────────────────────────────────────────────────────────────────
const sessionList    = document.getElementById('session-list');
const emptyState     = document.getElementById('empty-state');
const canvasSingle   = document.getElementById('canvas-single');
const splitContainer = document.getElementById('split-container');
const canvasA        = document.getElementById('canvas-a');
const canvasB        = document.getElementById('canvas-b');
const progressOverlay= document.getElementById('progress-overlay');
const progressFill   = document.getElementById('progress-fill');
const progressMsg    = document.getElementById('progress-msg');
const uploadModal    = document.getElementById('upload-modal');
const toast          = document.getElementById('toast');
const btnUpload      = document.getElementById('btn-upload');
const btnCompare     = document.getElementById('btn-compare');
const btnRefresh     = document.getElementById('btn-refresh');
const compareInfo    = document.getElementById('compare-info');

// ── 状态 ─────────────────────────────────────────────────────────────────────
let viewerSingle = null;
let viewerA = null;
let viewerB = null;
let sessions = [];
let selectedIds = [];      // 单选[id] 或 双选[idA, idB]
let compareMode = false;

// ── 初始化 ────────────────────────────────────────────────────────────────────
async function init() {
    // 初始化单视图查看器
    viewerSingle = new Viewer3D(canvasSingle);

    await refreshSessions();

    // 按钮事件
    btnUpload.addEventListener('click', () => showUploadModal());
    btnRefresh.addEventListener('click', refreshSessions);
    btnCompare.addEventListener('click', toggleCompareMode);

    // 上传表单
    document.getElementById('upload-form').addEventListener('submit', handleUpload);
    document.getElementById('btn-cancel-upload').addEventListener('click', hideUploadModal);
}

// ── 会话列表 ──────────────────────────────────────────────────────────────────
async function refreshSessions() {
    try {
        sessions = await api.listSessions();
        renderSessionList();
    } catch (e) {
        showToast('无法加载会话列表: ' + e.message, 'error');
    }
}

function renderSessionList() {
    sessionList.innerHTML = '';
    if (sessions.length === 0) {
        sessionList.innerHTML = '<div style="padding:16px;color:#555;font-size:12px">暂无会话，点击「新建重建」开始</div>';
        return;
    }
    sessions.forEach(s => {
        const el = document.createElement('div');
        el.className = 'session-item' + (selectedIds.includes(s.id) ? ' selected' : '');
        el.dataset.id = s.id;

        const dot = `<span class="status-dot status-${s.status}"></span>`;
        const date = new Date(s.created_at + 'Z').toLocaleDateString('zh-CN', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
        });
        el.innerHTML = `
            <div class="pid">${dot}${s.patient_id}</div>
            <div class="meta">${date}${s.notes ? ' · ' + s.notes : ''}</div>
        `;
        el.addEventListener('click', () => handleSessionClick(s));
        sessionList.appendChild(el);
    });
}

function handleSessionClick(sess) {
    if (compareMode) {
        // 最多选两个
        if (selectedIds.includes(sess.id)) {
            selectedIds = selectedIds.filter(id => id !== sess.id);
        } else if (selectedIds.length < 2) {
            selectedIds.push(sess.id);
        } else {
            selectedIds = [selectedIds[1], sess.id];
        }
        renderSessionList();
        if (selectedIds.length === 2) {
            loadCompare(selectedIds[0], selectedIds[1]);
        }
    } else {
        selectedIds = [sess.id];
        renderSessionList();
        if (sess.has_model) {
            loadSingleModel(sess.id);
        } else if (sess.status === 'running') {
            showProgressOverlay();
            watchProgress(sess.id);
        } else if (sess.status === 'error') {
            showToast('该会话重建失败，请重新上传', 'error');
        }
    }
}

// ── 单视图加载 ────────────────────────────────────────────────────────────────
async function loadSingleModel(sessionId) {
    if (emptyState) emptyState.style.display = 'none';
    splitContainer.classList.remove('active');
    canvasSingle.style.display = 'block';

    showProgressOverlay('加载 3D 模型...');
    try {
        await viewerSingle.loadGLB(api.modelUrl(sessionId));
        hideProgressOverlay();
    } catch (e) {
        hideProgressOverlay();
        showToast('模型加载失败: ' + e.message, 'error');
    }
}

// ── 分屏对比 ──────────────────────────────────────────────────────────────────
function toggleCompareMode() {
    compareMode = !compareMode;
    btnCompare.classList.toggle('active', compareMode);

    if (!compareMode) {
        // 退出对比模式
        compareInfo.textContent = '';
        splitContainer.classList.remove('active');
        canvasSingle.style.display = 'block';
        selectedIds = selectedIds.slice(0, 1);
        renderSessionList();
        if (viewerA) { viewerA.dispose(); viewerA = null; }
        if (viewerB) { viewerB.dispose(); viewerB = null; }
    } else {
        compareInfo.textContent = '请选择两个会话进行对比';
        selectedIds = [];
        renderSessionList();
    }
}

async function loadCompare(idA, idB) {
    const sessA = sessions.find(s => s.id === idA);
    const sessB = sessions.find(s => s.id === idB);
    if (!sessA?.has_model || !sessB?.has_model) {
        showToast('两个会话都需要有完成的模型', 'error');
        return;
    }

    canvasSingle.style.display = 'none';
    splitContainer.classList.add('active');

    // 重新创建分屏查看器
    if (viewerA) viewerA.dispose();
    if (viewerB) viewerB.dispose();

    viewerA = new Viewer3D(canvasA);
    viewerB = new Viewer3D(canvasB, { syncTarget: true });

    // 同步旋转：A 变化时同步给 B
    viewerA.onCameraChange((cam, target) => {
        viewerB.syncCamera(cam, target);
    });

    // 更新标签
    document.getElementById('label-a').textContent =
        `${sessA.patient_id}${sessA.notes ? ' · ' + sessA.notes : ''}`;
    document.getElementById('label-b').textContent =
        `${sessB.patient_id}${sessB.notes ? ' · ' + sessB.notes : ''}`;

    showProgressOverlay('加载模型...');
    try {
        await Promise.all([
            viewerA.loadGLB(api.modelUrl(idA)),
            viewerB.loadGLB(api.modelUrl(idB)),
        ]);
        hideProgressOverlay();
    } catch (e) {
        hideProgressOverlay();
        showToast('分屏加载失败: ' + e.message, 'error');
        return;
    }

    // 后台 ICP 对齐（可选，如果耗时长则先展示未对齐，完成后再更新）
    compareInfo.textContent = 'ICP 对齐中...';
    try {
        const { transform_matrix } = await api.compare(idA, idB);
        applyTransformToViewer(viewerB, transform_matrix);
        compareInfo.textContent = `对比: ${sessA.patient_id} vs ${sessB.patient_id}（已对齐）`;
    } catch (e) {
        compareInfo.textContent = `对比: ${sessA.patient_id} vs ${sessB.patient_id}（未对齐）`;
        console.warn('ICP 对齐失败:', e.message);
    }
}

function applyTransformToViewer(viewer, matrix4x4) {
    // matrix4x4: [[row0...], [row1...], [row2...], [row3...]]
    // Three.js Matrix4 是列主序，但我们传入行主序，需转置
    if (!viewer._model) return;
    const m = new (viewer._model.matrix.constructor)();
    const flat = matrix4x4.flat();  // 16 elements row-major
    m.set(...flat);                  // Three.js set() 接收行主序 ✓
    viewer._model.applyMatrix4(m);
}

// ── 进度 / WebSocket ──────────────────────────────────────────────────────────
function watchProgress(sessionId) {
    const ws = api.connectProgress(sessionId, (msg) => {
        if (msg.stage === 'ping') return;
        if (msg.stage === 'error') {
            hideProgressOverlay();
            showToast('重建失败: ' + msg.message, 'error');
            refreshSessions();
            ws.close();
        } else if (msg.stage === 'done') {
            hideProgressOverlay();
            showToast('重建完成！', 'success');
            refreshSessions().then(() => {
                loadSingleModel(sessionId);
            });
            ws.close();
        } else {
            updateProgress(msg.pct, msg.message);
        }
    });
}

function showProgressOverlay(msg = '重建中...') {
    progressOverlay.classList.add('visible');
    updateProgress(0, msg);
}
function hideProgressOverlay() {
    progressOverlay.classList.remove('visible');
}
function updateProgress(pct, msg) {
    progressFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    progressMsg.textContent = msg || '';
}

// ── 上传模态框 ────────────────────────────────────────────────────────────────
function showUploadModal() {
    uploadModal.classList.add('visible');
}
function hideUploadModal() {
    uploadModal.classList.remove('visible');
    document.getElementById('upload-form').reset();
}

async function handleUpload(e) {
    e.preventDefault();
    const form = e.target;
    const patientId = form.patient_id.value.trim();
    const notes     = form.notes.value.trim();
    const leftFile  = form.left_image.files[0];
    const frontFile = form.front_image.files[0];
    const rightFile = form.right_image.files[0];

    if (!patientId || !leftFile || !frontFile || !rightFile) {
        showToast('请填写患者ID并上传三张图片', 'error');
        return;
    }

    hideUploadModal();
    showProgressOverlay('上传图片...');

    try {
        const sess = await api.createSession({
            patientId, notes, leftFile, frontFile, rightFile,
        });
        updateProgress(5, '开始重建...');
        sessions.unshift(sess);
        selectedIds = [sess.id];
        renderSessionList();
        watchProgress(sess.id);
    } catch (e) {
        hideProgressOverlay();
        showToast('上传失败: ' + e.message, 'error');
    }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(msg, type = '') {
    toast.textContent = msg;
    toast.className = 'show' + (type ? ' ' + type : '');
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { toast.className = ''; }, 3500);
}

// ── 启动 ──────────────────────────────────────────────────────────────────────
init().catch(console.error);
