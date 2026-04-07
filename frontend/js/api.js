/**
 * api.js — face3D API 客户端
 */

// 优先使用外部注入的 API_BASE，否则使用相对路径（避免代理拦截 localhost）
const API_BASE = window.API_BASE || '';

export const api = {

    async listSessions(patientId = null) {
        const url = patientId
            ? `${API_BASE}/api/sessions?patient_id=${encodeURIComponent(patientId)}`
            : `${API_BASE}/api/sessions`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async getSession(id) {
        const r = await fetch(`${API_BASE}/api/sessions/${id}`);
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async createSession({ patientId, notes, leftFile, frontFile, rightFile }) {
        const form = new FormData();
        form.append('patient_id', patientId);
        form.append('notes', notes || '');
        form.append('left_image', leftFile, leftFile.name);
        form.append('front_image', frontFile, frontFile.name);
        form.append('right_image', rightFile, rightFile.name);

        const r = await fetch(`${API_BASE}/api/sessions`, {
            method: 'POST',
            body: form,
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async deleteSession(id) {
        const r = await fetch(`${API_BASE}/api/sessions/${id}`, { method: 'DELETE' });
        if (!r.ok) throw new Error(await r.text());
    },

    modelUrl(sessionId) {
        return `${API_BASE}/api/sessions/${sessionId}/model`;
    },

    async setCalibration({ fx, fy, cx, cy }) {
        const r = await fetch(`${API_BASE}/api/calibration`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fx, fy, cx, cy }),
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    },

    async compare(sessionIdA, sessionIdB) {
        const r = await fetch(`${API_BASE}/api/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id_a: sessionIdA, session_id_b: sessionIdB }),
        });
        if (!r.ok) throw new Error(await r.text());
        return r.json();  // { transform_matrix: [[...4x4...]] }
    },

    /**
     * 连接 WebSocket 接收进度。
     * onMessage: (msg: {stage, pct, message}) => void
     * 返回 ws 对象（可 .close()）
     */
    connectProgress(sessionId, onMessage) {
        // 如果 API_BASE 是相对路径，从当前页面 origin 推导 ws 地址
        const origin = API_BASE || `${location.protocol}//${location.host}`;
        const wsBase = origin.replace(/^http/, 'ws');
        const ws = new WebSocket(`${wsBase}/ws/${sessionId}`);
        ws.onmessage = (e) => {
            try { onMessage(JSON.parse(e.data)); }
            catch (_) {}
        };
        ws.onerror = (e) => console.warn('WS error', e);
        return ws;
    },
};
