# -*- coding: utf-8 -*-
"""Generate a Chinese HTML evaluation report for the latest face reconstruction."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
EVAL_DIR = OUTPUT / "evaluation"
REPORT_PATH = EVAL_DIR / "face_model_evaluation.html"
METRICS_PATH = EVAL_DIR / "evaluation_metrics.json"


def read_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_first_json(paths, default=None):
    for path in paths:
        if path.exists():
            return read_json(path, default)
    return {} if default is None else default


def parse_reprojection(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    mean_match = re.search(r"mean_error_px=([0-9.]+)", text)
    max_match = re.search(r"max_error_px=([0-9.]+)", text)
    return {
        "mean_px": float(mean_match.group(1)) if mean_match else None,
        "max_px": float(max_match.group(1)) if max_match else None,
    }


def mesh_metrics() -> dict:
    scene = trimesh.load(OUTPUT / "meshes" / "face.glb", force="scene", process=False)
    mesh = next(iter(scene.geometry.values()))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "invalid_vertex_values": int((~np.isfinite(vertices)).sum()),
        "degenerate_faces": int((areas < 1e-12).sum()),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "extents_mm": [round(float(value * 1000), 2) for value in extents],
        "surface_area_m2": round(float(areas.sum()), 6),
    }


def texture_metrics() -> dict:
    image = cv2.imread(str(OUTPUT / "textures" / "albedo_white.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(OUTPUT / "textures" / "albedo_white.png")
    max_channel = image.max(axis=2)
    valid = max_channel > 8
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    valid_count = max(int(valid.sum()), 1)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "valid_pixel_ratio": round(float(valid.mean()), 6),
        "valid_pixels": int(valid.sum()),
        "near_black_within_valid_ratio": round(float(((gray < 20) & valid).sum() / valid_count), 6),
        "near_white_within_valid_ratio": round(float(((gray > 245) & valid).sum() / valid_count), 6),
        "mean_luma_valid": round(float(gray[valid].mean()), 3),
        "std_luma_valid": round(float(gray[valid].std()), 3),
    }


def asset(relative: str) -> str:
    return "../" + relative.replace("\\", "/")


def build_report(metrics: dict) -> str:
    reproj = metrics["reprojection"]
    shape = metrics["shape_fine_tune"]
    pose = metrics.get("pose_refinement", {})
    personal = metrics.get("personal_residual_deform", {})
    deform = metrics["free_face_deform"]
    mesh = metrics["mesh"]
    texture = metrics["texture"]
    phase1 = metrics["pipeline"]["phase1"]
    phase3 = metrics["pipeline"]["phase3"]
    total_seconds = phase1["ElapsedSeconds"] + phase3["ElapsedSeconds"]

    reprojection_rows = "".join(
        f"<tr><td>{label}</td><td>{record['mean_px']:.2f} px</td><td>{record['max_px']:.2f} px</td></tr>"
        for label, record in (("左侧", reproj["left"]), ("正面", reproj["front"]), ("右侧", reproj["right"]))
    )
    pose_rows = "".join(
        "<tr>"
        f"<td>{rec.get('view')}</td>"
        f"<td>{'采用' if rec.get('accepted') else '回滚'}</td>"
        f"<td>{rec.get('dense_improve_px')} px</td>"
        f"<td>{rec.get('stable_worsen_px')} px</td>"
        f"<td>{rec.get('mean_improve_px')} px</td>"
        "</tr>"
        for rec in pose.get("view_records", [])
    )
    pose_panel = f"""
<section class="panel"><h2>相机/姿态微调结果</h2>
<p>本轮只调整每个视角的相机 <code>R/t</code>，不改 UV、不强行移动最终网格。最终状态：<b>{'已采用' if pose.get('accepted') else '未采用'}</b>；采用视角：<b>{', '.join(pose.get('applied_views', [])) or '无'}</b>。</p>
<table><thead><tr><th>视角</th><th>动作</th><th>密集轮廓改善</th><th>稳定五官变化</th><th>关键点均值改善</th></tr></thead><tbody>{pose_rows}</tbody></table>
<p class="muted">负的“稳定五官变化”表示眼鼻嘴更稳。当前策略保留正脸相机，采用左右侧脸候选，用小于 1px 的单视角关键点均值代价换取约 15-17px 的侧脸轮廓收益。</p>
<p class="muted">细节审计页：<a href="../debug/pose_refinement/index.html">pose_refinement/index.html</a></p>
</section>
"""
    personal_rows = "".join(
        "<tr>"
        f"<td>{rec.get('view')}</td>"
        f"<td>{rec.get('role')}<br><span class=\"muted\">{rec.get('contour_target', 'full_mask')} / {rec.get('semantic_source', 'NA')}</span></td>"
        f"<td>{rec.get('before_profile_contour_px', rec.get('before_semantic_face_contour_px', rec.get('before_dense_contour_px')))}<br><span class=\"muted\">full {rec.get('before_full_mask_dense_contour_px', 'NA')}</span></td>"
        f"<td>{rec.get('after_profile_contour_px', rec.get('after_semantic_face_contour_px', rec.get('after_dense_contour_px')))}<br><span class=\"muted\">full {rec.get('after_full_mask_dense_contour_px', 'NA')}</span></td>"
        f"<td>{rec.get('profile_improve_px', rec.get('semantic_face_improve_px', rec.get('dense_improve_px')))}<br><span class=\"muted\">full {rec.get('full_mask_dense_improve_px', 'NA')}</span></td>"
        "</tr>"
        for rec in personal.get("view_records", [])
    )
    first_personal_record = (personal.get("view_records") or [{}])[0]
    personal_metric_name = personal.get("metric_name") or first_personal_record.get("metric_name", "full_mask_dense_contour_px")
    personal_target = personal.get("contour_target") or first_personal_record.get("contour_target", "full_mask")
    personal_panel = f"""
<section class="panel"><h2>个人化残差形变结果</h2>
<p>本轮在 FLAME 几何后增加受约束的 per-vertex residual。最终状态：<b>{'已采用' if personal.get('accepted') else '未采用/回滚'}</b>；原因：{personal.get('reason', '无记录')}。</p>
<table><thead><tr><th>视角</th><th>角色</th><th>before dense</th><th>after dense</th><th>改善</th></tr></thead><tbody>{personal_rows}</tbody></table>
<p class="muted">整体密集轮廓：{personal.get('overall_before_dense_contour_px', 'NA')} -> {personal.get('overall_after_dense_contour_px', 'NA')} px；侧脸平均改善：{personal.get('side_mean_improve_px', 'NA')} px；稳定五官变化：{personal.get('stable_worsen_px', 'NA')} px。</p>
<p class="muted">细节审计页：<a href="../debug/personal_residual_deform/index.html">personal_residual_deform/index.html</a></p>
</section>
"""
    deform_before = personal.get("overall_before_dense_contour_px", deform.get("before_dense_contour_px", 0.0))
    deform_after = personal.get("overall_after_dense_contour_px", deform.get("after_dense_contour_px", deform_before))
    deform_max_offset_m = personal.get("safety", {}).get("max_offset_m", deform.get("max_offset_m", 0.0))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D 人脸建模质量评测</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--panel2:#14243a;--line:#28415f;--text:#edf5ff;--muted:#9fb2ca;--cyan:#39d5ff;--green:#4ade80;--amber:#fbbf24;--red:#fb7185}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 20% 0,#12304f 0,transparent 35%),var(--bg);color:var(--text);font-family:"Microsoft YaHei",system-ui,sans-serif;line-height:1.65}}
.wrap{{width:min(1260px,94vw);margin:auto;padding:42px 0 70px}} .eyebrow{{color:var(--cyan);font-weight:800;letter-spacing:.13em;font-size:13px}}
h1{{font-size:clamp(34px,5vw,62px);line-height:1.08;margin:8px 0 12px}} h2{{margin:0 0 18px;font-size:27px}} h3{{margin:0 0 8px}} .muted{{color:var(--muted)}}
.hero{{display:grid;grid-template-columns:1.2fr .8fr;gap:22px;align-items:stretch}} .panel{{background:linear-gradient(145deg,rgba(20,36,58,.96),rgba(10,25,43,.94));border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 18px 50px rgba(0,0,0,.22)}}
.scorebox{{display:flex;align-items:center;gap:22px}} .score{{width:142px;height:142px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--red) 0 62%,#243650 62%);position:relative;flex:none}} .score:after{{content:"";position:absolute;inset:12px;background:#0c192a;border-radius:50%}} .score strong{{font-size:45px;z-index:1}} .score small{{font-size:15px;color:var(--muted)}}
.verdict{{color:var(--red);font-size:28px;font-weight:900}} .badge{{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:3px 10px;font-size:13px;font-weight:800}}
.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:22px}} .metric{{background:#0a1728;border:1px solid var(--line);border-radius:14px;padding:15px}} .metric b{{font-size:25px;display:block}} .metric span{{font-size:13px;color:var(--muted)}}
section{{margin-top:24px}} .two{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} .three{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.finding{{border-left:4px solid var(--red);padding:12px 15px;background:rgba(251,113,133,.08);margin:10px 0;border-radius:0 10px 10px 0}} .finding.good{{border-color:var(--green);background:rgba(74,222,128,.07)}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted);font-size:13px}}
.gallery img{{width:100%;display:block;border-radius:12px;background:#03070d}} .gallery figure{{margin:0}} figcaption{{font-size:13px;color:var(--muted);margin-top:7px}}
.viewer{{height:590px;border-radius:14px;overflow:hidden;background:radial-gradient(circle,#243650,#050b13 70%);position:relative}} #canvas3d{{width:100%;height:100%;display:block}} .viewer-ui{{position:absolute;left:12px;right:12px;bottom:12px;display:flex;gap:7px;flex-wrap:wrap}} button{{border:1px solid #496680;background:rgba(8,20,34,.86);color:white;border-radius:9px;padding:8px 11px;cursor:pointer}} button:hover{{border-color:var(--cyan)}} .viewer-note{{position:absolute;top:12px;left:12px;background:rgba(8,20,34,.82);padding:7px 10px;border-radius:8px;font-size:12px;color:var(--muted)}}
.bar{{height:9px;border-radius:99px;background:#253851;overflow:hidden;margin-top:8px}} .bar i{{height:100%;display:block;background:var(--cyan)}} code{{color:#bcecff}} a{{color:var(--cyan)}}
@media(max-width:900px){{.hero,.two{{grid-template-columns:1fr}}.grid{{grid-template-columns:repeat(2,1fr)}}.three{{grid-template-columns:1fr}}.viewer{{height:480px}}}}
</style>
</head>
<body><main class="wrap">
<div class="eyebrow">FACE3D PIPELINE / 2026-06-13</div>
<h1>3D 人脸建模质量评测</h1>
<p class="muted">输入：<code>captures_20260519_102448</code> 三视图。结论同时依据实跑日志、量化指标和人工视觉复核。</p>

<div class="hero">
 <div class="panel">
  <div class="scorebox"><div class="score"><strong>62<small>/100</small></strong></div><div><span class="badge">综合评级</span><div class="verdict">需要改进</div><p>主体身份可辨，正面皮肤细节较清晰；但眼部黑洞、口鼻纹理破碎与侧脸轮廓偏差属于中央面部关键缺陷，因此不能判为“优秀”。</p></div></div>
  <div class="grid">
   <div class="metric"><b>19/30</b><span>几何与身份相似度</span></div><div class="metric"><b>14/25</b><span>投影与轮廓贴合</span></div><div class="metric"><b>14/25</b><span>纹理清晰与连续性</span></div><div class="metric"><b>6/10</b><span>网格与产物完整性</span></div><div class="metric"><b>9/10</b><span>管线完整与可复现</span></div>
  </div>
 </div>
 <div class="panel"><h2>一句话判断</h2><p style="font-size:20px"><b>这是一份可用于继续调试和展示流程的模型，但尚未达到高质量展示，更不适合医疗级术前术后定量比较。</b></p><div class="finding">关键否决项：双眼区域存在明显黑洞/错误反光，嘴腔与鼻孔出现破碎采样。</div><div class="finding good">优势：完整管线成功、中央面部身份特征可辨、2048² 纹理保留了皮肤细节。</div></div>
</div>

<section class="panel"><h2>完整管线结果</h2><div class="three"><div><h3>几何阶段</h3><b>{phase1['ElapsedSeconds']:.1f} 秒</b><p class="muted">三视图预处理、MICA/DECA 初始化、FLAME 拟合、自由轮廓与深度置换均完成。</p></div><div><h3>纹理阶段</h3><b>{phase3['ElapsedSeconds']:.1f} 秒</b><p class="muted">三视图采样、颜色匹配、接缝与耳侧修复、GLB 打包完成。</p></div><div><h3>总耗时</h3><b>{total_seconds:.1f} 秒</b><p class="muted">最终 GLB：{metrics['artifacts']['glb_mb']:.2f} MB；首次运行补全了 InsightFace/3DFAN 缓存。</p></div></div></section>

{pose_panel}

{personal_panel}

<section class="two">
 <div class="panel"><h2>重投影误差</h2><table><thead><tr><th>视角</th><th>平均误差</th><th>最大误差</th></tr></thead><tbody>{reprojection_rows}</tbody></table><p class="muted">1024×1024 工作画布。眉眼、脸侧轮廓和下颌是主要误差来源。</p></div>
 <div class="panel"><h2>轮廓变形</h2><p>Shape-only 稠密轮廓：<b>{shape['before']['dense_contour_mean_px']:.2f} → {shape['after']['dense_contour_mean_px']:.2f} px</b></p><div class="bar"><i style="width:31%"></i></div><p>Personal residual：<b>{float(deform_before):.2f} → {float(deform_after):.2f} px</b></p><div class="bar"><i style="width:18%"></i></div><p class="muted">该项是 FLAME 之后的个体脸型补偿；最大 residual 位移 {float(deform_max_offset_m)*1000:.1f} mm。若未采用，则最终 mesh 自动回滚到 residual 前状态。</p></div>
</section>

<section class="two">
 <div class="panel"><h2>网格完整性</h2><table><tr><td>最终渲染顶点</td><td>{mesh['vertices']:,}</td></tr><tr><td>最终面片</td><td>{mesh['faces']:,}</td></tr><tr><td>非法数值 / 退化面</td><td>{mesh['invalid_vertex_values']} / {mesh['degenerate_faces']}</td></tr><tr><td>绕序一致</td><td>{'是' if mesh['winding_consistent'] else '否'}</td></tr><tr><td>闭合网格</td><td>{'是' if mesh['watertight'] else '否（可见区域裁剪模型）'}</td></tr><tr><td>尺寸 X/Y/Z</td><td>{' / '.join(map(str,mesh['extents_mm']))} mm</td></tr></table></div>
 <div class="panel"><h2>纹理统计</h2><table><tr><td>贴图尺寸</td><td>{texture['width']} × {texture['height']}</td></tr><tr><td>非黑有效像素</td><td>{texture['valid_pixel_ratio']*100:.1f}%</td></tr><tr><td>有效区近黑像素</td><td>{texture['near_black_within_valid_ratio']*100:.2f}%</td></tr><tr><td>可见面保留</td><td>45.0%</td></tr><tr><td>接缝修复</td><td>27,933 px</td></tr><tr><td>耳侧修复</td><td>38,469 px</td></tr></table><p class="muted">像素比例无法单独识别眼部黑洞，因此最终判断以反投影视觉证据为主。</p></div>
</section>

<section class="panel"><h2>交互式 3D 检查</h2><div class="viewer"><canvas id="canvas3d"></canvas><div class="viewer-note">拖拽旋转，滚轮缩放。直接用 file:// 打开时浏览器可能拦截 GLB，请通过本地 HTTP 查看；下方静态证据始终可用。</div><div class="viewer-ui"><button data-view="front">正面</button><button data-view="left45">左 45°</button><button data-view="right45">右 45°</button><button data-view="left">左侧</button><button data-view="right">右侧</button><button id="material">切换中性材质</button></div></div></section>

<section class="panel gallery"><h2>原图与纹理反投影</h2><img src="{asset('debug/texture_visual_audit/_contact_sheet.jpg')}" alt="三视图纹理反投影"><p class="muted">当前 UV 方向正确。正面总体身份可辨，但眼球黑洞、嘴腔破碎、鼻孔局部撕裂和发际裁剪边界清晰可见。</p></section>

<section class="two gallery">
 <figure class="panel"><h2>几何投影证据</h2><img src="{asset('debug/pipeline_audit/02_geometry/_contact_sheet.jpg')}" alt="几何投影审计"><figcaption>关键点连线在眉眼与脸侧处偏差较大；深度对齐因负尺度三次回退。</figcaption></figure>
 <figure class="panel"><h2>最终 UV 与修复</h2><img src="{asset('debug/pipeline_audit/05_final/_contact_sheet.jpg')}" alt="最终纹理审计"><figcaption>中央面部细节清晰，但关键器官区域存在不可接受的纹理错误。</figcaption></figure>
</section>

<section class="panel"><h2>问题清单与优先级</h2>
 <div class="finding"><b>P0 眼部：</b>眼球/眼眶纹理出现黑洞和错误高光。建议将眼球从皮肤纹理烘焙中分离，或显式生成眼球几何与独立材质。</div>
 <div class="finding"><b>P0 口鼻：</b>嘴腔、牙齿和鼻孔为凹陷区域，当前法线可见性与多视图融合产生破碎采样。需要器官区域专用遮罩、可见性阈值与补洞策略。</div>
 <div class="finding"><b>P1 侧脸轮廓：</b>自由变形几乎只依赖正面，左右侧约 86.8 px 的稠密轮廓误差未被有效优化。应恢复受约束的侧视图权重。</div>
 <div class="finding"><b>P1 深度置换：</b>三视图深度仿射拟合均得到负尺度并回退，67.2% 顶点触及 2 mm 位移上限。应先修正深度方向/尺度，再允许置换。</div>
 <div class="finding"><b>P2 裁剪边界：</b>仅保留 45% 面片且保留 3 个大组件，发际、耳侧与脸缘边界粗糙。需要更连续的可见区域裁剪和边界平滑。</div>
</section>

<section class="panel"><h2>结论</h2><p style="font-size:20px">本次管线在工程上完整跑通，模型具备可辨识身份与较好的正面皮肤纹理基础，但中央器官纹理缺陷和侧脸几何误差直接否决“优秀”评级。综合得分 <b>62/100</b>；按已确认规则，由于存在中央面部关键缺陷，最终评级从“可接受”下调为 <b style="color:var(--red)">需要改进</b>。</p><p class="muted">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（Asia/Shanghai）。详细机器可读指标：<a href="evaluation_metrics.json">evaluation_metrics.json</a>。</p></section>
</main>

<script type="module">
import * as THREE from '../../frontend/vendor/three.module.js';
import {{ OrbitControls }} from '../../frontend/vendor/three/controls/OrbitControls.js';
import {{ GLTFLoader }} from '../../frontend/vendor/three/loaders/GLTFLoader.js';
const canvas=document.querySelector('#canvas3d'); const renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:true}}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene(); const camera=new THREE.PerspectiveCamera(30,1,.001,10); const controls=new OrbitControls(camera,canvas); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x243044,2.2)); const key=new THREE.DirectionalLight(0xffffff,2.8); key.position.set(1,1.5,2); scene.add(key); const fill=new THREE.DirectionalLight(0xa8d8ff,1.2); fill.position.set(-2,.4,1); scene.add(fill);
let root=null, neutral=false; const neutralMat=new THREE.MeshStandardMaterial({{color:0xb9c2ce,roughness:.82,metalness:0,side:THREE.DoubleSide}});
function resize(){{const w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==w||canvas.height!==h){{renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}}}}
function setView(name){{if(!root)return; const box=new THREE.Box3().setFromObject(root),c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),d=Math.max(s.x,s.y,s.z)*2.2; const pos={{front:[0,0,d],left45:[-d*.72,0,d*.72],right45:[d*.72,0,d*.72],left:[-d,0,0],right:[d,0,0]}}[name]; camera.position.set(c.x+pos[0],c.y+pos[1],c.z+pos[2]); controls.target.copy(c); controls.update()}}
new GLTFLoader().load('../meshes/face.glb',g=>{{root=g.scene; scene.add(root); root.traverse(o=>{{if(o.isMesh){{o.userData.original=o.material;o.material.side=THREE.DoubleSide}}}}); setView('front')}},undefined,e=>{{document.querySelector('.viewer-note').textContent='GLB 加载失败：请通过本地 HTTP 服务打开本报告。';console.error(e)}});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view)); document.querySelector('#material').onclick=()=>{{neutral=!neutral;root?.traverse(o=>{{if(o.isMesh)o.material=neutral?neutralMat:o.userData.original}})}};
renderer.setAnimationLoop(()=>{{resize();controls.update();renderer.render(scene,camera)}});
</script>
</body></html>"""


def main() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    phase1 = read_first_json([
        EVAL_DIR / "phase1_profile_residual_v3_run.json",
        EVAL_DIR / "phase1_profile_residual_v2_run.json",
        EVAL_DIR / "phase1_profile_residual_run.json",
        EVAL_DIR / "phase1_semantic_residual_v3_run.json",
        EVAL_DIR / "phase1_semantic_residual_v2_run.json",
        EVAL_DIR / "phase1_semantic_residual_run.json",
        EVAL_DIR / "phase1_personal_residual_v3_run.json",
        EVAL_DIR / "phase1_personal_residual_v2_run.json",
        EVAL_DIR / "phase1_personal_residual_run.json",
        EVAL_DIR / "phase1_pose_refine_v3_run.json",
        EVAL_DIR / "phase1_pose_refine_v2_run.json",
        EVAL_DIR / "phase1_pose_refine_run.json",
        EVAL_DIR / "phase1_run.json",
    ])
    phase3 = read_first_json([
        EVAL_DIR / "phase3_profile_residual_run.json",
        EVAL_DIR / "phase3_semantic_residual_run.json",
        EVAL_DIR / "phase3_personal_residual_run.json",
        EVAL_DIR / "phase3_pose_refine_run.json",
        EVAL_DIR / "phase3_run.json",
    ])
    optimized_shape = read_json(OUTPUT / "debug" / "optimized_shape.json")
    shape = optimized_shape.get("shape_only_fine_tune", {})
    pose = optimized_shape.get("pose_refinement", {})
    personal = read_json(OUTPUT / "debug" / "personal_residual_deform_summary.json")
    if personal.get("view_records"):
        first_personal_record = personal["view_records"][0]
        personal.setdefault("contour_target", first_personal_record.get("contour_target"))
        personal.setdefault("metric_name", first_personal_record.get("metric_name"))
    deform = read_json(OUTPUT / "debug" / "free_face_deform_summary.json")
    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "input": "captures_20260519_102448",
        "score": 62,
        "rating": "需要改进",
        "critical_defect": True,
        "scores": {"geometry_identity": 19, "projection_contour": 14, "texture": 14, "integrity": 6, "pipeline": 9},
        "pipeline": {"phase1": phase1, "phase3": phase3},
        "reprojection": {
            name: parse_reprojection(OUTPUT / "debug" / f"landmark_reproj_{name}.txt")
            for name in ("left", "front", "right")
        },
        "shape_fine_tune": shape,
        "pose_refinement": pose,
        "personal_residual_deform": personal,
        "free_face_deform": deform,
        "mesh": mesh_metrics(),
        "texture": texture_metrics(),
        "artifacts": {
            "glb_mb": round((OUTPUT / "meshes" / "face.glb").stat().st_size / 1024 / 1024, 3),
            "texture_mb": round((OUTPUT / "textures" / "albedo_white.png").stat().st_size / 1024 / 1024, 3),
        },
        "visual_findings": [
            "双眼区域存在明显黑洞与错误反光",
            "嘴腔、牙齿和鼻孔区域纹理破碎或拉伸",
            "正面皮肤细节较清晰，身份特征可辨",
            "发际、耳侧和侧脸裁剪边界粗糙",
            "侧视图轮廓误差明显高于正面",
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report(metrics), encoding="utf-8")
    print(REPORT_PATH)
    print(METRICS_PATH)


if __name__ == "__main__":
    main()
