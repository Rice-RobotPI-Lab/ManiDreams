"""
Zero-shot Real-to-Sim Demo — ManiDreams integration

RealSense D415 + FFS stereo + SAM2 tracking + Newton physics simulation.

Workflow:
  Phase 1: Select table -> SAM2 segments -> plane fits & locks
  Phase 2: Select object -> SAM2 tracks -> live OBB
  Simulate: Snapshot OBB -> Newton multi-world sim in same Viser view
  Gizmo:   Drag actor in Viser to push objects (teleop)

Left panel:  Live RGB + SAM2 mask (MJPEG stream)
Right panel: Viser 3D — real point cloud + Newton simulation

Usage:
  conda activate ffs
  python examples/tasks/zeroshot_real2sim_demo/main.py
  Open http://localhost:<WEB_PORT> in browser
"""

import os
import sys
import time
import logging
import threading
import socket

import numpy as np
import torch
import yaml
import cv2
import viser
import pyrealsense2 as rs
from flask import Flask, Response, request, jsonify

import warp as wp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# ManiDreams root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MANIDREAMS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
sys.path.insert(0, os.path.join(_MANIDREAMS_ROOT, "src"))
sys.path.insert(0, _MANIDREAMS_ROOT)

# Fast-FoundationStereoPose (FFS) — set via env var or default to sibling directory
FFS_DIR = os.environ.get("FFS_DIR",
    os.path.normpath(os.path.join(_MANIDREAMS_ROOT, "..", "Fast-FoundationStereoPose")))
if not os.path.isdir(FFS_DIR):
    raise RuntimeError(
        f"Fast-FoundationStereoPose not found at: {FFS_DIR}\n"
        f"Clone it:  git clone https://github.com/Vector-Wangel/Fast-FoundationStereoPose.git\n"
        f"Or set:    export FFS_DIR=/path/to/Fast-FoundationStereoPose")
sys.path.insert(0, FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

# SAM2 (bundled inside FFS repo)
SAM2_DIR = os.path.join(FFS_DIR, "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# ManiDreams components
from manidreams.physics.simulation_tsip import SimulationBasedTSIP
from examples.physics.newton_backend import NewtonBackend, DRISSim, make_unit_box, gen_colors
from examples.physics.newton_backend import ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z, ACTOR_HALF
from examples.executors.d415_ffs_executor import (
    D415FFSExecutor, sim_pos_to_cam, sim_quat_to_cam, create_plane_mesh,
    rotmat_to_quat_wxyz, MASK_ALPHA, MASK_COLOR_BGR, TABLE_COLOR_BGR,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"

VALID_ITERS = 8
MAX_DISP = 192
IR_PROJECTOR_ON = True
IMG_WIDTH = 640
IMG_HEIGHT = 480
N_ENVS = 32


def find_free_port(start=9090, end=9200):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError(f"No free port in {start}-{end}")


# ---------------------------------------------------------------------------
# Load models & camera
# ---------------------------------------------------------------------------

def load_models_and_camera():
    """Load FFS, SAM2, RealSense. Returns all needed objects."""
    # GPU config
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # FFS
    logging.info("Loading FFS model...")
    torch.autograd.set_grad_enabled(False)
    with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['valid_iters'] = VALID_ITERS
    cfg['max_disp'] = MAX_DISP
    ffs_model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
    ffs_model.args.valid_iters = VALID_ITERS
    ffs_model.args.max_disp = MAX_DISP
    ffs_model.cuda().eval()
    logging.info("FFS model loaded")

    # SAM2
    logging.info("Loading SAM2 model...")
    sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
    sam2_predictor.fill_hole_area = 0
    logging.info("SAM2 model loaded")

    # RealSense D415
    logging.info("Initializing RealSense D415...")
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
    rs_config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
    rs_config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
    profile = rs_pipeline.start(rs_config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)

    # Intrinsics & extrinsics
    frames = rs_pipeline.wait_for_frames()
    ir_left_frame = frames.get_infrared_frame(1)
    color_frame = frames.get_color_frame()

    ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
    ir_intrinsics = ir_left_profile.get_intrinsics()
    K_ir = np.array([[ir_intrinsics.fx, 0, ir_intrinsics.ppx],
                      [0, ir_intrinsics.fy, ir_intrinsics.ppy],
                      [0, 0, 1]], dtype=np.float32)

    color_profile = color_frame.get_profile().as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    K_color = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                         [0, color_intrinsics.fy, color_intrinsics.ppy],
                         [0, 0, 1]], dtype=np.float32)

    ir_to_color_ext = ir_left_profile.get_extrinsics_to(color_profile)
    R_ir_to_color = np.array(ir_to_color_ext.rotation).reshape(3, 3).astype(np.float32)
    T_ir_to_color = np.array(ir_to_color_ext.translation).astype(np.float32)

    ir_right_frame = frames.get_infrared_frame(2)
    ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
    ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
    baseline = abs(ir_left_to_right.translation[0])
    logging.info(f"Baseline: {baseline*1000:.1f}mm")

    # Warm up FFS
    logging.info("Warming up FFS...")
    dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
    padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
    d0, d1 = padder.pad(dummy, dummy)
    with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
        _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True,
                              optimize_build_volume='pytorch1')
    del dummy, d0, d1
    torch.cuda.empty_cache()
    logging.info("Warm-up complete")

    return {
        'ffs_model': ffs_model,
        'sam2_predictor': sam2_predictor,
        'rs_pipeline': rs_pipeline,
        'K_ir': K_ir,
        'K_color': K_color,
        'R_ir_to_color': R_ir_to_color,
        'T_ir_to_color': T_ir_to_color,
        'baseline': baseline,
    }


# ---------------------------------------------------------------------------
# Flask Web UI
# ---------------------------------------------------------------------------

def build_html_page(viser_port):
    """Build the HTML page string with embedded Viser iframe."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ManiDreams — Zero-shot Real2Sim Demo</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ display: flex; flex-direction: column; height: 100vh; background: #1a1a1a; color: #eee; font-family: system-ui, sans-serif; }}
  .toolbar {{
    padding: 8px 16px; background: #2a2a2a; display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid #444; flex-wrap: wrap;
  }}
  .toolbar button {{
    padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
    font-size: 13px; font-weight: 500; transition: opacity 0.15s;
  }}
  .toolbar button:hover {{ opacity: 0.85; }}
  .btn-table {{ background: #e8a020; color: white; }}
  .btn-point {{ background: #4a9eff; color: white; }}
  .btn-bbox  {{ background: #50c878; color: white; }}
  .btn-sim   {{ background: #9b59b6; color: white; }}
  .btn-reset {{ background: #ff4a4a; color: white; }}
  .btn-active {{ outline: 2px solid #fff; outline-offset: 2px; }}
  .sep {{ width: 1px; height: 24px; background: #555; }}
  #status {{ font-size: 13px; color: #aaa; margin-left: 12px; }}
  .main {{ display: flex; flex: 1; overflow: hidden; }}
  .panel {{ flex: 1; position: relative; overflow: hidden; }}
  .panel-label {{
    position: absolute; top: 8px; left: 12px; z-index: 10;
    background: rgba(0,0,0,0.6); padding: 3px 10px; border-radius: 4px;
    font-size: 12px; color: #ccc; pointer-events: none;
  }}
  #video-wrap {{ position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #111; }}
  #video {{ max-width: 100%; max-height: 100%; display: block; }}
  #overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
  #click-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }}
  iframe {{ width: 100%; height: 100%; border: none; }}
  .divider {{ width: 3px; background: #444; cursor: col-resize; }}
</style>
</head>
<body>
  <div class="toolbar">
    <button class="btn-table" id="btn-table" onclick="setMode('table')">1. Select Table</button>
    <div class="sep"></div>
    <button class="btn-point" id="btn-point" onclick="setMode('point')">2. Select Point</button>
    <button class="btn-bbox" id="btn-bbox" onclick="setMode('bbox')">2. Select BBox</button>
    <div class="sep"></div>
    <button class="btn-sim" id="btn-sim" onclick="startSim()">Simulate</button>
    <button class="btn-sim" id="btn-pause" onclick="pauseSim()" style="background:#e67e22;display:none;">Pause Sim</button>
    <div class="sep"></div>
    <button class="btn-reset" onclick="resetAll()">Reset All</button>
    <span id="status">Step 1: Select table surface</span>
  </div>
  <div class="main">
    <div class="panel" id="left-panel">
      <div class="panel-label">RGB + SAM2</div>
      <div id="video-wrap">
        <img id="video" src="/video">
        <canvas id="overlay"></canvas>
        <div id="click-layer"></div>
      </div>
    </div>
    <div class="divider" id="divider"></div>
    <div class="panel" id="right-panel">
      <div class="panel-label">3D View (Viser)</div>
      <iframe id="viser-frame"></iframe>
    </div>
  </div>
<script>
  document.getElementById('viser-frame').src = 'http://' + window.location.hostname + ':{viser_port}';
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const clickLayer = document.getElementById('click-layer');
  const ctx = overlay.getContext('2d');
  const statusEl = document.getElementById('status');
  let mode = 'table', drawing = false, sx = 0, sy = 0;

  function syncOverlay() {{
    const r = video.getBoundingClientRect();
    overlay.style.left = video.offsetLeft + 'px';
    overlay.style.top = video.offsetTop + 'px';
    overlay.width = r.width; overlay.height = r.height;
    clickLayer.style.left = overlay.style.left; clickLayer.style.top = overlay.style.top;
    clickLayer.style.width = r.width + 'px'; clickLayer.style.height = r.height + 'px';
  }}
  video.onload = syncOverlay;
  window.addEventListener('resize', syncOverlay);
  new ResizeObserver(syncOverlay).observe(video);

  function imgCoords(e) {{
    const r = video.getBoundingClientRect();
    return {{ x: Math.round((e.clientX - r.left) / r.width * {IMG_WIDTH}),
              y: Math.round((e.clientY - r.top) / r.height * {IMG_HEIGHT}) }};
  }}
  function setMode(m) {{
    mode = m;
    document.getElementById('btn-table').classList.toggle('btn-active', m === 'table');
    document.getElementById('btn-point').classList.toggle('btn-active', m === 'point');
    document.getElementById('btn-bbox').classList.toggle('btn-active', m === 'bbox');
  }}
  clickLayer.addEventListener('mousedown', e => {{
    if (mode === 'bbox' || mode === 'table') {{ drawing = true; const p = imgCoords(e); sx = p.x; sy = p.y; }}
  }});
  clickLayer.addEventListener('mousemove', e => {{
    if (!drawing) return;
    const p = imgCoords(e), r = video.getBoundingClientRect();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.strokeStyle = mode === 'table' ? '#e8a020' : '#00ff00';
    ctx.lineWidth = 2; ctx.setLineDash([6,3]);
    ctx.strokeRect(sx/{IMG_WIDTH}*r.width, sy/{IMG_HEIGHT}*r.height,
      (p.x-sx)/{IMG_WIDTH}*r.width, (p.y-sy)/{IMG_HEIGHT}*r.height);
  }});
  clickLayer.addEventListener('mouseup', e => {{
    const p = imgCoords(e);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (mode === 'table') {{
      drawing = false;
      const x1=Math.min(sx,p.x), y1=Math.min(sy,p.y), x2=Math.max(sx,p.x), y2=Math.max(sy,p.y);
      if (Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5)
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'table_bbox',x1,y1,x2,y2}})}});
      else
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'table_point',x:sx,y:sy}})}});
      statusEl.textContent = 'Fitting table plane...';
    }} else if (mode === 'point') {{
      fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{mode:'point',x:p.x,y:p.y}})}});
      statusEl.textContent = 'Tracking object...';
    }} else if (mode === 'bbox' && drawing) {{
      drawing = false;
      const x1=Math.min(sx,p.x), y1=Math.min(sy,p.y), x2=Math.max(sx,p.x), y2=Math.max(sy,p.y);
      if (Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5)
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'bbox',x1,y1,x2,y2}})}});
      statusEl.textContent = 'Tracking object...';
    }}
  }});
  function resetAll() {{
    fetch('/api/reset', {{method:'POST'}});
    statusEl.textContent = 'Reset. Step 1: Select table surface';
    mode = 'table'; setMode('table');
  }}
  const divider = document.getElementById('divider');
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');
  let draggingDiv = false;
  divider.addEventListener('mousedown', () => {{ draggingDiv = true; document.body.style.cursor = 'col-resize'; }});
  document.addEventListener('mousemove', e => {{
    if (!draggingDiv) return;
    const pct = (e.clientX / document.querySelector('.main').clientWidth) * 100;
    leftPanel.style.flex = 'none'; leftPanel.style.width = pct + '%'; rightPanel.style.flex = '1';
    syncOverlay();
  }});
  document.addEventListener('mouseup', () => {{ draggingDiv = false; document.body.style.cursor = ''; }});
  function startSim() {{
    fetch('/api/simulate', {{method:'POST'}}).then(r => r.json()).then(d => {{
      if (d.ok) {{
        statusEl.textContent = 'Newton simulation running (perception paused)';
        document.getElementById('btn-pause').style.display = '';
        document.getElementById('btn-pause').textContent = 'Pause Sim';
      }} else statusEl.textContent = 'Simulate failed: ' + (d.error || 'unknown');
    }});
  }}
  function pauseSim() {{
    fetch('/api/pause_sim', {{method:'POST'}}).then(r => r.json()).then(d => {{
      if (d.ok) {{
        const btn = document.getElementById('btn-pause');
        if (d.paused) {{
          btn.textContent = 'Resume Sim';
          btn.style.background = '#27ae60';
          statusEl.textContent = 'Sim paused — perception active';
        }} else {{
          btn.textContent = 'Pause Sim';
          btn.style.background = '#e67e22';
          statusEl.textContent = 'Newton simulation running (perception paused)';
        }}
      }}
    }});
  }}
  setMode('table');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ===== 1. Load models & camera =====
    models = load_models_and_camera()
    wp.init()

    WEB_PORT = find_free_port(9090, 9200)
    VISER_PORT = find_free_port(WEB_PORT + 1, 9200)

    # ===== 2. Create Executor =====
    executor = D415FFSExecutor(
        ffs_model=models['ffs_model'],
        sam2_predictor=models['sam2_predictor'],
        rs_pipeline=models['rs_pipeline'],
        K_ir=models['K_ir'],
        K_color=models['K_color'],
        R_ir_to_color=models['R_ir_to_color'],
        T_ir_to_color=models['T_ir_to_color'],
        baseline=models['baseline'],
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        input_padder_cls=InputPadder,
        amp_dtype=AMP_DTYPE,
        valid_iters=VALID_ITERS,
    )
    executor.initialize({})

    # ===== 3. Viser server =====
    viser_server = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
    viser_server.scene.set_up_direction("-y")

    # ===== 4. Shared state =====
    lock = threading.Lock()
    latest_jpeg = [None]  # mutable container for thread sharing
    pending_action = [None]
    need_reset = [False]
    pending_simulate = [False]

    # Simulation state
    tsip = [None]             # SimulationBasedTSIP (recreated each Simulate)
    sim_running = [False]
    sim_paused = [False]
    sim_params = [None]       # randomize_envs output
    sim_boxes_handle = [None]
    sim_actors_handle = [None]
    sim_gizmo_handle = [None]
    pending_actor_target = [None]
    plane_handle = [None]
    obb_handle = [None]

    # ===== 5. Flask Web UI =====
    HTML_PAGE = build_html_page(VISER_PORT)
    app = Flask(__name__)

    @app.route('/')
    def index():
        return HTML_PAGE

    @app.route('/video')
    def video_feed():
        def generate():
            while True:
                with lock:
                    jpeg = latest_jpeg[0]
                if jpeg is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                time.sleep(0.033)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/select', methods=['POST'])
    def api_select():
        with lock:
            pending_action[0] = request.json
        return jsonify(ok=True)

    @app.route('/api/reset', methods=['POST'])
    def api_reset():
        need_reset[0] = True
        return jsonify(ok=True)

    @app.route('/api/simulate', methods=['POST'])
    def api_simulate():
        if not executor.plane_locked:
            return jsonify(ok=False, error="Table plane not locked yet")
        if executor.obb_smooth_extent is None:
            return jsonify(ok=False, error="No OBB tracked yet")
        pending_simulate[0] = True
        return jsonify(ok=True)

    @app.route('/api/pause_sim', methods=['POST'])
    def api_pause_sim():
        if not sim_running[0]:
            return jsonify(ok=False, error="No simulation running")
        sim_paused[0] = not sim_paused[0]
        return jsonify(ok=True, paused=sim_paused[0])

    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=WEB_PORT, threaded=True),
        daemon=True).start()
    logging.info(f"Web UI:  http://localhost:{WEB_PORT}")
    logging.info(f"Viser:   http://localhost:{VISER_PORT}")

    # ===== 6. Helper: start Newton sim =====
    def start_newton_sim():
        obb_sim = executor.get_obb_sim()
        if obb_sim is None:
            return
        ctx = executor.get_transform_context()

        logging.info(
            f"Starting Newton sim: half_ext="
            f"({obb_sim['half_extents'][0]*100:.2f}, "
            f"{obb_sim['half_extents'][1]*100:.2f}, "
            f"{obb_sim['half_extents'][2]*100:.2f})cm")

        # Create TSIP with Newton backend
        backend = NewtonBackend()
        env_config = {
            'half_extents': obb_sim['half_extents'],
            'quat_wxyz': obb_sim['quat_wxyz'],
            'z_sim': obb_sim['z_sim'],
            'obj_xy_sim': obb_sim['obj_xy_sim'],
            'n_envs': N_ENVS,
        }
        tsip[0] = SimulationBasedTSIP(backend=backend, env_config=env_config)
        sim_env = tsip[0].env  # DRISSim instance
        sim_params[0] = sim_env.params

        # Get initial state for visualization
        box_pos, box_quat, actor_pos, actor_quat = sim_env.get_state()
        R_c2s = ctx['R_c2s']
        plane_center = ctx['plane_center']

        box_pos_cam = sim_pos_to_cam(box_pos, R_c2s, plane_center)
        box_quat_cam = sim_quat_to_cam(box_quat, R_c2s)
        actor_pos_cam = sim_pos_to_cam(actor_pos, R_c2s, plane_center)
        actor_quat_cam = sim_quat_to_cam(actor_quat, R_c2s)

        box_verts, box_faces = make_unit_box()
        scales = (sim_params[0]["half_extents"] * 2).astype(np.float32)
        colors = gen_colors(N_ENVS)

        sim_boxes_handle[0] = viser_server.scene.add_batched_meshes_simple(
            name="/sim/boxes", vertices=box_verts, faces=box_faces,
            batched_positions=box_pos_cam, batched_wxyzs=box_quat_cam,
            batched_scales=scales, batched_colors=colors, opacity=0.5)

        actor_scales = np.tile(
            np.array([ACTOR_HALF * 2] * 3, dtype=np.float32), (N_ENVS, 1))
        sim_actors_handle[0] = viser_server.scene.add_batched_meshes_simple(
            name="/sim/actors", vertices=box_verts, faces=box_faces,
            batched_positions=actor_pos_cam, batched_wxyzs=actor_quat_cam,
            batched_scales=actor_scales, batched_colors=colors, opacity=0.5)

        # Actor control gizmo
        actor_init_sim = np.array([ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z], dtype=np.float32)
        gizmo_pos_cam = sim_pos_to_cam(actor_init_sim.reshape(1, 3), R_c2s, plane_center)[0]
        gizmo_quat = rotmat_to_quat_wxyz(R_c2s.T)

        if sim_gizmo_handle[0] is not None:
            sim_gizmo_handle[0].remove()

        sim_gizmo_handle[0] = viser_server.scene.add_transform_controls(
            "/sim/actor_gizmo", scale=0.08,
            position=tuple(gizmo_pos_cam.tolist()),
            wxyz=tuple(gizmo_quat.tolist()),
            active_axes=(True, True, True),
            disable_rotations=True, disable_sliders=True,
            depth_test=False, line_width=3.0)

        def _on_gizmo_update(_event):
            pos_cam = np.array(sim_gizmo_handle[0].position, dtype=np.float64)
            pos_sim = R_c2s @ (pos_cam - plane_center)
            pending_actor_target[0] = (float(pos_sim[0]), float(pos_sim[1]), float(pos_sim[2]))

        sim_gizmo_handle[0].on_update(_on_gizmo_update)

        sim_running[0] = True
        sim_paused[0] = False
        pending_actor_target[0] = None
        logging.info("Newton simulation started (perception paused to save VRAM)")

    # ===== 7. Helper: cleanup sim visuals =====
    def cleanup_sim_visuals():
        for h in [sim_boxes_handle, sim_actors_handle, sim_gizmo_handle]:
            if h[0] is not None:
                h[0].remove()
                h[0] = None

    # ===== 8. Helper: reset all =====
    def reset_all():
        executor.reset()
        with lock:
            pending_action[0] = None
        pending_simulate[0] = False
        sim_running[0] = False
        sim_paused[0] = False
        tsip[0] = None
        sim_params[0] = None
        pending_actor_target[0] = None
        cleanup_sim_visuals()
        if plane_handle[0] is not None:
            plane_handle[0].remove()
            plane_handle[0] = None
        if obb_handle[0] is not None:
            obb_handle[0].remove()
            obb_handle[0] = None
        logging.info("Reset all")

    # ===== 9. Main loop =====
    frame_count = 0

    try:
        while True:
            t0 = time.time()

            # --- Always capture frames (keep 2D display live during sim) ---
            ir_left, ir_right, color_bgr = executor.capture_frames()
            executor.last_color_bgr = color_bgr

            # --- Reset ---
            if need_reset[0]:
                need_reset[0] = False
                reset_all()

            # --- Handle simulate request ---
            if pending_simulate[0]:
                pending_simulate[0] = False
                if executor.plane_locked and executor.obb_smooth_extent is not None:
                    if sim_running[0]:
                        sim_running[0] = False
                        tsip[0] = None
                        cleanup_sim_visuals()
                    start_newton_sim()

            # --- Handle pending UI action ---
            with lock:
                action = pending_action[0]
                pending_action[0] = None

            if action is not None:
                act_mode = action['mode']

                if act_mode in ('table_point', 'table_bbox') and not executor.plane_locked:
                    executor.add_prompt(act_mode, color_bgr, **action)
                elif act_mode in ('point', 'bbox') and executor.plane_locked:
                    # Stop any running sim
                    if sim_running[0]:
                        sim_running[0] = False
                        tsip[0] = None
                        cleanup_sim_visuals()
                    executor.add_prompt(act_mode, color_bgr, **action)

            # --- VRAM mutex: skip perception when sim is active ---
            run_perception = not (sim_running[0] and not sim_paused[0])

            # --- Perception ---
            if run_perception:
                dris = executor.get_obs(captured_frames=(ir_left, ir_right, color_bgr))

                # Viser: point cloud
                if len(executor.last_points) > 0:
                    viser_server.scene.add_point_cloud(
                        "/point_cloud", points=executor.last_points,
                        colors=executor.last_colors, point_size=0.002,
                        point_shape="rounded")

                # Viser: table plane (on lock)
                if executor.plane_locked and plane_handle[0] is None:
                    ctx = executor.get_transform_context()
                    verts, faces = create_plane_mesh(
                        ctx['plane_normal'], ctx['plane_d'], ctx['plane_center'])
                    plane_handle[0] = viser_server.scene.add_mesh_simple(
                        "/table_plane", vertices=verts, faces=faces,
                        color=(75, 150, 203), opacity=0.35, flat_shading=True)

                # Viser: OBB wireframe
                obb = executor.get_current_obb()
                if obb is not None:
                    ext = obb['extent']
                    corners_local = np.array([
                        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
                    ], dtype=np.float64) * (ext / 2)
                    corners_w = corners_local @ obb['R'].T + obb['center']
                    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
                             [4, 5], [5, 6], [6, 7], [7, 4],
                             [0, 4], [1, 5], [2, 6], [3, 7]]
                    seg = np.array([[corners_w[a], corners_w[b]] for a, b in edges],
                                   dtype=np.float32)
                    obb_handle[0] = viser_server.scene.add_line_segments(
                        "/obb", points=seg,
                        colors=np.full((len(edges), 2, 3), [0, 255, 0], dtype=np.uint8))
                elif obb_handle[0] is not None and executor.phase != 'object':
                    obb_handle[0].remove()
                    obb_handle[0] = None

            # --- Newton simulation step ---
            if sim_running[0] and tsip[0] is not None and not sim_paused[0]:
                sim_env = tsip[0].env  # DRISSim

                # Apply gizmo target
                if pending_actor_target[0] is not None:
                    t = pending_actor_target[0]
                    sim_env.set_actor_target(t[0], t[1], t[2])
                    pending_actor_target[0] = None

                box_pos, box_quat, actor_pos, actor_quat = sim_env.step()

                ctx = executor.get_transform_context()
                R_c2s = ctx['R_c2s']
                plane_center = ctx['plane_center']

                sim_boxes_handle[0].batched_positions = sim_pos_to_cam(box_pos, R_c2s, plane_center)
                sim_boxes_handle[0].batched_wxyzs = sim_quat_to_cam(box_quat, R_c2s)
                sim_actors_handle[0].batched_positions = sim_pos_to_cam(actor_pos, R_c2s, plane_center)
                sim_actors_handle[0].batched_wxyzs = sim_quat_to_cam(actor_quat, R_c2s)

            # --- 2D display (always use fresh frame from capture above) ---
            display = color_bgr.copy()

            # Overlay masks
            if executor.table_mask is not None and np.any(executor.table_mask):
                ov = display.copy()
                ov[executor.table_mask > 0] = TABLE_COLOR_BGR
                display = cv2.addWeighted(display, 1 - MASK_ALPHA, ov, MASK_ALPHA, 0)
                contours, _ = cv2.findContours(
                    executor.table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 200, 255), 2)

            if executor.current_mask is not None and np.any(executor.current_mask):
                ov = display.copy()
                ov[executor.current_mask > 0] = MASK_COLOR_BGR
                display = cv2.addWeighted(display, 1 - MASK_ALPHA, ov, MASK_ALPHA, 0)
                contours, _ = cv2.findContours(
                    executor.current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

            # Status text
            if not executor.plane_locked and executor.phase == 'table':
                cv2.putText(display,
                            f"Plane: frame={executor.plane_frame_count}",
                            (10, IMG_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            if executor.plane_locked:
                cv2.putText(display, "Table: LOCKED", (10, IMG_HEIGHT - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            if executor.obb_smooth_extent is not None and executor.phase == 'object':
                ext = executor.obb_smooth_extent
                cv2.putText(display,
                            f"OBB: {ext[0]*100:.1f}x{ext[1]*100:.1f}x{ext[2]*100:.1f}cm "
                            f"f={executor.extent_frame_count}",
                            (10, IMG_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if sim_running[0] and not sim_paused[0] and tsip[0] is not None:
                cv2.putText(display, f"SIM: step={tsip[0].env.step_count}",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            elif sim_running[0] and sim_paused[0] and tsip[0] is not None:
                cv2.putText(display, f"SIM: PAUSED (step={tsip[0].env.step_count})",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

            t1 = time.time()
            fps = 1.0 / max(t1 - t0, 1e-6)

            phase_str = executor.phase.upper() if executor.phase != 'idle' else "READY"
            if sim_running[0] and not sim_paused[0]:
                phase_str = "SIM"
            elif sim_running[0] and sim_paused[0]:
                phase_str = "SIM-PAUSED"
            cv2.putText(display, f"[{phase_str}] FPS: {fps:.1f}",
                        (IMG_WIDTH - 200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            _, jpeg_buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with lock:
                latest_jpeg[0] = jpeg_buf.tobytes()

            frame_count += 1
            if frame_count % 30 == 0:
                logging.info(
                    f"Frame {frame_count}, FPS: {fps:.1f}, "
                    f"phase: {executor.phase}, sim_paused: {sim_paused[0]}")

    except KeyboardInterrupt:
        pass
    finally:
        models['rs_pipeline'].stop()
        logging.info("Exited")


if __name__ == "__main__":
    main()
