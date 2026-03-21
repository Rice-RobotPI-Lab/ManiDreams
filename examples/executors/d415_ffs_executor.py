"""
D415 + FFS + SAM2 real-world executor for ManiDreams.

Provides real-time perception from an Intel RealSense D415 camera:
  - FFS stereo matching -> depth -> point cloud
  - SAM2 segmentation + tracking (table plane / object)
  - Table plane fitting with auto-locking
  - Object OBB estimation with temporal smoothing
  - Camera-to-sim coordinate transform

Ported from: d415_ffs_realtime_sim.py (standalone demo)

Usage:
    Models (FFS, SAM2) and RealSense pipeline must be loaded externally
    and passed to the constructor, keeping model lifecycle in main.py.
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import torch

from manidreams.base.executor import ExecutorBase
from manidreams.base.dris import DRIS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Perception parameters (from standalone demo)
# ---------------------------------------------------------------------------
ZFAR = 5.0
ZNEAR = 0.16
PCD_STRIDE = 2
MASK_ERODE_KERNEL = 5
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.uint8)
TABLE_COLOR_BGR = [203, 150, 75]
TABLE_COLOR_RGB = np.array([75, 150, 203], dtype=np.uint8)

# OBB stabilization
OBB_SMOOTH = 0.75
EXTENT_WINDOW = 20
EXTENT_ALPHA_INIT = 0.4
EXTENT_ALPHA_MIN = 0.02
EXTENT_ALPHA_DECAY = 0.92
EXTENT_MAX_CHANGE_RATE = 0.05

# Table plane stabilization
PLANE_SMOOTH_INIT = 0.5
PLANE_SMOOTH_MIN = 0.02
PLANE_SMOOTH_DECAY = 0.85
PLANE_LOCK_AFTER = 10
PLANE_LOCK_VAR_THRESH = 1e-6
PLANE_HISTORY_LEN = 10
PLANE_VIS_SIZE = 0.8


# ---------------------------------------------------------------------------
# Coordinate transform utilities
# ---------------------------------------------------------------------------

def compute_R_cam_to_sim(n_cam):
    """Rotation matrix that maps table normal n_cam -> [0,0,1] (Rodrigues)."""
    n = n_cam / np.linalg.norm(n_cam)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, target)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(n, target)
    if sin_a < 1e-6:
        return np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotmat_from_quat_wxyz(q):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def sim_pos_to_cam(pos_sim, R_c2s, plane_center):
    """Transform positions from sim coords (z-up) to camera coords."""
    return (R_c2s.T @ pos_sim.T).T + plane_center


def sim_quat_to_cam(quat_wxyz_batch, R_c2s):
    """Transform quaternions (N,4 wxyz) from sim coords to camera coords."""
    out = np.empty_like(quat_wxyz_batch)
    for i in range(len(quat_wxyz_batch)):
        R_sim = rotmat_from_quat_wxyz(quat_wxyz_batch[i])
        R_cam = R_c2s.T @ R_sim
        out[i] = rotmat_to_quat_wxyz(R_cam)
    return out.astype(np.float32)


def create_plane_mesh(normal, d, center, size=PLANE_VIS_SIZE):
    """Create a quad mesh for visualizing the table plane."""
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        t1 = np.cross(n, np.array([1, 0, 0]))
    else:
        t1 = np.cross(n, np.array([0, 1, 0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    half = size / 2
    corners = np.array([
        center - half * t1 - half * t2, center + half * t1 - half * t2,
        center + half * t1 + half * t2, center - half * t1 + half * t2,
    ], dtype=np.float32)
    return corners, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class D415FFSExecutor(ExecutorBase):
    """Real-world executor: D415 + FFS stereo + SAM2 tracking -> DRIS.

    This executor owns the perception pipeline state (SAM2 tracking,
    plane fitting, OBB smoothing) but does NOT own the models or camera.
    Models and pipeline are passed in from main.py.

    Args:
        ffs_model: Loaded FFS stereo model (on CUDA)
        sam2_predictor: Loaded SAM2 camera predictor
        rs_pipeline: Started RealSense pipeline
        K_ir: (3,3) IR camera intrinsics
        K_color: (3,3) color camera intrinsics
        R_ir_to_color: (3,3) rotation from IR to color frame
        T_ir_to_color: (3,) translation from IR to color frame
        baseline: float, stereo baseline in meters
        img_width: int
        img_height: int
        input_padder_cls: InputPadder class from FFS
        amp_dtype: torch dtype for FFS inference
        valid_iters: int, FFS iteration count
    """

    def __init__(self, ffs_model, sam2_predictor, rs_pipeline,
                 K_ir, K_color, R_ir_to_color, T_ir_to_color, baseline,
                 img_width=640, img_height=480,
                 input_padder_cls=None, amp_dtype=None, valid_iters=8):
        super().__init__(name="D415FFSExecutor")

        # External references (not owned)
        self.ffs_model = ffs_model
        self.sam2_predictor = sam2_predictor
        self.rs_pipeline = rs_pipeline

        # Camera params
        self.K_ir = K_ir
        self.K_color = K_color
        self.R_ir_to_color = R_ir_to_color
        self.T_ir_to_color = T_ir_to_color
        self.baseline = baseline
        self.W = img_width
        self.H = img_height
        self.InputPadder = input_padder_cls
        self.amp_dtype = amp_dtype
        self.valid_iters = valid_iters

        # Derived camera params
        self.fx_ir = K_ir[0, 0]
        self.fy_ir = K_ir[1, 1]
        self.cx_ir = K_ir[0, 2]
        self.cy_ir = K_ir[1, 2]
        u_grid, v_grid = np.meshgrid(
            np.arange(0, img_width, PCD_STRIDE),
            np.arange(0, img_height, PCD_STRIDE))
        self.u_flat = u_grid.reshape(-1).astype(np.float32)
        self.v_flat = v_grid.reshape(-1).astype(np.float32)

        # Perception state
        self._reset_perception_state()

    def _reset_perception_state(self):
        """Reset all mutable perception state."""
        # SAM2
        self.sam2_initialized = False
        self.current_mask = None
        self.current_mask_eroded = None
        self.phase = 'idle'  # 'idle' | 'table' | 'object'

        # Table plane
        self.table_mask = None
        self.table_mask_eroded = None
        self.plane_smooth_normal = None
        self.plane_smooth_d = None
        self.plane_frame_count = 0
        self.plane_normal_history = deque(maxlen=PLANE_HISTORY_LEN)
        self.plane_locked = False
        self.plane_locked_normal = None
        self.plane_locked_d = None
        self.plane_locked_center = None
        self.R_c2s = None

        # OBB
        self.prev_axes = None
        self.obb_smooth_center = None
        self.obb_smooth_extent = None
        self.obb_smooth_R = None
        self.extent_history = deque(maxlen=EXTENT_WINDOW)
        self.extent_frame_count = 0

        # Latest raw frame data (for display in main.py)
        self.last_color_bgr = None
        self.last_points = np.empty((0, 3), dtype=np.float32)
        self.last_colors = np.empty((0, 3), dtype=np.uint8)
        self.last_u_rgb = np.empty(0, dtype=np.int32)
        self.last_v_rgb = np.empty(0, dtype=np.int32)

    # ----- ExecutorBase interface -----

    def initialize(self, config: Dict[str, Any]) -> None:
        self.initialized = True

    def execute(self, actions: Union[Any, List[Any]],
                get_feedback: bool = True) -> Union[Tuple[Any, Dict], Tuple[List[Any], List[Dict]]]:
        """Teleop demo: no physical action execution. Returns current obs."""
        obs = self.get_obs()
        feedback = {"phase": self.phase, "plane_locked": self.plane_locked}
        if isinstance(actions, list):
            return [obs] * len(actions), [feedback] * len(actions)
        return obs, feedback

    def reset(self) -> Any:
        self.sam2_predictor.reset_state()
        self._reset_perception_state()
        logger.info("D415FFSExecutor reset")
        return None

    def get_obs(self) -> Optional[DRIS]:
        """Run one frame of perception: capture -> FFS -> SAM2 -> OBB -> DRIS.

        Returns:
            DRIS with OBB observation if object is tracked, else None.
            Always updates self.last_* fields for visualization.
        """
        # 1. Capture
        ir_left, ir_right, color_bgr = self.capture_frames()
        self.last_color_bgr = color_bgr

        # 2. SAM2 track
        if self.sam2_initialized:
            self._track_sam2(color_bgr)

        # 3. FFS -> point cloud
        points, colors, u_rgb, v_rgb = self._run_ffs_pcd(ir_left, ir_right, color_bgr)
        self.last_points = points
        self.last_colors = colors
        self.last_u_rgb = u_rgb
        self.last_v_rgb = v_rgb

        # 4. Table plane fitting (phase == 'table')
        if self.phase == 'table' and self.table_mask_eroded is not None and len(points) > 0:
            self._fit_table_plane(points, u_rgb, v_rgb)

        # 5. Object OBB fitting (phase == 'object')
        dris = None
        if self.phase == 'object' and self.current_mask_eroded is not None and len(points) > 0:
            dris = self._fit_obb(points, u_rgb, v_rgb)

        return dris

    def close(self) -> None:
        pass  # Pipeline lifecycle managed by main.py

    # ----- SAM2 prompt interface (called from main.py) -----

    def add_prompt(self, mode: str, color_bgr: np.ndarray, **kwargs):
        """Add a SAM2 prompt. Called by main.py when user clicks.

        Args:
            mode: 'table_point', 'table_bbox', 'point', 'bbox'
            color_bgr: current color frame
            **kwargs: x, y for point; x1, y1, x2, y2 for bbox
        """
        if self.sam2_initialized:
            self.sam2_predictor.reset_state()
            self.sam2_initialized = False

        if mode.startswith('table') and not self.plane_locked:
            self.plane_smooth_normal = None
            self.plane_smooth_d = None
            self.plane_frame_count = 0
            self.plane_normal_history.clear()
            self.sam2_predictor.load_first_frame(color_bgr)
            if mode == 'table_point':
                self.sam2_predictor.add_new_prompt(
                    frame_idx=0, obj_id=1,
                    points=np.array([[kwargs['x'], kwargs['y']]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32))
            else:
                self.sam2_predictor.add_new_prompt(
                    frame_idx=0, obj_id=1,
                    bbox=np.array([[kwargs['x1'], kwargs['y1']],
                                   [kwargs['x2'], kwargs['y2']]], dtype=np.float32))
            self.sam2_initialized = True
            self.phase = 'table'
            logger.info(f"Table selection: {mode}")

        elif mode in ('point', 'bbox') and self.plane_locked:
            # Reset OBB state
            self.prev_axes = None
            self.obb_smooth_center = None
            self.obb_smooth_extent = None
            self.obb_smooth_R = None
            self.extent_history.clear()
            self.extent_frame_count = 0
            self.current_mask = None
            self.current_mask_eroded = None

            self.sam2_predictor.load_first_frame(color_bgr)
            if mode == 'point':
                self.sam2_predictor.add_new_prompt(
                    frame_idx=0, obj_id=1,
                    points=np.array([[kwargs['x'], kwargs['y']]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32))
            else:
                self.sam2_predictor.add_new_prompt(
                    frame_idx=0, obj_id=1,
                    bbox=np.array([[kwargs['x1'], kwargs['y1']],
                                   [kwargs['x2'], kwargs['y2']]], dtype=np.float32))
            self.sam2_initialized = True
            self.phase = 'object'
            logger.info(f"Object tracking: {mode}")

    # ----- Accessors for main.py visualization -----

    def get_current_obb(self) -> Optional[Dict[str, Any]]:
        """Get current smoothed OBB parameters (in camera coords)."""
        if self.obb_smooth_extent is None:
            return None
        return {
            "center": self.obb_smooth_center.copy(),
            "extent": self.obb_smooth_extent.copy(),
            "R": self.obb_smooth_R.copy(),
        }

    def get_obb_sim(self) -> Optional[Dict[str, Any]]:
        """Get current OBB transformed to sim coords. For TSIP construction."""
        if self.obb_smooth_extent is None or self.R_c2s is None:
            return None
        R_obj_sim = self.R_c2s @ self.obb_smooth_R
        q_wxyz = rotmat_to_quat_wxyz(R_obj_sim)
        half_ext = self.obb_smooth_extent / 2
        obj_sim = self.R_c2s @ (self.obb_smooth_center - self.plane_locked_center)
        return {
            "half_extents": half_ext.astype(np.float32),
            "quat_wxyz": q_wxyz.astype(np.float32),
            "z_sim": float(obj_sim[2]),
            "obj_xy_sim": obj_sim[:2].astype(np.float32),
        }

    def get_transform_context(self) -> Optional[Dict[str, Any]]:
        """Get camera-to-sim transform (for Viser visualization)."""
        if self.R_c2s is None:
            return None
        return {
            "R_c2s": self.R_c2s.copy(),
            "plane_center": self.plane_locked_center.copy(),
            "plane_normal": self.plane_locked_normal.copy(),
            "plane_d": self.plane_locked_d,
        }

    # ----- Internal perception methods -----

    def capture_frames(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Capture one frame from RealSense. Returns (ir_left, ir_right, color_bgr)."""
        frames = self.rs_pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        return ir_left, ir_right, color_bgr

    def _track_sam2(self, color_bgr: np.ndarray):
        """Run SAM2 tracking on current frame, update masks."""
        out_obj_ids, out_mask_logits = self.sam2_predictor.track(color_bgr)
        if len(out_obj_ids) > 0:
            raw_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
            erode_kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (MASK_ERODE_KERNEL, MASK_ERODE_KERNEL))
            eroded = cv2.erode(raw_mask, erode_kern, iterations=1)
            if self.phase == 'table':
                self.table_mask = raw_mask
                self.table_mask_eroded = eroded
                self.current_mask = None
                self.current_mask_eroded = None
            else:
                self.current_mask = raw_mask
                self.current_mask_eroded = eroded
                self.table_mask = None
                self.table_mask_eroded = None
        else:
            if self.phase == 'table':
                self.table_mask = None
                self.table_mask_eroded = None
            else:
                self.current_mask = None
                self.current_mask_eroded = None

    def _run_ffs_pcd(self, ir_left, ir_right, color_bgr):
        """FFS stereo matching -> depth -> colored point cloud.

        Returns:
            (points, colors, u_rgb, v_rgb) — filtered, valid points only.
        """
        H, W = ir_left.shape[:2]
        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)
        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)

        padder = self.InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)
        with torch.amp.autocast('cuda', enabled=True, dtype=self.amp_dtype):
            disp = self.ffs_model.forward(
                img0_p, img1_p, iters=self.valid_iters,
                test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)

        xx = np.arange(W)[None, :].repeat(H, axis=0)
        disp[((xx - disp) < 0)] = np.inf
        depth = self.fx_ir * self.baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0
        gx = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        depth[(gx > 0.5) | (gy > 0.5)] = 0

        depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
        z_flat = depth_ds.reshape(-1)
        valid_mask = z_flat > 0
        z = z_flat[valid_mask]
        u = self.u_flat[valid_mask]
        v = self.v_flat[valid_mask]

        x3d = (u - self.cx_ir) * z / self.fx_ir
        y3d = (v - self.cy_ir) * z / self.fy_ir
        pts_ir = np.stack([x3d, y3d, z], axis=-1)

        pts_color = (self.R_ir_to_color @ pts_ir.T).T + self.T_ir_to_color
        u_rgb = (self.K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + self.K_color[0, 2]).astype(np.int32)
        v_rgb = (self.K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + self.K_color[1, 2]).astype(np.int32)
        in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

        colors = np.zeros((len(z), 3), dtype=np.uint8)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1]
        final_valid = in_bounds & (colors.sum(axis=1) > 0)

        return (pts_ir[final_valid].astype(np.float32),
                colors[final_valid],
                u_rgb[final_valid],
                v_rgb[final_valid])

    def _fit_table_plane(self, points, u_rgb, v_rgb):
        """Fit table plane from masked point cloud. Locks after convergence."""
        tbl_hl = self.table_mask_eroded[v_rgb, u_rgb] > 0
        if not np.any(tbl_hl):
            return

        # Tint table points for visualization
        cf = self.last_colors.astype(np.float32)
        cf[tbl_hl] = cf[tbl_hl] * 0.3 + TABLE_COLOR_RGB.astype(np.float32) * 0.7
        self.last_colors = cf.clip(0, 255).astype(np.uint8)

        table_pts = points[tbl_hl]
        if len(table_pts) < 20 or self.plane_locked:
            return

        centroid = table_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(table_pts - centroid, full_matrices=False)
        raw_normal = Vt[2]
        if raw_normal[2] > 0:
            raw_normal = -raw_normal
        raw_d = np.dot(raw_normal, centroid)

        self.plane_frame_count += 1
        if self.plane_smooth_normal is not None:
            alpha = max(PLANE_SMOOTH_MIN,
                        PLANE_SMOOTH_INIT * (PLANE_SMOOTH_DECAY ** self.plane_frame_count))
            if np.dot(raw_normal, self.plane_smooth_normal) < 0:
                raw_normal = -raw_normal
                raw_d = -raw_d
            self.plane_smooth_normal = alpha * raw_normal + (1 - alpha) * self.plane_smooth_normal
            self.plane_smooth_normal /= np.linalg.norm(self.plane_smooth_normal)
            self.plane_smooth_d = alpha * raw_d + (1 - alpha) * self.plane_smooth_d
        else:
            self.plane_smooth_normal = raw_normal.copy()
            self.plane_smooth_d = raw_d

        self.plane_normal_history.append(self.plane_smooth_normal.copy())

        if (self.plane_frame_count >= PLANE_LOCK_AFTER
                and len(self.plane_normal_history) >= PLANE_HISTORY_LEN):
            nvar = np.var(np.array(self.plane_normal_history), axis=0).sum()
            if nvar < PLANE_LOCK_VAR_THRESH:
                self.plane_locked = True
                self.plane_locked_normal = self.plane_smooth_normal.copy()
                self.plane_locked_d = float(self.plane_smooth_d)
                self.plane_locked_center = centroid.copy()
                self.R_c2s = compute_R_cam_to_sim(self.plane_locked_normal)

                logger.info(
                    f"Table plane LOCKED: n=({self.plane_locked_normal[0]:.4f},"
                    f"{self.plane_locked_normal[1]:.4f},{self.plane_locked_normal[2]:.4f})")

                # Reset SAM2 for next phase
                self.sam2_predictor.reset_state()
                self.sam2_initialized = False
                self.table_mask = None
                self.table_mask_eroded = None
                self.phase = 'idle'

    def _fit_obb(self, points, u_rgb, v_rgb) -> Optional[DRIS]:
        """Fit object OBB from masked point cloud with temporal smoothing.

        Returns:
            DRIS with OBB observation in sim coords, or None if not enough points.
        """
        highlight = self.current_mask_eroded[v_rgb, u_rgb] > 0
        if not np.any(highlight):
            return None

        # Tint object points for visualization
        cf = self.last_colors.astype(np.float32)
        cf[highlight] = cf[highlight] * 0.2 + MASK_COLOR_RGB.astype(np.float32) * 0.8
        self.last_colors = cf.clip(0, 255).astype(np.uint8)

        obj_pts = points[highlight]
        if len(obj_pts) < 10:
            return None

        centroid = obj_pts.mean(axis=0)
        dists = np.linalg.norm(obj_pts - centroid, axis=1)
        filtered = obj_pts[dists <= np.percentile(dists, 90)]
        if len(filtered) < 10:
            return None

        center = filtered.mean(axis=0)
        cov = np.cov((filtered - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, idx_sort]

        if np.linalg.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]
        if self.prev_axes is not None:
            for i in range(3):
                if np.dot(axes[:, i], self.prev_axes[:, i]) < 0:
                    axes[:, i] = -axes[:, i]
        self.prev_axes = axes.copy()

        local = (filtered - center) @ axes
        raw_extent = local.max(axis=0) - local.min(axis=0)
        center = center + axes @ ((local.max(axis=0) + local.min(axis=0)) / 2)

        self.extent_frame_count += 1

        if self.obb_smooth_center is not None:
            self.obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * self.obb_smooth_center

            self.obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * self.obb_smooth_R
            u0 = self.obb_smooth_R[:, 0]
            u0 /= np.linalg.norm(u0)
            u1 = self.obb_smooth_R[:, 1] - np.dot(self.obb_smooth_R[:, 1], u0) * u0
            u1 /= np.linalg.norm(u1)
            self.obb_smooth_R = np.column_stack([u0, u1, np.cross(u0, u1)])

            self.extent_history.append(raw_extent.copy())
            ext_alpha = max(EXTENT_ALPHA_MIN,
                            EXTENT_ALPHA_INIT * (EXTENT_ALPHA_DECAY ** self.extent_frame_count))
            if len(self.extent_history) >= 3:
                candidate_extent = 0.5 * raw_extent + 0.5 * np.median(
                    np.array(self.extent_history), axis=0)
            else:
                candidate_extent = raw_extent
            max_delta = self.obb_smooth_extent * EXTENT_MAX_CHANGE_RATE
            delta = candidate_extent - self.obb_smooth_extent
            clamped = self.obb_smooth_extent + np.clip(delta, -max_delta, max_delta)
            self.obb_smooth_extent = ext_alpha * clamped + (1 - ext_alpha) * self.obb_smooth_extent
        else:
            self.obb_smooth_center = center.copy()
            self.obb_smooth_extent = raw_extent.copy()
            self.obb_smooth_R = axes.copy()
            self.extent_history.append(raw_extent.copy())

        # Build DRIS in sim coords
        obb_sim = self.get_obb_sim()
        if obb_sim is None:
            return None

        obs = np.concatenate([
            obb_sim["obj_xy_sim"],
            [obb_sim["z_sim"]],
            obb_sim["quat_wxyz"],
            obb_sim["half_extents"],
        ])
        return DRIS(
            observation=obs,
            context=self.get_transform_context(),
            metadata={
                "half_extents": obb_sim["half_extents"],
                "quat_wxyz": obb_sim["quat_wxyz"],
                "z_sim": obb_sim["z_sim"],
                "obj_xy_sim": obb_sim["obj_xy_sim"],
                "obb_cam": self.get_current_obb(),
                "extent_frame_count": self.extent_frame_count,
            },
        )
