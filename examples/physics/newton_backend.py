"""
Newton XPBD backend with domain-randomized box objects.

Implements SimulationBackend using Newton physics engine with multi-world
parallel simulation. Each world contains a randomized box object and a
prismatic-joint actuator (the "actor").

Ported from: d415_ffs_realtime_sim.py (standalone demo)
"""

import colorsys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym

import warp as wp
import newton

from manidreams.base.dris import DRIS
from manidreams.physics.simulation_tsip import SimulationBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Newton simulation parameters
# ---------------------------------------------------------------------------
N_ENVS_DEFAULT = 32
SIM_FPS = 100
SIM_SUBSTEPS = 10
SIM_FRAME_DT = 1.0 / SIM_FPS
SIM_DT = SIM_FRAME_DT / SIM_SUBSTEPS

# Domain randomization ranges
BOX_SIZE_RAND = 0.10       # +/-10% per axis
BOX_INIT_Z_OFFSET = 0.01   # above ground to avoid collision
POS_XY_RANGE = 0.005       # +/-5mm
ORI_MAX_ANGLE = 0.06
BOX_MASS = 0.2
BOX_MASS_RAND = 0.20       # +/-20%
BOX_MU_RANGE = (0.2, 0.8)
BOX_RESTITUTION_RANGE = (0.0, 0.5)

# Actor (pusher) parameters
ACTOR_HALF = 0.01
ACTOR_Z = 0.022
ACTOR_INIT_X = 0.10
ACTOR_INIT_Y = 0.10
ACTOR_KP = 9e5
ACTOR_DAMPING = 3e4
ACTOR_MASS = 0.1


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_unit_box():
    """Unit box vertices and faces for batched mesh rendering."""
    v = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
    ], dtype=np.float32)
    f = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.uint32)
    return v, f


def gen_colors(n):
    """Generate N distinct HSV colors."""
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.7, 0.9)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

def randomize_envs(base_half_extents, base_quat_wxyz, base_z_sim, n=N_ENVS_DEFAULT):
    """Generate randomized box params from locked OBB values.

    Args:
        base_half_extents: (3,) half-extents of the detected OBB
        base_quat_wxyz: (4,) quaternion (w,x,y,z) of the OBB in sim coords
        base_z_sim: float, OBB center height above table in sim coords
        n: number of environments

    Returns:
        dict with keys: half_extents, positions, quats, scale_factors,
                        masses, mus, restitutions  (all (n,...) arrays)
    """
    scale_factors = np.random.uniform(1 - BOX_SIZE_RAND, 1 + BOX_SIZE_RAND, (n, 3))
    half_extents = base_half_extents * scale_factors

    positions = np.zeros((n, 3))
    positions[:, 0] = np.random.uniform(-POS_XY_RANGE, POS_XY_RANGE, n)
    positions[:, 1] = np.random.uniform(-POS_XY_RANGE, POS_XY_RANGE, n)
    positions[:, 2] = base_z_sim + BOX_INIT_Z_OFFSET

    # Small orientation perturbation around the base orientation
    axes = np.random.randn(n, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8
    angles = np.random.uniform(0, ORI_MAX_ANGLE, n)
    half_a = angles / 2
    dq = np.column_stack([np.cos(half_a), axes * np.sin(half_a)[:, None]])
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)

    # Compose: q_final = dq * base_quat (Hamilton product)
    quats = np.zeros((n, 4))
    bw, bx, by, bz = base_quat_wxyz
    for i in range(n):
        dw, dx, dy, dz = dq[i]
        quats[i] = [
            dw * bw - dx * bx - dy * by - dz * bz,
            dw * bx + dx * bw + dy * bz - dz * by,
            dw * by - dx * bz + dy * bw + dz * bx,
            dw * bz + dx * by - dy * bx + dz * bw,
        ]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    masses = BOX_MASS * np.random.uniform(1 - BOX_MASS_RAND, 1 + BOX_MASS_RAND, n)
    mus = np.random.uniform(BOX_MU_RANGE[0], BOX_MU_RANGE[1], n)
    restitutions = np.random.uniform(BOX_RESTITUTION_RANGE[0], BOX_RESTITUTION_RANGE[1], n)

    return {
        "half_extents": half_extents.astype(np.float32),
        "positions": positions.astype(np.float32),
        "quats": quats.astype(np.float32),
        "scale_factors": scale_factors.astype(np.float32),
        "masses": masses.astype(np.float32),
        "mus": mus.astype(np.float32),
        "restitutions": restitutions.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Newton multi-world simulation
# ---------------------------------------------------------------------------

class DRISSim:
    """Newton XPBD multi-world simulation for planar pushing."""

    def __init__(self):
        self._target_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self._graph = None
        self._build_count = 0
        self.params = None

    def build(self, params):
        """Build Newton model with N_ENVS worlds, each containing a box + actor."""
        self.params = params
        n_envs = len(params["half_extents"])
        half_extents = params["half_extents"]
        masses = params["masses"]
        mus = params["mus"]
        restitutions = params["restitutions"]
        actor_vol = (2 * ACTOR_HALF) ** 3
        actor_density = float(ACTOR_MASS / actor_vol)

        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=2e3, kd=180.0)
        builder.add_ground_plane(cfg=ground_cfg)

        actor_cfg = newton.ModelBuilder.ShapeConfig(
            mu=1.0, ke=2e3, kd=180.0, restitution=0.0, density=actor_density)
        actor_pos = wp.vec3(ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z)

        self._box_body_ids = []
        self._actor_body_ids = []
        self._j_x_ids = []
        self._j_y_ids = []
        self._j_z_ids = []

        for i in range(n_envs):
            builder.begin_world(label=f"env_{i}")

            hx, hy, hz = half_extents[i]
            pos = params["positions"][i]
            q = params["quats"][i]  # (w, x, y, z)
            box_vol = float((2 * hx) * (2 * hy) * (2 * hz))
            box_density = float(masses[i] / max(box_vol, 1e-9))
            obj_cfg = newton.ModelBuilder.ShapeConfig(
                mu=float(mus[i]), ke=2e3, kd=180.0,
                restitution=float(restitutions[i]), density=box_density)

            box_link = builder.add_link(
                xform=wp.transform(
                    p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    q=wp.quat(float(q[1]), float(q[2]), float(q[3]), float(q[0])),
                ),
                label=f"box_{i}",
            )
            builder.add_shape_box(box_link, hx=float(hx), hy=float(hy), hz=float(hz), cfg=obj_cfg)
            j_free = builder.add_joint_free(box_link)
            builder.add_articulation([j_free], label=f"box_{i}")
            self._box_body_ids.append(box_link)

            # Actor: 3-DOF prismatic chain (x, y, z)
            anchor_link = builder.add_link(
                xform=wp.transform(p=actor_pos, q=wp.quat_identity()))
            inter_x_link = builder.add_link(
                xform=wp.transform(p=actor_pos, q=wp.quat_identity()), mass=0.01)
            inter_y_link = builder.add_link(
                xform=wp.transform(p=actor_pos, q=wp.quat_identity()), mass=0.01)
            actor_link = builder.add_link(
                xform=wp.transform(p=actor_pos, q=wp.quat_identity()), label=f"actor_{i}")
            builder.add_shape_box(actor_link, hx=ACTOR_HALF, hy=ACTOR_HALF, hz=ACTOR_HALF, cfg=actor_cfg)

            j_fixed = builder.add_joint_fixed(
                parent=-1, child=anchor_link,
                parent_xform=wp.transform(p=actor_pos, q=wp.quat_identity()),
                child_xform=wp.transform_identity())
            j_x = builder.add_joint_prismatic(
                parent=anchor_link, child=inter_x_link,
                axis=wp.vec3(1, 0, 0),
                parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0,
                limit_lower=-0.8, limit_upper=0.8)
            j_y = builder.add_joint_prismatic(
                parent=inter_x_link, child=inter_y_link,
                axis=wp.vec3(0, 1, 0),
                parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0,
                limit_lower=-0.8, limit_upper=0.8)
            j_z = builder.add_joint_prismatic(
                parent=inter_y_link, child=actor_link,
                axis=wp.vec3(0, 0, 1),
                parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0,
                limit_lower=-0.8, limit_upper=0.8)
            builder.add_articulation([j_fixed, j_x, j_y, j_z], label=f"actor_{i}")

            self._actor_body_ids.append(actor_link)
            self._j_x_ids.append(j_x)
            self._j_y_ids.append(j_y)
            self._j_z_ids.append(j_z)
            builder.end_world()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        qd_starts = self.model.joint_qd_start.numpy()
        self._actor_x_dofs = [int(qd_starts[j]) for j in self._j_x_ids]
        self._actor_y_dofs = [int(qd_starts[j]) for j in self._j_y_ids]
        self._actor_z_dofs = [int(qd_starts[j]) for j in self._j_z_ids]

        self.step_count = 0
        self._build_count += 1
        if self._build_count == 1:
            self._capture_graph()
        else:
            self._graph = None

    def _capture_graph(self):
        if not wp.get_device().is_cuda:
            self._graph = None
            return
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self._simulate()
        self._graph = capture.graph

    def _simulate(self):
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def set_actor_target(self, x, y, z=None):
        """Set target position for all actors (prismatic joint targets)."""
        dx = x - ACTOR_INIT_X
        dy = y - ACTOR_INIT_Y
        dz = (z - ACTOR_Z) if z is not None else 0.0
        target_pos = self.control.joint_target_pos.numpy()
        for dof_x, dof_y, dof_z in zip(self._actor_x_dofs, self._actor_y_dofs, self._actor_z_dofs):
            target_pos[dof_x] = dx
            target_pos[dof_y] = dy
            target_pos[dof_z] = dz
        wp.copy(self.control.joint_target_pos, wp.array(target_pos, dtype=wp.float32))

    def step(self):
        """Advance simulation by one frame (SIM_SUBSTEPS sub-steps)."""
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        wp.synchronize()
        self.step_count += SIM_SUBSTEPS
        return self._extract_state()

    def get_state(self):
        return self._extract_state()

    def _extract_state(self):
        """Extract box and actor poses from Newton state.

        Returns:
            (box_pos, box_quat, actor_pos, actor_quat)
            All arrays are (n_envs, ...) with quats in (w,x,y,z) order.
        """
        body_q = self.state_0.body_q.numpy()
        box_ids = np.array(self._box_body_ids)
        box_tf = body_q[box_ids]
        box_pos = box_tf[:, 0:3].astype(np.float32)
        # Newton quat order is (x,y,z,w) in body_q → convert to (w,x,y,z)
        box_quat = np.column_stack([box_tf[:, 6], box_tf[:, 3], box_tf[:, 4], box_tf[:, 5]]).astype(np.float32)

        actor_ids = np.array(self._actor_body_ids)
        actor_tf = body_q[actor_ids]
        actor_pos = actor_tf[:, 0:3].astype(np.float32)
        actor_quat = np.column_stack([actor_tf[:, 6], actor_tf[:, 3], actor_tf[:, 4], actor_tf[:, 5]]).astype(np.float32)

        return box_pos, box_quat, actor_pos, actor_quat


# ---------------------------------------------------------------------------
# SimulationBackend implementation
# ---------------------------------------------------------------------------

class NewtonBackend(SimulationBackend):
    """Newton XPBD multi-world backend for planar pushing.

    Each call to create_environment() builds a new Newton model with
    domain-randomized box objects and prismatic-joint actors.
    """

    def create_environment(self, env_config: Dict[str, Any]) -> DRISSim:
        """Build Newton simulation from OBB parameters.

        Expected env_config keys:
            half_extents: (3,) base half-extents from OBB detection
            quat_wxyz: (4,) base quaternion (w,x,y,z) in sim coords
            z_sim: float, OBB center height above table in sim coords
            obj_xy_sim: (2,) XY position offset in sim coords
            n_envs: int (default 32)
        """
        half_extents = np.asarray(env_config["half_extents"], dtype=np.float32)
        quat_wxyz = np.asarray(env_config["quat_wxyz"], dtype=np.float32)
        z_sim = float(env_config["z_sim"])
        obj_xy_sim = np.asarray(env_config.get("obj_xy_sim", [0.0, 0.0]), dtype=np.float32)
        n_envs = env_config.get("n_envs", N_ENVS_DEFAULT)

        params = randomize_envs(half_extents, quat_wxyz, z_sim, n=n_envs)
        # Offset box positions to object's XY in sim coords
        params["positions"][:, 0] += obj_xy_sim[0]
        params["positions"][:, 1] += obj_xy_sim[1]

        sim = DRISSim()
        sim.build(params)
        # Warm up with one step
        sim.set_actor_target(ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z)
        sim.step()

        logger.info(
            f"Newton env built: {n_envs} worlds, "
            f"half_ext=({half_extents[0]*100:.2f}, {half_extents[1]*100:.2f}, {half_extents[2]*100:.2f})cm"
        )
        return sim

    def reset_environment(self, env: DRISSim, seed: int = None, options: Dict = None):
        """Reset not supported for Newton — rebuild via create_environment()."""
        return env.get_state(), {}

    def step_env(self, env: DRISSim, action: Any):
        """Low-level step (not used directly — use step_act instead)."""
        return env.step()

    def get_state(self, env: DRISSim) -> Dict[str, np.ndarray]:
        box_pos, box_quat, actor_pos, actor_quat = env.get_state()
        return {
            "box_pos": box_pos,
            "box_quat": box_quat,
            "actor_pos": actor_pos,
            "actor_quat": actor_quat,
        }

    def set_state(self, env: DRISSim, state: Dict[str, np.ndarray]) -> None:
        """Set actor target position. Box state is physics-driven (not settable)."""
        if "actor_target" in state:
            t = state["actor_target"]
            env.set_actor_target(float(t[0]), float(t[1]),
                                 float(t[2]) if len(t) > 2 else None)

    def dris2state(self, dris: DRIS) -> Dict[str, np.ndarray]:
        if dris.observation is not None and "actor_target" in dris.metadata:
            return {"actor_target": dris.metadata["actor_target"]}
        return {}

    def get_action_space(self, env: DRISSim) -> gym.Space:
        """Action space: (x, y, z) target position for the actor in sim coords."""
        return gym.spaces.Box(
            low=np.array([-0.8, -0.8, -0.8], dtype=np.float32),
            high=np.array([0.8, 0.8, 0.8], dtype=np.float32),
        )

    def load_env(self, context: Dict[str, Any]) -> None:
        pass  # handled in create_environment

    def load_object(self, context: Dict[str, Any]) -> None:
        pass  # handled in create_environment

    def load_robot(self, context: Dict[str, Any]) -> None:
        pass  # handled in create_environment

    def state2dris(self, observations: Any,
                   env_indices: Optional[List[int]] = None,
                   env_config: Optional[Dict[str, Any]] = None) -> List[DRIS]:
        """Convert Newton state dict to list of DRIS (one per environment)."""
        if isinstance(observations, tuple) and len(observations) == 4:
            box_pos, box_quat, actor_pos, actor_quat = observations
        elif isinstance(observations, dict):
            box_pos = observations["box_pos"]
            box_quat = observations["box_quat"]
            actor_pos = observations["actor_pos"]
            actor_quat = observations["actor_quat"]
        else:
            return []

        n_envs = len(box_pos)
        indices = env_indices if env_indices is not None else range(n_envs)
        dris_list = []
        for i in indices:
            obs = np.concatenate([box_pos[i], box_quat[i], actor_pos[i], actor_quat[i]])
            dris_list.append(DRIS(
                observation=obs,
                metadata={
                    "box_pos": box_pos[i],
                    "box_quat": box_quat[i],
                    "actor_pos": actor_pos[i],
                    "actor_quat": actor_quat[i],
                    "env_idx": i,
                },
            ))
        return dris_list

    def step_act(self, actions: Union[Any, List[Any]], env: DRISSim = None,
                 cage=None, single_action: bool = False) -> Any:
        """Set actor target and advance simulation one frame.

        Args:
            actions: (x, y, z) target position or list thereof.
                     If single_action=True, the same action is applied to all envs.
            env: DRISSim instance
            cage: unused
            single_action: if True, broadcast single action to all envs

        Returns:
            (box_pos, box_quat, actor_pos, actor_quat) tuple
        """
        if single_action:
            action = actions[0] if isinstance(actions, list) else actions
        else:
            action = actions[0] if isinstance(actions, list) else actions
        # All envs share the same actor target (same prismatic joint targets)
        action = np.asarray(action, dtype=np.float32)
        if len(action) >= 3:
            env.set_actor_target(float(action[0]), float(action[1]), float(action[2]))
        else:
            env.set_actor_target(float(action[0]), float(action[1]))
        return env.step()

    def close_environment(self, env: DRISSim) -> None:
        pass  # Newton has no explicit close
