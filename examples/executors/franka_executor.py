"""Franka robot executor — concrete RealWorldExecutor for Franka Emika Panda."""

import socket
import json
import numpy as np
from typing import Dict, Any, Tuple, Optional

from manidreams.executors.real_executor import RealWorldExecutor
from manidreams.base.dris import DRIS


class FrankaExecutor(RealWorldExecutor):
    """Executor for Franka Emika Panda via TCP/JSON socket server.

    Franka-specific interfaces:
    - EE pose control (get_ee_pose, set_ee_pose, move_ee_delta)
    - Gripper control (open_gripper, close_gripper)
    - Camera perception (capture_image, perceive → DRIS)
    """

    def __init__(self, host: str = "172.16.0.1", port: int = 9999,
                 camera_id: Optional[int] = None, name: Optional[str] = None):
        super().__init__(name=name or "FrankaExecutor")
        self.host = host
        self.port = port
        self.camera_id = camera_id
        self._sock: Optional[socket.socket] = None
        self._camera = None

    # --- Connection / ExecutorBase ---

    def initialize(self, config: Dict[str, Any]) -> None:
        self.host = config.get('host', self.host)
        self.port = config.get('port', self.port)
        self.camera_id = config.get('camera_id', self.camera_id)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        if self.camera_id is not None:
            import cv2
            self._camera = cv2.VideoCapture(self.camera_id)
        self.initialized = True

    def _send(self, command: dict) -> dict:
        """Send JSON command to Franka server and return response."""
        self._sock.sendall((json.dumps(command) + "\n").encode())
        return json.loads(self._sock.recv(4096).decode())

    def _execute_single(self, action: Any, get_feedback: bool = True) -> Tuple[Any, Dict]:
        """Execute one EE delta action [dx, dy, dz, drx, dry, drz]."""
        delta = action.tolist() if hasattr(action, 'tolist') else list(action)
        response = self._send({"type": "move", "delta": delta})
        obs = self.get_obs()
        feedback = response if get_feedback else {}
        return obs, feedback

    def reset(self) -> Any:
        self._send({"type": "reset"})
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        return self._send({"type": "get_state"})

    # --- EE Control (Franka-specific) ---

    def get_ee_pose(self) -> np.ndarray:
        """Get current EE pose [x, y, z, rx, ry, rz]."""
        state = self._send({"type": "get_state"})
        return np.array(state['pose'])

    def set_ee_pose(self, pose: np.ndarray) -> None:
        """Set absolute EE pose."""
        self._send({"type": "set_pose", "pose": pose.tolist()})

    def move_ee_delta(self, delta: np.ndarray) -> None:
        """Move EE by relative delta [dx, dy, dz, drx, dry, drz]."""
        self._send({"type": "move", "delta": delta.tolist()})

    # --- Gripper (Franka-specific) ---

    def open_gripper(self) -> None:
        self._send({"type": "gripper", "action": "open"})

    def close_gripper(self) -> None:
        self._send({"type": "gripper", "action": "close"})

    # --- Perception ---

    def capture_image(self) -> np.ndarray:
        """Capture RGB image from camera. Returns HWC numpy array."""
        if self._camera is None:
            raise RuntimeError("No camera configured. Pass camera_id in config.")
        ret, frame = self._camera.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        return frame

    def perceive(self) -> DRIS:
        """Capture camera image and return as DRIS."""
        image = self.capture_image()
        return DRIS(observation=image, metadata={'source': 'camera'})

    # --- Cleanup ---

    def close(self) -> None:
        if self._sock:
            self._sock.close()
        if self._camera is not None:
            self._camera.release()
