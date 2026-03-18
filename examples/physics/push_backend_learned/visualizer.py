"""
Cage Visualization for Diffusion-based TSIP

Provides realtime pygame display and video recording with cage overlay for pixel-based predictions.

Features:
    - Realtime pygame window display (default: enabled)
    - Video recording to MP4 (optional)
    - Interactive controls (pause/resume, single-step, exit)
    - Cage constraint visualization (circle overlay)
    - Action direction indicators (arc segments)
    - Information overlay (timestep, action, cage position)

Interactive Controls:
    SPACE       - Pause/Resume execution
    RIGHT ARROW - Single step when paused
    ESC         - Exit program

Usage:
    # Realtime display only
    viz = DiffusionVisualizer(enable_realtime=True, enable_video=False)

    # Realtime + video recording
    viz = DiffusionVisualizer(
        output_path=Path('video.mp4'),
        enable_realtime=True,
        enable_video=True
    )

    # Video recording only (legacy mode)
    viz = DiffusionVisualizer(
        output_path=Path('video.mp4'),
        enable_realtime=False,
        enable_video=True
    )
"""

import logging

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import torch

logger = logging.getLogger(__name__)


class DiffusionVisualizer:
    """Visualizer for diffusion model predictions with cage overlay.

    Supports both realtime pygame display and video recording.
    """

    def __init__(self,
                 output_path: Optional[Path] = None,
                 scale_factor: int = 4,
                 fps: int = 15,
                 cage_params: Optional[Dict] = None,
                 enable_realtime: bool = True,
                 enable_video: bool = True):
        """
        Initialize visualizer with realtime display and/or video recording.

        Args:
            output_path: Path to save video file (required if enable_video=True)
            scale_factor: Upscaling factor for pixel art clarity
            fps: Frame rate for both display and video
            cage_params: Cage configuration (radius, orbit_radius, orbit_speed)
            enable_realtime: Enable realtime pygame display (default: True)
            enable_video: Enable video recording (default: True)
        """
        # Parameter validation
        if enable_video and output_path is None:
            raise ValueError("output_path required when enable_video=True")

        # Video recording attributes
        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.scale_factor = scale_factor
        self.fps = fps
        self.cage_params = cage_params or {}

        self.enable_video = enable_video
        self.video_writer = None
        self.frame_count = 0

        # Pygame realtime display attributes
        self.enable_realtime = enable_realtime
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.is_paused = False
        self.should_exit = False

        if self.enable_realtime:
            logger.info("Realtime display enabled (Press SPACE to pause, ESC to exit)")

    def _initialize_pygame(self, frame_shape):
        """Initialize pygame window on first frame."""
        if self.pygame_initialized:
            return

        try:
            import pygame
            pygame.init()

            h, w = frame_shape[:2]

            # Create window
            self.screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("ManiDreams - Pushing")

            # Create clock for FPS control
            self.clock = pygame.time.Clock()

            self.pygame_initialized = True
            logger.info("Pygame window initialized: %dx%d", w, h)

        except Exception as e:
            logger.error("Pygame failed to initialize: %s", e)
            self.enable_realtime = False

    def _initialize_writer(self, frame_shape):
        """Initialize VideoWriter on first frame."""
        h, w = frame_shape[:2]
        interface_height = 50 * self.scale_factor
        total_height = h + interface_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (w, total_height)
        )
        logger.info("Video recording to: %s", self.output_path)

    def _handle_events(self) -> Dict[str, bool]:
        """
        Process pygame events.

        Returns:
            dict: {
                'should_exit': bool,   # User pressed ESC
                'do_step': bool,       # User pressed RIGHT ARROW (single step)
                'toggle_pause': bool   # User pressed SPACE
            }
        """
        import pygame

        result = {
            'should_exit': False,
            'do_step': False,
            'toggle_pause': False
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                result['should_exit'] = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    result['should_exit'] = True
                    logger.info("User exit requested")

                elif event.key == pygame.K_SPACE:
                    result['toggle_pause'] = True

                elif event.key == pygame.K_RIGHT:
                    result['do_step'] = True

        return result

    def _show_realtime(self, combined_frame) -> bool:
        """
        Display frame in pygame window with event handling.

        Args:
            combined_frame: Combined frame with info overlay (BGR format)

        Returns:
            bool: True to continue, False to exit
        """
        if not self.pygame_initialized:
            self._initialize_pygame(combined_frame.shape)
            if not self.pygame_initialized:  # Failed to initialize
                return True

        import pygame

        # Handle events
        events = self._handle_events()

        if events['should_exit']:
            self.should_exit = True
            return False

        if events['toggle_pause']:
            self.is_paused = not self.is_paused
            status = "PAUSED" if self.is_paused else "RESUMED"
            logger.info("User %s (Press SPACE to toggle, RIGHT ARROW to step)", status)

        # Wait loop if paused
        while self.is_paused:
            events = self._handle_events()

            if events['should_exit']:
                self.should_exit = True
                return False

            if events['toggle_pause']:
                self.is_paused = False
                logger.info("User RESUMED")
                break

            if events['do_step']:
                logger.info("User single step")
                break

            self.clock.tick(10)  # Low FPS during pause

        # Convert BGR (OpenCV) to RGB (Pygame)
        rgb_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)

        # Convert to pygame surface
        surface = pygame.surfarray.make_surface(rgb_frame.transpose(1, 0, 2))

        # Display
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

        # FPS control
        self.clock.tick(self.fps)

        return True

    def process_frame(self,
                     dris: Any,
                     cage_center: Optional[np.ndarray],
                     radius: Optional[float],
                     direction_index: Optional[int],
                     timestep: int,
                     action_info: Dict[str, Any]) -> bool:
        """
        Process and display/record a single frame with cage overlay.

        Args:
            dris: Current DRIS observation
            cage_center: Cage center position [x, y]
            radius: Cage radius
            direction_index: Action direction (0-15)
            timestep: Current timestep
            action_info: Additional action metadata

        Returns:
            bool: True to continue execution, False if user requested exit
        """
        # Extract frame from DRIS observation
        frame = self._extract_frame(dris)
        if frame is None:
            return True

        # Draw cage overlay
        if cage_center is not None and radius is not None:
            frame = self._draw_cage(frame, cage_center, radius)

            if direction_index is not None:
                frame = self._draw_direction(frame, cage_center, radius, direction_index)

        # Scale up for clarity
        frame = cv2.resize(
            frame,
            (frame.shape[1] * self.scale_factor, frame.shape[0] * self.scale_factor),
            interpolation=cv2.INTER_NEAREST
        )

        # Add info overlay
        info_dict = {
            'timestep': timestep,
            'action': direction_index,
            'cage_center': cage_center,
            **action_info
        }
        interface = self._create_info_overlay(frame, info_dict)

        # Combine frame and interface
        combined = np.vstack([frame, interface])

        # Realtime display
        if self.enable_realtime:
            should_continue = self._show_realtime(combined)
            if not should_continue:
                return False  # User requested exit

        # Video recording
        if self.enable_video:
            if self.video_writer is None:
                self._initialize_writer(combined.shape)
            self.video_writer.write(combined)
            self.frame_count += 1

        return True  # Continue execution

    def _extract_frame(self, dris) -> Optional[np.ndarray]:
        """Extract and normalize frame from DRIS observation."""
        obs = dris.observation

        # Handle tensor
        if torch.is_tensor(obs):
            obs = obs.cpu().numpy()

        # Handle batch dimension
        if len(obs.shape) == 4:
            obs = obs[0]

        # Convert CHW to HWC
        if obs.shape[0] in [1, 3]:  # Channels first
            obs = obs.transpose(1, 2, 0)

        # Normalize from [0, 1] to [0, 255]
        # DiffusionBackend already normalizes to [0, 1] range
        frame = (obs * 255).astype(np.uint8)

        # Convert to BGR for OpenCV
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    def _draw_cage(self, frame: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
        """Draw cage circle on frame."""
        center_int = (int(center[0]), int(center[1]))
        radius_int = int(radius)

        # Draw cage circle (light gray)
        cv2.circle(frame, center_int, radius_int + 2, (185, 185, 185), 1)

        # Draw center point (dark gray)
        cv2.circle(frame, center_int, 1, (100, 100, 100), -1)

        return frame

    def _draw_direction(self, frame: np.ndarray, center: np.ndarray,
                       radius: float, direction_index: int) -> np.ndarray:
        """Draw action direction indicator (arc segment)."""
        # Adjust direction index (reference: play_cage_env.py:261)
        adjusted_idx = direction_index - 4

        # Calculate sector angles (16 directions)
        angle1 = (adjusted_idx - 0.5) * 2 * np.pi / 16
        angle2 = (adjusted_idx + 0.5) * 2 * np.pi / 16

        # Calculate arc endpoints
        r = radius + 2
        pt1 = (
            int(center[0] + r * np.cos(angle1)),
            int(center[1] - r * np.sin(angle1))  # Y-axis inverted
        )
        pt2 = (
            int(center[0] + r * np.cos(angle2)),
            int(center[1] - r * np.sin(angle2))
        )

        # Draw arc (red)
        cv2.line(frame, pt1, pt2, (60, 60, 250), 2)

        return frame

    def _create_info_overlay(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """Create text overlay with execution info."""
        h, w = frame.shape[:2]
        interface_height = 50 * self.scale_factor

        # Dark gray background
        overlay = np.ones((interface_height, w, 3), dtype=np.uint8) * 40

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.20 * self.scale_factor
        color = (200, 200, 200)
        thickness = max(1, int(self.scale_factor * 0.2))

        # Format text
        timestep = info.get('timestep', 0)
        action = info.get('action', 'N/A')
        cage_center = info.get('cage_center', None)

        text = f"Step: {timestep}  Action: {action}"
        if cage_center is not None:
            text += f"  Cage: ({cage_center[0]:.1f}, {cage_center[1]:.1f})"

        # Draw text
        x_pos = 5 * self.scale_factor
        y_pos = 25 * self.scale_factor
        cv2.putText(overlay, text, (x_pos, y_pos),
                   font, font_scale, color, thickness, cv2.LINE_AA)

        # Top border line
        cv2.line(overlay, (0, 0), (w, 0), (100, 100, 100), thickness)

        return overlay

    def finalize(self):
        """Release resources and close windows."""
        # Close video writer
        if self.enable_video and self.video_writer is not None:
            self.video_writer.release()
            logger.info("Video saved: %s (%d frames)", self.output_path, self.frame_count)
            self.video_writer = None

        # Close pygame window
        if self.enable_realtime and self.pygame_initialized:
            import pygame
            pygame.quit()
            logger.info("Pygame window closed")
            self.pygame_initialized = False
