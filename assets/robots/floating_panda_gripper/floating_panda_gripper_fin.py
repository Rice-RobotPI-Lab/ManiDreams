"""
FloatingPandaGripperFin Agent for ManiDreams.

This is a local copy of the floating panda gripper with FinRay fingers,
using URDF and mesh assets bundled with ManiDreams instead of ManiSkill's
internal asset directory.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils

# Use local asset directory instead of PACKAGE_ASSET_DIR
ROBOT_DIR = Path(__file__).parent


@register_agent()
class FloatingPandaGripperFin(BaseAgent):
    uid = "floating_panda_gripper_fin"
    # Use local URDF path instead of ManiSkill's PACKAGE_ASSET_DIR
    urdf_path = str(ROBOT_DIR / "panda_fin_gripper.urdf")

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            panda_leftfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            panda_rightfinger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    keyframes = dict(
        open_facing_up=Keyframe(
            qpos=[0, 0, 0, 0, 0, 0, 0.04, 0.04], pose=sapien.Pose(p=[0, 0, 0.5])
        ),
        open_facing_side=Keyframe(
            qpos=[0, 0, 0, 0, np.pi / 2, 0, 0.04, 0.04], pose=sapien.Pose(p=[0, 0, 0.5])
        ),
        open_facing_down=Keyframe(
            qpos=[0, 0, 0, 0, np.pi, 0, 0.04, 0.04], pose=sapien.Pose(p=[0, 0, 0.5])
        ),
    )
    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    ee_link_name = "panda_hand_tcp"

    @property
    def _controller_configs(self):
        pd_ee_pose = PDEEPoseControllerConfig(
            self.root_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=False,
            frame="base",
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        pd_ee_pose_quat = deepcopy(pd_ee_pose)
        pd_ee_pose_quat.rotation_convention = "quaternion"
        pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.root_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=-0.2,
            upper=0.2,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            use_delta=True,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.04,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )
        return dict(
            pd_joint_delta_pos=dict(
                root=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(root=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(root=pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            pd_ee_pose=dict(root=pd_ee_pose, gripper=gripper_pd_joint_pos),
            pd_ee_pose_quat=dict(root=pd_ee_pose_quat, gripper=gripper_pd_joint_pos),
        )

    @property
    def _sensor_configs(self):
        return []

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger"
        )
        self.finger1pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_leftfinger_pad"
        )
        self.finger2pad_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "panda_rightfinger_pad"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
