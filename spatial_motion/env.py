"""ManiSkill task registration for Panda free-space motion."""
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env

CAMERAS = ("third_view", "wrist")
CONTROL_MODE = "pd_ee_pose"

@register_env("SpatialMotion-v1", max_episode_steps=20000)
class SpatialMotionEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, robot_uids="panda", control_mode=CONTROL_MODE, **kwargs):
        if control_mode not in (CONTROL_MODE, "pd_ee_delta_pos"):
            raise ValueError(f"Unsupported SpatialMotion control mode: {control_mode}")
        super().__init__(*args, robot_uids=robot_uids, control_mode=control_mode, **kwargs)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.03]))

    def _load_scene(self, options):
        build_ground(self.scene, floor_width=4)

    def _initialize_episode(self, env_idx, options):
        qpos = torch.as_tensor(
            self.agent.keyframes["rest"].qpos, device=self.device, dtype=torch.float32
        )
        self.agent.robot.set_qpos(qpos.expand(len(env_idx), -1))

    @property
    def tcp_in_base(self):
        return self.agent.robot.pose.inv() * self.agent.tcp.pose

    def _get_obs_extra(self, info):
        return {"tcp_pose_base": self.tcp_in_base.raw_pose}

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                CAMERAS[0], sapien_utils.look_at([1.1, -0.85, 0.78], [0.38, 0, 0.28]),
                256, 192, np.pi / 3, 0.01, 10,
            ),
            CameraConfig(
                CAMERAS[1], sapien_utils.look_at([-0.055, 0, 0.015], [0, 0, 0.23]),
                256, 192, np.pi / 2, 0.01, 10,
                mount=self.agent.robot.links_map["panda_hand"],
            ),
        ]
