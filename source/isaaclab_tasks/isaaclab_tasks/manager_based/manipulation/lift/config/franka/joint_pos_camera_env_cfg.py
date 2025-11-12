# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_camera_env_cfg import LiftCameraEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns, TiledCameraCfg
import isaaclab.sim as sim_utils

@configclass
class FrankaCubeLiftCameraEnvCfg(LiftCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot        
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = [0.0, 0.0, -0.045]
        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"
        
        self.img_resolution_scale = 2    # sacle the image resolution
        
        # External camera: front
        self.scene.camera_ext1 = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/exterior1",
            update_period=0.0,
            height=360,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.8,0.0,0.35), rot=(0.5,-0.5,-0.5,0.5), convention="ros"),
        )

        # # External camera: side
        self.scene.camera_ext2 = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/exterior2",
            update_period=0.0,
            height=360,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.25,1.55,0.25), rot=(0.0,0.0,0.70711,-0.70711), convention="ros"),
        )

        # # External camera: bird-eye
        self.scene.camera_bird = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/bird",
            update_period=0.0,
            height=360,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=13.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.25,0,1.75), rot=(0.0,0.70711,0.70711,0.0), convention="ros"),
        )

        # Set Cube as object
        # Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd",

                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
            ),
        )
        
        # Wall
        wall_size = 5.0
        wall_thickness = 0.01
        self.scene.cuboid_wall_1 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/wall_1",
            spawn=sim_utils.CuboidCfg(size=[wall_size, wall_thickness, wall_size]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, wall_size/2, 0])
        )

        self.scene.cuboid_wall_2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/wall_2",
            spawn=sim_utils.CuboidCfg(size=[wall_thickness, wall_size, wall_size]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=[-wall_size/2, 0, 0])
        )

        self.scene.cuboid_wall_3 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/wall_3",
            spawn=sim_utils.CuboidCfg(size=[wall_size, wall_thickness, wall_size]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0, -wall_size/2, 0])
        )

        self.scene.cuboid_wall_4 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/wall_4",
            spawn=sim_utils.CuboidCfg(size=[wall_thickness, wall_size, wall_size]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=[wall_size/2, 0, 0])
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftCameraEnvCfg_PLAY(FrankaCubeLiftCameraEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 25
        # disable randomization for play
        self.observations.policy.enable_corruption = False