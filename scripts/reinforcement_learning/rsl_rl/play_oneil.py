# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.math import subtract_frame_transforms
# PLACEHOLDER: Extension template (do not remove this comment)
import numpy as np
from isaaclab.sensors import save_images_to_file, depth_to_rgba
from torchvision.utils import make_grid, save_image

def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    import csv
    sensor = env.unwrapped.scene["camera_ext1"]
    sensor1 = env.unwrapped.scene["camera_ext2"]
    sensor2 = env.unwrapped.scene["camera_bird"]
    asset = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]
    # File to write to
    csv_file = "data.csv"
    counter = 0

    # If file doesn’t exist yet, create and write header
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["frame", "env_id", "cube_pos", "cube_ore","robot_pos","robot_ore","cube_changed_pos","cube_changed_ore","front_img_rgb","side_img_rgb","bird_img_rgb",
                             "front_img_dp","side_img_dp","bird_img_dp","env_origin","front_sem","side_sem","bird_sem","front_sem_robot","side_sem_robot","bird_sem_robot",
                             "front_sem_object","side_sem_object","bird_sem_object"])  # header row
    frame_idx =0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            frame_idx+=1
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            robot_pos = robot._data.root_state_w[:,:3]
            robot_ore = robot._data.root_state_w[:,3:7]



            env.step(actions)
            env.step(actions)

            cube_data = asset._data.root_link_pose_w

            cube_changed = subtract_frame_transforms(robot_pos,robot_ore,cube_data[...,:3],cube_data[...,3:7]) #cube wrt robot

            cube_changed_pos = cube_changed[0]
            cube_changed_ore = cube_changed[1]
            # import pdb; pdb.set_trace()
            # for _ in range(50):
            #     env.render()
            
            
            env.unwrapped.sim.render()
            sensor.reset()
            sensor.update(dt=0, force_recompute=True)   
            images = sensor.data.output["rgb"]
            images1 = sensor1.data.output["rgb"]
            images2 = sensor2.data.output["rgb"]
            depth = sensor.data.output["depth"]
            depth1 = sensor1.data.output["depth"]
            depth2 = sensor2.data.output["depth"]
            semantic = sensor.data.output["semantic_segmentation"]
            semantic1 = sensor1.data.output["semantic_segmentation"]
            semantic2 = sensor2.data.output["semantic_segmentation"]
            for i in range(env_cfg.scene.num_envs):
                    save_images_to_file(semantic[i].cpu()/255.0,f"frames/front/semantic_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic1[i].cpu()/255.0,f"frames/side/semantic_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic2[i].cpu()/255.0,f"frames/bird/semantic_env{i}_{frame_idx}.jpg")
            robot.set_visibility(False)
            env.unwrapped.sim.render()
            sensor.reset()
            sensor.update(dt=0, force_recompute=True) 
            semantic = sensor.data.output["semantic_segmentation"]
            semantic1 = sensor1.data.output["semantic_segmentation"]
            semantic2 = sensor2.data.output["semantic_segmentation"]
            for i in range(env_cfg.scene.num_envs):
                    save_images_to_file(semantic[i].cpu()/255.0,f"frames/front/semantic_robot_only_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic1[i].cpu()/255.0,f"frames/side/semantic_robot_only_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic2[i].cpu()/255.0,f"frames/bird/semantic_robot_only_env{i}_{frame_idx}.jpg")
            asset.set_visibility(False)
            robot.set_visibility(True)
            env.unwrapped.sim.render()
            sensor.reset()
            sensor.update(dt=0, force_recompute=True) 
            semantic = sensor.data.output["semantic_segmentation"]
            semantic1 = sensor1.data.output["semantic_segmentation"]
            semantic2 = sensor2.data.output["semantic_segmentation"]
            for i in range(env_cfg.scene.num_envs):
                    save_images_to_file(semantic[i].cpu()/255.0,f"frames/front/semantic_object_only_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic1[i].cpu()/255.0,f"frames/side/semantic_object_only_env{i}_{frame_idx}.jpg")
                    save_images_to_file(semantic2[i].cpu()/255.0,f"frames/bird/semantic_object_only_env{i}_{frame_idx}.jpg")
            np.savez_compressed(f"frames/front/depth_{frame_idx}.npz", depth=depth.cpu().numpy())
            np.savez_compressed(f"frames/side/depth_{frame_idx}.npz", depth=depth1.cpu().numpy())
            np.savez_compressed(f"frames/bird/depth_{frame_idx}.npz", depth=depth2.cpu().numpy())
            # Append row to CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for i in range(env_cfg.scene.num_envs):
                    save_images_to_file(images[i].cpu()/255.0,f"frames/front/rgb_env{i}_{frame_idx}.jpg")
                    save_images_to_file(images1[i].cpu()/255.0,f"frames/side/rgb_env{i}_{frame_idx}.jpg")
                    save_images_to_file(images2[i].cpu()/255.0,f"frames/bird/rgb_env{i}_{frame_idx}.jpg")
                    # Save depth maps as compressed .npz
                    writer.writerow([frame_idx, i, cube_data[i][:3].cpu().numpy(), cube_data[i][3:7].cpu().numpy(),robot._data.root_state_w[i][:3].cpu().numpy(),robot._data.root_state_w[i][3:7].cpu().numpy(),cube_changed_pos[i].cpu().numpy(),cube_changed_ore[i].cpu().numpy(),
                                     f"frames/front/rgb_env{i}_{frame_idx}.jpg",f"frames/side/rgb_env{i}_{frame_idx}.jpg",f"frames/bird/rgb_env{i}_{frame_idx}.jpg",f"frames/front/depth_{frame_idx}.npz",f"frames/side/depth_{frame_idx}.npz",f"frames/bird/depth_{frame_idx}.npz",
                                     env.unwrapped.scene.env_origins[i].cpu().numpy(),f"frames/front/semantic_env{i}_{frame_idx}.jpg",f"frames/side/semantic_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_env{i}_{frame_idx}.jpg",f"frames/front/semantic_object_only_env{i}_{frame_idx}.jpg",
                                     f"frames/side/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/front/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/side/semantic_object_only_env{i}_{frame_idx}.jpg",f"frames/bird/semantic_object_only_env{i}_{frame_idx}.jpg"])
            asset.set_visibility(True)
            robot.set_visibility(True)


        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
