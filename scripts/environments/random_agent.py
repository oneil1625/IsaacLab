# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""
'''
./isaaclab.bat -p scripts/environments/random_agent.py --enable_cameras --num_envs=1 --task=Isaac-Lift-Cube-Franka-IK-Abs-v0 --headless
./isaaclab.sh -p scripts/environments/random_agent.py --enable_cameras --num_envs=1 --task=Isaac-Lift-Cube-Franka-IK-Abs-v0 --headless
'''
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(

    "--camera_id",

    type=int,

    choices={0, 1},

    default=0,

    help=(

        "The camera ID to use for displaying points or saving the camera data. Default is 0."

        " The viewport will always initialize with the perspective of camera 0."

    ))
parser.add_argument(

    "--save",

    action="store_true",

    default=False,

    help="Save the data from camera at index specified by ``--camera_id``.",

)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import os
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
# PLACEHOLDER: Extension template (do not remove this comment)
import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend
def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    scene = env.unwrapped.scene
    from isaaclab.sensors.camera import TiledCamera
    tiled_camera = scene["camera"]
    data_type = "rgb"
    frame_idx =0
    from isaaclab.sensors import TiledCameraCfg, CameraCfg, Camera
    import isaaclab.sim as sim_utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    camera = env.unwrapped.scene["camera_ext1"]
        # Create replicator writer

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    os.makedirs(output_dir, exist_ok=True)
    rep_writer = rep.BasicWriter(

        output_dir=output_dir,

        frame_padding=0,


    )
    rep_writer.initialize(output_dir=output_dir)

    import pdb; pdb.set_trace()

    camera_index = args_cli.camera_id
    '''
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/Visuals"):
        stage.RemovePrim("/Visuals")'''
    # reset environment
    env.reset()
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            
            print()
            
            from isaaclab.sensors import save_images_to_file, create_pointcloud_from_depth, create_pointcloud_from_rgbd
            
            
            '''
            rgb_images = tiled_camera.data.output[data_type]
            rgb_normalized = rgb_images[0:1].float().cpu() / 255.0
            print("rgbimages" ,rgb_images, "rgb normalised", rgb_normalized)
            with open("file.txt", "w") as f:
                f.write(str(rgb_images))'''
            #save_images_to_file(env.unwrapped.observation_space["image"], f"frames/rgb_out_env0_{frame_idx:04d}.png")
            frame_idx+=1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            obs,_,_,_,_ = env.step(actions)

            # Save images from camera at camera_index
            
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            for cam_idx in range(env_cfg.scene.num_envs):
                single_cam_data = convert_dict_to_backend(

                    {k: v[cam_idx] for k, v in camera.data.output.items()}, backend="numpy"

                )

                import pdb;pdb.set_trace()
                # Extract the other information
                
                emp = {}
                for data_types in camera.data.output.keys():
                    if data_types in camera.data.info.keys():
                        emp[data_types] = camera.data.info[data_types]
                    else:
                        emp[data_types] = None
                    

                single_cam_info = emp
                '''
                single_cam_info = {
                    k: camera.data.info[k][cam_idx] if camera.data.info[k] is not None else None
                    for k in camera.data.output.keys()
                }
                '''
            # Pack data back into replicator format to save them using its writer

            rep_output = {"annotators": {}}

            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):

                if info is not None:

                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}

                else:

                    rep_output["annotators"][key] = {"render_product": {"data": data}}

            # Save images

            # Note: We need to provide On-time data for Replicator to save the images.

            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            import pdb;pdb.set_trace()
            rep_writer.write(rep_output)
            '''
            # extract the used quantities (to enable type-hinting)
            images = sensor.data.output["semantic_segmentation"]
            #import pdb; pdb.set_trace()
            '''
            distance_to_camera_file_name = "distance_to_image_plane_1_0000.npy"
            distance_to_camera_data = np.load(os.path.join(output_dir, distance_to_camera_file_name))
            distance_to_camera_file_name = np.nan_to_num(distance_to_camera_data, posinf=0)

            
            #depth_data = np.nan_to_num(images.cpu().numpy(),posinf=0)
            near = 0.01
            far = 2.5
            clipped = np.clip(distance_to_camera_data, near, far)
            # depth_data = depth_data / far
            ha = (np.log(clipped) - np.log(near)) / (np.log(far) - np.log(near))
            # depth_data = depth_data / far
            ans = 1.0 - ha
            depth_data_uint8 = (ans * 255).astype(np.uint8)
            
            distance_to_camera_image = Image.fromarray(depth_data_uint8)
            distance_to_camera_image.save(os.path.join(output_dir, "distance_to_camera.png"))
            #depth_data_uint8 = (depth_data[1:3] * 255).astype(np.uint8)
            #import pdb; pdb.set_trace()
            #image = Image.fromarray(depth_data_uint8)
            #image.save("frames/front/rgb_out{frame_idx:04d}.png")
            # obtain the input image
            t = torch.from_numpy(ans)
            color_to_labels = sensor.data.info["semantic_segmentation"]
            #
            print(images)
            
            '''
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(images[0].cpu())
            items = list(color_to_labels.values())[0]
            
            import pdb; pdb.set_trace()
            color_patch_list = []
            for color, labels in items.items():
                
                color_val = eval(color)
                color_patch = patches.Patch(color=[i / 255 for i in color_val], label=labels)
                color_patch_list.append(color_patch)

            ax.legend(handles=color_patch_list)

            plt.savefig(f"instance.png")'''
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import numpy as np
            import ast
            import math
            '''
            # Assuming `images` is a tensor of shape (N, H, W, 4)
            images_np = images.cpu().numpy()  # Convert to NumPy (N, H, W, 4)
            N, H, W, C = images_np.shape

            # Determine grid size for plotting
            cols = math.ceil(math.sqrt(N))
            rows = math.ceil(N / cols)

            # Create a figure large enough to hold all tiles
            fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axs = axs.flatten()

            for i in range(N):
                axs[i].imshow(images_np[i])
                axs[i].axis('off')
                axs[i].set_title(f'Tile {i}')

            # Hide any unused subplots
            for j in range(N, len(axs)):
                axs[j].axis('off')

            # Optional: Add a legend using semantic_segmentation info
            seg_info = sensor.data.info["semantic_segmentation"]
            label_dict = list(seg_info.values())[0]
            color_patch_list = []

            for color_str, label_info in label_dict.items():
                rgba = eval(color_str)
                color_norm = [c / 255 for c in rgba[:3]]  # Drop alpha
                color_patch = patches.Patch(color=color_norm, label=label_info['class'])
                color_patch_list.append(color_patch)
            
            # Place legend outside the plot
            fig.legend(handles=color_patch_list, loc='upper right', bbox_to_anchor=(1.1, 1.05))

            plt.tight_layout()
            plt.savefig(f"all_tiles_combined{frame_idx:04d}.png", bbox_inches='tight')
            plt.close()'''

            
            #rgb_normalized = images / 255.0
            #save_images_to_file(obs["image"]["image1"],f"frames/front/rgb_out{frame_idx:04d}.png")
            #save_images_to_file(t,f"frames/front/rgb_out{frame_idx:04d}.png")
            '''
            save_images_to_file(obs["policy"]["image"],f"frames/hand/rgb_out{frame_idx:04d}.png")
            
            save_images_to_file(obs["policy"]["image2"],f"frames/side/rgb_out{frame_idx:04d}.png")
            save_images_to_file(obs["policy"]["image3"],f"frames/bird/rgb_out{frame_idx:04d}.png")
            '''
            '''
            ffmpeg -framerate 30 -i frames/hand/rgb_out%04d.png -vf scale=640:480 -c:v libx264 -pix_fmt yuv420p hand_video.mp4
            ffmpeg -framerate 30 -i frames/front/rgb_out%04d.png -vf scale=640:480 -c:v libx264 -pix_fmt yuv420p front_video.mp4
            ffmpeg -framerate 30 -i frames/side/rgb_out%04d.png -vf scale=640:480 -c:v libx264 -pix_fmt yuv420p side_video.mp4
            ffmpeg -framerate 30 -i frames/bird/rgb_out%04d.png -vf scale=640:480 -c:v libx264 -pix_fmt yuv420p bird_video.mp4
            '''
   

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
