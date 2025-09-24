# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
from pxr import Sdf, Usd, UsdGeom
import os
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)
import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend
from isaaclab.sim.utils import apply_nested
import omni.usd
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import subtract_frame_transforms
def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    scene = env.unwrapped.scene
    frame_idx =0
    from isaaclab.sensors import TiledCameraCfg, CameraCfg, Camera
    import isaaclab.sim as sim_utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    from isaaclab.sensors import save_images_to_file, depth_to_rgba
    from torchvision.utils import make_grid, save_image
    import csv
    sensor = env.unwrapped.scene["camera_ext1"]
    sensor1 = env.unwrapped.scene["camera_ext2"]
    sensor2 = env.unwrapped.scene["camera_bird"]
    asset = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]
    # reset environment
    env.reset()
    # File to write to
    csv_file = "log.csv"
    counter = 0

    # If file doesn’t exist yet, create and write header
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["frame", "env_id", "cube_pos", "cube_ore","robot_pos","robot_ore","cube_changed_pos","cube_changed_ore","front_img_rgb","side_img_rgb","bird_img_rgb","front_img_dp","side_img_dp","bird_img_dp"])  # header row
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # step environment
            
            
            
            
            '''
            rgb_images = tiled_camera.data.output[data_type]
            rgb_normalized = rgb_images[0:1].float().cpu() / 255.0
            print("rgbimages" ,rgb_images, "rgb normalised", rgb_normalized)
            with open("file.txt", "w") as f:
                f.write(str(rgb_images))'''
            #save_images_to_file(env.unwrapped.observation_space["image"], f"frames/rgb_out_env0_{frame_idx:04d}.png")
            frame_idx+=1
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            
            limits = robot._data.joint_pos_limits
            # Suppose `limits` is your tensor of shape (num_envs, num_joints, 2)
            low = limits[..., 0]
            high = limits[..., 1]

            # Uniform random sample in [low, high]
            rand_joints = torch.rand_like(low) * (high - low) + low  # (num_envs, num_joints)


            #import pdb; pdb.set_trace()

            robot.write_joint_position_to_sim(rand_joints)
            

            obs,_,_,_,_ = env.step(actions)
            pose_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-0.1,1.0)}
            root_states = asset.data.default_root_state.clone()

            range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
            ranges = torch.tensor(range_list, device=asset.device)
            rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (env_cfg.scene.num_envs, 3), device=asset.device)

            positions = root_states[:, 0:3] + env.unwrapped.scene.env_origins+ rand_samples
            orientations = math_utils.random_orientation(env_cfg.scene.num_envs, device=asset.device)
            #quaternion orientation in (w, x, y, z)
            # set into the physics simulation
            asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1))
            
            

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
            # Append row to CSV
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                np.savez_compressed(f"frames/depth_{frame_idx}.npz", 
                   front_depth=depth.cpu().numpy(),
                   side_depth=depth1.cpu().numpy(), 
                   bird_depth=depth2.cpu().numpy())
                for i in range(env_cfg.scene.num_envs):
                    save_images_to_file(images[i].cpu()/255.0,f"frames/front/rgb_env{i}_{frame_idx}.jpg")
                    save_images_to_file(images1[i].cpu()/255.0,f"frames/side/rgb_env{i}_{frame_idx}.jpg")
                    save_images_to_file(images2[i].cpu()/255.0,f"frames/bird/rgb_env{i}_{frame_idx}.jpg")
                    # Save depth maps as compressed .npz
                    writer.writerow([frame_idx, i, cube_data[i][:3].cpu().numpy(), cube_data[i][3:7].cpu().numpy(),robot._data.root_state_w[i][:3].cpu().numpy(),robot._data.root_state_w[i][3:7].cpu().numpy(),cube_changed_pos[i].cpu().numpy(),cube_changed_ore[i].cpu().numpy(),
                                     f"frames/front/rgb_env{i}_{frame_idx}.jpg",f"frames/side/rgb_env{i}_{frame_idx}.jpg",f"frames/bird/rgb_env{i}_{frame_idx}.jpg",f"frames/front/depth_{frame_idx}.npz",f"frames/side/depth_{frame_idx}.npz",f"frames/bird/depth_{frame_idx}.npz"])

                    # counter += 1
                    # print(counter)
            
                                                        
            
            if frame_idx>10000:
                break
            # Save images from camera at camera_index
            '''
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
                
                single_cam_info = {
                    k: camera.data.info[k][cam_idx] if camera.data.info[k] is not None else None
                    for k in camera.data.output.keys()
                }
            
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
            
            # extract the used quantities (to enable type-hinting)
            
            #import pdb; pdb.set_trace()
            '''
            
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
            t = torch.from_numpy(ans)'''
            # color_to_labels = sensor.data.info["semantic_segmentation"]
            # from pxr import Gf, Sdf

            # for i in range(env_cfg.scene.num_envs):
            #     robot_path = f"/World/envs/env_{i}/Object"  # adjust if needed
            #     prim = stage.GetPrimAtPath(robot_path)
            #     apply_nested(hard_reset_prim_visibility(robot_path,True))
                
            #     '''
            #     if not prim.IsValid():
            #         print(f"[WARN] Robot prim not found at {robot_path}")
            #         continue
            #     #UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited")
            #     #add_update_semantics(prim, 'object')
            #     imageable = UsdGeom.Imageable(prim)
            #     #imageable.GetVisibilityAttr().Set('visible')
            #     #prim.CreateAttribute("semantics:class_object:params:semanticData:color", Sdf.ValueTypeNames.Color4f).Set(Gf.Vec4f(1.0, 0.0, 0.0, 1.0)) 
            #     visibility_attr = imageable.GetVisibilityAttr()
    
            #     # Remove the visibility attribute entirely (clears any overrides)
            #     if visibility_attr.HasAuthoredValueOpinion():
            #         prim.RemoveProperty('visibility')

            #     # Recreate the attribute to force a fresh value
            #     visibility_attr = imageable.CreateVisibilityAttr()
            #     visibility_attr.Set('visible')
            #     '''


            

            # images = sensor.data.output["semantic_segmentation"]
            
            # draw_semantic(images,sensor,frame_idx)
           
            # robot = env_cfg.scene.robot.prim_path
            # prim = stage.GetPrimAtPath('/World/envs/env_0/Robot')
            
            # #
            # #vis = prim.GetAttribute("visibility")
            
            # for i in range(env_cfg.scene.num_envs):
            #     robot_path = f"/World/envs/env_{i}/Object"  # adjust if needed
            #     hard_reset_prim_visibility(robot_path,False)
            #     '''
            #     prim = stage.GetPrimAtPath(robot_path)
            #     if not prim.IsValid():
            #         print(f"[WARN] Robot prim not found at {robot_path}")
            #         continue
            #     # Option A: Remove the semantic label entirely
            #     #prim.RemoveProperty("semantics:class_object:params:semanticData")
            #     #UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible")
            #     #UsdGeom.Imageable(prim).MakeInvisible()
            #     #imageable = UsdGeom.Imageable(prim)
            #     #imageable.GetVisibilityAttr().Set('invisible')
            #     imageable = UsdGeom.Imageable(prim)
            #     visibility_attr = imageable.GetVisibilityAttr()
    
            #     # Remove the visibility attribute entirely (clears any overrides)
            #     if visibility_attr.HasAuthoredValueOpinion():
            #         prim.RemoveProperty('visibility')

            #     # Recreate the attribute to force a fresh value
            #     visibility_attr = imageable.CreateVisibilityAttr()
            #     visibility_attr.Set('invisible')
            #     '''

            
            
            # images_invis = sensor.data.output["semantic_segmentation"]



            # draw_semantic(images_invis,sensor,frame_idx,False)
            
            


            
            
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


def hard_reset_prim_visibility(prim_path, visible=True):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[WARN] Prim not found at {prim_path}")
        return
    
    imageable = UsdGeom.Imageable(prim)
    visibility_attr = imageable.GetVisibilityAttr()
    
    # Remove the visibility attribute entirely (clears any overrides)
    if visibility_attr.HasAuthoredValueOpinion():
        prim.RemoveProperty('visibility')

    # Recreate the attribute to force a fresh value
    visibility_attr = imageable.CreateVisibilityAttr()
    visibility_attr.Set('visible' if visible else 'invisible')

def draw_semantic(image,sensor,frame_idx,vis=True):         
            # Assuming `images` is a tensor of shape (N, H, W, 4)
            images_np = image.cpu().numpy()  # Convert to NumPy (N, H, W, 4)
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
            if vis:
                plt.savefig(f"all_tiles{frame_idx:04d}.png", bbox_inches='tight')
            else:
                plt.savefig(f"all_tiles_minus{frame_idx:04d}.png", bbox_inches='tight')
            
            plt.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
