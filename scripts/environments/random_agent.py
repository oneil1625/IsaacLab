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
# import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

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

import isaaclab_tasks  # noqa: F401
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
    # from isaaclab.sensors.camera import TiledCamera
    # tiled_camera = scene["camera"]
    # data_type = "rgb"
    frame_idx =0
    from isaaclab.sensors import TiledCameraCfg, CameraCfg, Camera
    import isaaclab.sim as sim_utils
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import isaaclab.utils.io as io_utils
    from PIL import Image
    import numpy as np
    # sensor = env.unwrapped.scene["camera_ext1"]
    # sensor1 = env.unwrapped.scene["camera_ext2"]
    # sensor2 = env.unwrapped.scene["camera_bird"]
        # Create replicator writer
    asset = env.unwrapped.scene["object"]
    robot = env.unwrapped.scene["robot"]
    # output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    # os.makedirs(output_dir, exist_ok=True)
    # rep_writer = rep.BasicWriter(

    #     output_dir=output_dir,

    #     frame_padding=0,


    # )
    # rep_writer.initialize(output_dir=output_dir)

    
    
    
    # reset environment
    env.reset()
    import csv
    import os

    # File to write to
    csv_file = "log.csv"

    # If file doesn’t exist yet, create and write header
    # file_exists = os.path.isfile(csv_file)
    # with open(csv_file, mode="a", newline="") as f:
    #     writer = csv.writer(f)
    #     if not file_exists:
    #         writer.writerow(["frame", "env_id", "cube_pos", "cube_ore","robot_pos","robot_ore","cube_chnged_pos","cube_changed_ore","front_img_rgb"])  # header row
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            # limits = robot._data.joint_pos_limits
            # # Suppose `limits` is your tensor of shape (num_envs, num_joints, 2)
            # low = limits[..., 0]
            # high = limits[..., 1]

            # # Uniform random sample in [low, high]
            # rand_joints = torch.rand_like(low) * (high - low) + low  # (num_envs, num_joints)


            # #import pdb; pdb.set_trace()

            # robot.write_joint_position_to_sim(rand_joints)
            

            # obs,_,_,_,_ = env.step(actions)
            # pose_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0,1.0)}
            # root_states = asset.data.default_root_state.clone()

            # range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
            # ranges = torch.tensor(range_list, device=asset.device)
            # rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (env_cfg.scene.num_envs, 3), device=asset.device)

            # positions = root_states[:, 0:3] + env.unwrapped.scene.env_origins+ rand_samples
            # orientations = math_utils.random_orientation(env_cfg.scene.num_envs, device=asset.device)
            # #quaternion orientation in (w, x, y, z)
            # # set into the physics simulation
            # asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1))

            # from isaaclab.utils.math import subtract_frame_transforms

            # robot_pos = robot._data.root_state_w[:,:3]
            # robot_ore = robot._data.root_state_w[:,3:7]

            # cube_changed = subtract_frame_transforms(positions,orientations,robot_pos,robot_ore)
            # cube_changed_pos = cube_changed[0]
            # cube_changed_ore = cube_changed[1]


            env.step(actions)
            # for _ in range(50):
            #     env.render()


            
            # sensor.reset()
            # sensor.update(dt=0, force_recompute=True)   
            # images = sensor.data.output["rgb"]
            #images1 = sensor1.data.output["rgb"]
           # images2 = sensor2.data.output["rgb"]
            
            # # Append row to CSV
            # with open(csv_file, mode="a", newline="") as f:
            #     writer = csv.writer(f)
            #     for i in range(env_cfg.scene.num_envs):
            #         writer.writerow([frame_idx, i, positions[i].cpu().numpy(), orientations[i].cpu().numpy(),robot._data.root_state_w[i][:3].cpu().numpy(),robot._data.root_state_w[i][3:7].cpu().numpy(),cube_changed_pos[i].cpu().numpy(),cube_changed_ore[i].cpu().numpy(),f"frames/front/random_rgb_{frame_idx:04d}.jpg"])
            
            #   Note: for semantic segmentation, one render() call is enough, but for rgb, multiple render() calls are needed.
                                                        
            # save_images_to_file(images.cpu()/255.0,f"frames/front/random_rgb_{frame_idx:04d}.jpg")
            #save_images_to_file(images1.cpu()/255.0,f"frames/side/random_rgb_{frame_idx:04d}.png")
            #save_images_to_file(images2.cpu()/255.0,f"frames/bird/random_rgb_{frame_idx:04d}.png")
            # from torchvision.utils import make_grid, save_image
            #import pdb; pdb.set_trace()
            # for i in range(env_cfg.scene.num_envs):
            #     save_image(torch.swapaxes(images[i], 0, -1).cpu()/255.0,f"frames/random_rgb_{i}_{frame_idx:04d}.png")

            
            #import pdb;pdb.set_trace()
            # Save to a pickle file
            #body = asset._data.root_state_w
            #print(body)
            #     io_utils.dump_pickle(f"frames/cube_data/position/pkl/cube_pos_{frame_idx:04d}.pkl", positions)
            #     io_utils.dump_pickle(f"frames/cube_data/orientation/pkl/cube_ore_{frame_idx:04d}.pkl", orientations)
            #     io_utils.dump_pickle(f"frames/cube_data/pkl/cube_data_{frame_idx:04d}.pkl", body)
            #     io_utils.dump_pickle(f"frames/pkl/cam_{frame_idx:04d}.pkl", images)
            #     #io_utils.dump_pickle(f"frames/pkl/full/cam_full_res_{frame_idx:04d}.pkl", images1)
            #     # Save the tensor to a file
            #     torch.save(positions, f"frames/cube_data/position/pt/cube_pos_{frame_idx:04d}.pt")
            #     torch.save(orientations, f"frames/cube_data/orientation/pt/cube_ore_{frame_idx:04d}.pt")
            #     torch.save(body, f"frames/cube_data/pt/cube_data_{frame_idx:04d}.pt")
            #     torch.save(images, f"frames/pt/cam_{frame_idx:04d}.pt")
            #     #torch.save(images1, f"frames/pt/full/cam_full_res_{frame_idx:04d}.pt")
            #     # Save to a YAML file
            #     io_utils.dump_yaml(f"frames/cube_data/position/yaml/cube_pos_{frame_idx:04d}.yaml", positions)
            #     io_utils.dump_yaml(f"frames/cube_data/orientation/yaml/cube_ore_{frame_idx:04d}.yaml", orientations)
            #     io_utils.dump_yaml(f"frames/cube_data/yaml/cube_data_{frame_idx:04d}.yaml", body)
            #     io_utils.dump_yaml(f"frames/yaml/cam_{frame_idx:04d}.yaml", images)
            #    # io_utils.dump_yaml(f"frames/yaml/full/cam_full_res_{frame_idx:04d}.yaml", images1)


            # sensor.reset()
            # sensor.update(dt=0, force_recompute=True)
            #Save images from camera at camera_index
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
            
            #
            #vis = prim.GetAttribute("visibility")
            
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

            
            
            #images_invis = sensor.data.output["semantic_segmentation"]



            #draw_semantic(images_invis,sensor,frame_idx,False)
            
            


            
            
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
