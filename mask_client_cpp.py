#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   
import zmq
import rospy
import os
import time
import threading
from std_msgs.msg import Float32MultiArray, Float32
from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
from geometry_msgs.msg import Pose, PoseStamped, Twist, PoseArray
import numpy as np
from scipy.spatial.transform import Rotation
import h5py
import pickle
import json
import argparse
from robot.constants import *
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from flask import Flask, render_template, Response, request, jsonify
from external.tamapp.efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
import torchvision.transforms as transforms
import torch.nn.functional as F

from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
import torch
import torch.nn as nn
from einops import rearrange

from robot.metaquest_teleop import drlControl
from robot.schunk_gripper_control import gripperControl
from robot.robot_utils import ImageRecorder
# for single robot 
try:
    ROBOT_ID     = rospy.get_param('/dsr/robot_id')
except:
    ROBOT_ID     = "dsr_l"
ROBOT_MODEL = "a0509"

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:1113")

predictor_ready= False
init_masks=[]

no_obj_points = np.array([[0, 0]], dtype=np.float32)
no_obj_labels = np.array([-1], dtype=np.int32)

# def send_predictor_command(command, payload):
#     socket.send_pyobj({"command": command, "payload": payload})
#     return socket.recv_pyobj()

def send_image(image: np.ndarray):
    success, encoded_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("이미지 인코딩 실패")
    socket.send_multipart(encoded_img.tobytes())

def send_array_recv_mask(array: np.ndarray, meta: dict = {}):
    # 메타정보: shape, dtype + 추가 사용자 정의
    meta.update({'dtype': str(array.dtype), 'shape': array.shape})
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),  # 헤더: JSON 문자열
        array.tobytes()                    # 본문: 실제 데이터
    ])
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode('utf-8'))
    dtype = np.dtype(meta['dtype'])
    shape = tuple(meta['shape'])
    array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return array

# Shared global used by Flask to stream
latest_mask_bytes = None

current_pose_l = None
current_pose_r = None

pose_lock = threading.Lock()

def current_pose_callback_l(msg):
    global current_pose_l
    with pose_lock:
        current_pose_l = np.array(msg.data)

def current_pose_callback_r(msg):
    global current_pose_r
    with pose_lock:
        current_pose_r = np.array(msg.data)

def main(args):
    global init_masks, predictor_ready

    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    robot_id_list = task_config['robot_id_list']
    
    rospy.init_node('dsr_mask_control_py')
    image_recorder = ImageRecorder(camera_names = task_config['camera_names'], init_node=False)
    # dsr = drlControl(robot_id_list = robot_id_list, hz = HZ, init_node=True, teleop=False, thru_cpp=True)
    gripper = gripperControl(robot_id_list = robot_id_list, hz = HZ, init_node=False, teleop=False)
    
    pose_state_sub_l = rospy.Subscriber('/dsr_l/state/pose_to_python', Float32MultiArray, current_pose_callback_l)
    pose_action_pub_l = rospy.Publisher('/dsr_l/action/pose_from_python', PoseStamped, tcp_nodelay=True, queue_size=1, latch=False)
    pose_state_sub_r = rospy.Subscriber('/dsr_r/state/pose_to_python', Float32MultiArray, current_pose_callback_r)
    pose_action_pub_r = rospy.Publisher('/dsr_r/action/pose_from_python', PoseStamped, tcp_nodelay=True, queue_size=1, latch=False)

    # Parameters
    temporal_ensemble = True
    esb_k = 0.05
    policy_update_period = 10000 # tick, work without temporal ensemble
    use_depth = False
    use_masks = args['use_masks']
    overwrite = False
    relative_obs_mode = False
    relative_action_mode = False
    trained_on_gpu_server = False # True if use ckpt in gpu_server folder
    use_rotm6d = True
    img_downsampling = True
    img_downsampling_size = (240, 640) #(640, 180) or (640, 240)
    use_inter_gripper_proprio_input = True
    use_gpu_for_inference = True
    
    inference_batch = 1

    ### Experiment Parameters
    dsr_pose_action_skip = 3
    gripper_action_skip = 4
    record_snapshot = True
    img_name = 'test'
    
    num_robots = len(robot_id_list)

    dataset_name = 'real_robot_evaluation'
    print(args['task_name'] + ', ' + dataset_name + '\n' )

    rate = rospy.Rate(HZ)
    # rate = rospy.Rate(15)
    
    # load model parameters
    ckpt_dir = args['ckpt_dir']
    # ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    # ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    ckpt_path = os.path.join(ckpt_dir, 'policy_step_20000_seed_10.ckpt')
    
    print('ckpt_path: ', ckpt_path)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
        print('config: \n', config)
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    policy = make_policy(policy_class, policy_config)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Extract the state_dict (depending on your format)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    load_status = policy.deserialize(state_dict['model_state'])
    print(f'Loaded policy from: {ckpt_path}')
    # Print results
    print("=== Missing keys (in model, not in checkpoint) ===")
    print(load_status.missing_keys)

    print("\n=== Unexpected keys (in checkpoint, not in model) ===")
    print(load_status.unexpected_keys)
    
    if use_gpu_for_inference:
        policy.cuda()
    else:
        policy.cpu()

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # prepare action list for temporal ensemble
    state_dim = policy_config['state_dim']
    action_dim = policy_config['action_dim']
    num_queries = policy_config['num_queries']
    num_robot_obs = policy_config['num_robot_observations']
    num_image_obs = policy_config['num_image_observations']
    image_obs_every = policy_config['image_observation_skip']

    image_obs_history = dict()
    depth_obs_history = dict()
    mask_obs_history = dict()
    if relative_obs_mode:
        robot_obs_history = np.zeros((num_robot_obs+1, state_dim), dtype=np.float32)
        relative_robot_obs_history = np.zeros((num_robot_obs, state_dim), dtype=np.float32)
    else:
        robot_obs_history = np.zeros((num_robot_obs, state_dim), dtype=np.float32)
    all_time_actions = np.zeros((max_timesteps, max_timesteps+num_queries, action_dim), dtype=np.float32)
    
    action = np.zeros(action_dim, dtype=np.float32)


    # print('---dataset stats--- \n', stats)
    # if policy_class == 'Diffusion':
    #     post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    # else:
    pre_process = lambda s: (s - stats['state_mean']) / stats['state_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # saving dataset
    if not os.path.isdir(dataset_dir[0]):
        os.makedirs(dataset_dir[0])
    dataset_path = os.path.join(dataset_dir[0], dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    data_dict = {
        '/observations/xpos': [],
        '/observations/euler': [],
        '/observations/gripper_pos': [],
        '/actions/pose': [],
        '/actions/gripper_pos': [],
        '/all_actions/left_pos_traj': [],
        '/all_actions/right_pos_traj': [],
        '/all_actions/left_quat_traj': [],
        '/all_actions/right_quat_traj': [],
        '/all_actions/left_gripper_pos_traj': [],
        '/all_actions/right_gripper_pos_traj': [],
    }
    
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if use_depth:
            data_dict[f'/observations/depth_images/{cam_name}'] = []


    images = dict()
    depth_images = dict()
    actual_dt_history = []

    images_size = dict()
    # check camera streaming
    for cam_name in camera_names:
        images = image_recorder.get_images()
        while images[cam_name] is None:
            print('waiting '+cam_name+' image streaming...')
            if rospy.is_shutdown():
                break
            time.sleep(1.0)
            images = image_recorder.get_images()

        images_size[cam_name] = images[cam_name].shape # h w c

    print('Are you ready?')
    for cnt in range(3):
        if rospy.is_shutdown():
            break
        print(3 - cnt, 'seconds before to start!!!')
        time.sleep(1.0)

    #ready gripper thread
    gripper.control_thread_start()
    # dsr.control_thread_start()
    dsr_state_xpos = np.zeros(0)
    with pose_lock:
        dsr_state_xpos = np.concatenate( (dsr_state_xpos, current_pose_l[:3], current_pose_r[:3]) )
    
    dsr_state_euler= np.zeros(0)
    with pose_lock:
        dsr_state_euler = np.concatenate( (dsr_state_euler, current_pose_l[3:], current_pose_r[3:]) )

    # dsr_state_xpos = dsr.get_xpos()
    # dsr_state_euler = dsr.get_euler()
    gripper_state = gripper.get_state()

    if use_rotm6d is True:
        dsr_state_rotm6d = np.zeros(0)
        for r in range(num_robots):
            dsr_state_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3*r:3*r+3], degrees=True)
            dsr_state_rotm = dsr_state_rotation.as_matrix()
            dsr_state_rotm6d = np.concatenate( (dsr_state_rotm6d, dsr_state_rotm[:, 0], dsr_state_rotm[:, 1]), axis = -1)

        if use_inter_gripper_proprio_input:
            rel_xpos = np.zeros(3)
            rel_rotm6d = np.zeros(6)

            reference_ee_pos = dsr_state_xpos[0:3]
            reference_ee_euler = dsr_state_euler[0:3]
            reference_ee_rotation = Rotation.from_euler("ZYZ", reference_ee_euler, degrees=True)
            reference_ee_rotm = reference_ee_rotation.as_matrix()

            rel_xpos[:] = reference_ee_rotm.transpose().dot(dsr_state_xpos[3:6] - reference_ee_pos)
            right_hand_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3:6], degrees=True)
            rel_rotm_temp = reference_ee_rotm.transpose()*right_hand_rotation.as_matrix()
            rel_rotm6d[:] = np.concatenate( (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis = -1)

            robot_state = np.concatenate([dsr_state_xpos, dsr_state_rotm6d, rel_xpos, rel_rotm6d, gripper_state], axis=-1)
        else:    
        #initialize
            robot_state = np.concatenate([dsr_state_xpos, dsr_state_rotm6d, gripper_state], axis=-1)
    else:
        if use_inter_gripper_proprio_input:
            rel_xpos = np.zeros(3)
            rel_rotm6d = np.zeros(6)

            reference_ee_pos = dsr_state_xpos[0:3]
            reference_ee_euler = dsr_state_euler[0:3]
            reference_ee_rotation = Rotation.from_euler("ZYZ", reference_ee_euler, degrees=True)
            reference_ee_rotm = reference_ee_rotation.as_matrix()

            rel_xpos[:] = reference_ee_rotm.transpose().dot(dsr_state_xpos[3:6] - reference_ee_pos)
            right_hand_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3:6], degrees=True)
            rel_rotm_temp = reference_ee_rotm.transpose()*right_hand_rotation.as_matrix()
            rel_rotm6d[:] = np.concatenate( (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis = -1)
            
            robot_state = np.concatenate([dsr_state_xpos, dsr_state_euler, rel_xpos, rel_rotm6d, gripper_state], axis=-1)
        else:
            robot_state = np.concatenate([dsr_state_xpos, dsr_state_euler, gripper_state], axis=-1)

    # robot_state = pre_process(robot_state)
    robot_obs_history[:] = robot_state
    image_sampling = range(0, (num_image_obs-1)*image_obs_every+1, image_obs_every)
    for cam_name in camera_names:
        first_image = image_recorder.get_images()[cam_name]
        # first_image_for_show = cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('first_image', first_image_for_show)
        # cv2.waitKey(0)
        if cam_name == 'head_camera':
            # cropped_first_image = first_image[140:-100,:,:].copy()
            first_image = first_image[:,:640]
            if use_masks:
                binary_mask = send_array_recv_mask(first_image)
                first_mask = (binary_mask > 0).astype(np.float32) * 255
                if img_downsampling:
                    # first_mask = cv2.resize(first_mask[0], dsize=img_downsampling_size, interpolation=cv2.INTER_LINEAR)
                    first_mask = first_mask[0][np.newaxis, :, :]
                # print("first_mask", first_mask.shape)
                mask_obs_history[cam_name] = np.repeat(first_mask[np.newaxis, :, :], (num_image_obs-1)*image_obs_every + 1, axis=0)
                mask_obs_history['head_camera'][0] = first_mask
                # print("next value", mask_obs_history["head_camera"].shape)
            # first_image = cv2.resize(cropped_first_image, dsize=(1280,480), interpolation=cv2.INTER_LINEAR)
        else:
            first_image = cv2.resize(first_image, dsize=(640,480), interpolation=cv2.INTER_LINEAR)
        first_image = rearrange(first_image, 'h w c -> c h w')

        image_obs_history[cam_name] = np.repeat(first_image[np.newaxis, :, :, :], (num_image_obs-1)*image_obs_every + 1, axis=0) #0xxx0xxx0
        if use_depth:
            first_depth = np.array([image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name], image_recorder.get_depth_images()[cam_name]])
            depth_obs_history[cam_name] = np.repeat(first_depth[np.newaxis, :, :, :], (num_image_obs-1)*image_obs_every + 1, axis=0) #0xxx0xxx0
    
    if record_snapshot:
        head_img = image_recorder.get_images()['head_camera']     
        head_img = head_img[:, :, [2, 1, 0]] # swap B and R channel
        cv2.imwrite(f'picture/{img_name}.jpg', head_img)
        # cv2.imshow('first_shot', head_img)
        # cv2.waitKey(0)
        
    print('Start!')

    time0 = time.time()
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        # dsr_state_xpos = dsr.get_xpos()
        # dsr_state_euler = dsr.get_euler()
        
        gripper_state = gripper.get_state()
        
        dsr_state_xpos = np.zeros(0)
        dsr_state_xpos = np.concatenate( (dsr_state_xpos, current_pose_l[:3], current_pose_r[:3]) )
        
        dsr_state_euler= np.zeros(0)
        dsr_state_euler = np.concatenate( (dsr_state_euler, current_pose_l[3:], current_pose_r[3:]) )
        dsr_state_rotm6d = np.zeros(0)
        for r in range(num_robots):
            dsr_state_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3*r:3*r+3], degrees=True)
            dsr_state_rotm = dsr_state_rotation.as_matrix()
            dsr_state_rotm6d = np.concatenate( (dsr_state_rotm6d, dsr_state_rotm[:, 0], dsr_state_rotm[:, 1]), axis = -1)

        # if t%1 == 0:
        if True:
            with torch.inference_mode():
                ### get input data
                ## ROBOT STATE INPUT
                if use_rotm6d is True:
                    if use_inter_gripper_proprio_input:
                        rel_xpos = np.zeros(3)
                        rel_rotm6d = np.zeros(6)

                        reference_ee_pos = dsr_state_xpos[0:3]
                        reference_ee_euler = dsr_state_euler[0:3]
                        reference_ee_rotation = Rotation.from_euler("ZYZ", reference_ee_euler, degrees=True)
                        reference_ee_rotm = reference_ee_rotation.as_matrix()

                        rel_xpos[:] = reference_ee_rotm.transpose().dot(dsr_state_xpos[3:6] - reference_ee_pos)
                        right_hand_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3:6], degrees=True)
                        rel_rotm_temp = reference_ee_rotm.transpose()*right_hand_rotation.as_matrix()
                        rel_rotm6d[:] = np.concatenate( (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis = -1)

                        robot_state = np.concatenate([dsr_state_xpos, dsr_state_rotm6d, rel_xpos, rel_rotm6d, gripper_state], axis=-1)
                    else:
                        robot_state = np.concatenate([dsr_state_xpos, dsr_state_rotm6d, gripper_state], axis=-1)
                else:
                    if use_inter_gripper_proprio_input:
                        rel_xpos = np.zeros(3)
                        rel_rotm6d = np.zeros(6)

                        reference_ee_pos = dsr_state_xpos[0:3]
                        reference_ee_euler = dsr_state_euler[0:3]
                        reference_ee_rotation = Rotation.from_euler("ZYZ", reference_ee_euler, degrees=True)
                        reference_ee_rotm = reference_ee_rotation.as_matrix()

                        rel_xpos[:] = reference_ee_rotm.transpose().dot(dsr_state_xpos[3:6] - reference_ee_pos)
                        right_hand_rotation = Rotation.from_euler("ZYZ", dsr_state_euler[3:6], degrees=True)
                        rel_rotm_temp = reference_ee_rotm.transpose()*right_hand_rotation.as_matrix()
                        rel_rotm6d[:] = np.concatenate( (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis = -1)

                        robot_state = np.concatenate([dsr_state_xpos, dsr_state_euler, rel_xpos, rel_rotm6d, gripper_state], axis=-1)
                    else:
                        robot_state = np.concatenate([dsr_state_xpos, dsr_state_euler, gripper_state], axis=-1)
                
                if num_robot_obs >1:
                    robot_obs_history[1:] = robot_obs_history[:-1]
                robot_obs_history[0] = robot_state[:]

                # print('robot_obs_history: \n', robot_obs_history)
                if relative_obs_mode:
                    for r in range(num_robots):
                        reference_state_pos = robot_obs_history[0, 3*r:3*(r+1)]
                        reference_state_euler = robot_obs_history[0, 3*num_robots+3*r:3*num_robots+3*(r+1)]
                        reference_state_rotation = Rotation.from_euler("ZYZ", reference_state_euler, degrees=True)
                        for t_obs in range(num_robot_obs):
                            # orientation
                            robot_rotation = Rotation.from_euler("ZYZ", robot_obs_history[t_obs+1, 3*num_robots+3*r:3*num_robots+3*(r+1)], degrees=True)
                            robot_rotm = reference_state_rotation.as_matrix().transpose() * robot_rotation.as_matrix()
                            rel_robot_rotation = Rotation.from_matrix(robot_rotm)
                            relative_robot_obs_history[t_obs, 3*num_robots+3*r:3*num_robots+3*(r+1)] = rel_robot_rotation.as_rotvec()
                            # position
                            relative_robot_obs_history[t_obs, 3*r:3*(r+1)] = (reference_state_rotation.as_matrix().transpose()).dot(robot_obs_history[t_obs+1, 3*r:3*(r+1)] - reference_state_pos)
                        # pass gripper position
                        relative_robot_obs_history[:, :-num_robots] = robot_obs_history[:num_robot_obs, :-num_robots]
                        print('relative_robot_obs_history pre: \n', relative_robot_obs_history)
                        relative_robot_obs_history = pre_process(relative_robot_obs_history)
                        # print('relative_robot_obs_history post: \n', relative_robot_obs_history)
                        robot_obs_history_flat = relative_robot_obs_history.reshape([-1])
                else:
                    robot_obs_history_flat = pre_process(robot_obs_history[:])
                    robot_obs_history_flat = robot_obs_history_flat.reshape([-1])
                
                t1 = time.time()
                ## CAMERA INPUT
                for cam_name in camera_names:
                    current_image = image_recorder.get_images()[cam_name]

                    if cam_name == 'head_camera':
                        # cropped_image = current_image[140:-100,:,:].copy()
                        current_image = current_image[:,:640]
                        if use_masks:
                            binary_mask = send_array_recv_mask(current_image)
                            current_mask = (binary_mask > 0).astype(np.float32) * 255
                            # current_mask = np.expand_dims(binary_mask, axis=0) # channel dim
                            if img_downsampling:
                                # binary_mask = cv2.resize(binary_mask[0], dsize=img_downsampling_size, interpolation=cv2.INTER_LINEAR)
                                current_mask = current_mask[0][np.newaxis, :, :]
                            # print("head_cam_mask", binary_mask.shape)
                            if num_image_obs >1:
                                mask_obs_history[cam_name][1:] =  mask_obs_history[cam_name][:-1]
                                mask_obs_history[cam_name][0] = current_mask
                            else:
                                mask_obs_history[cam_name][0] = current_mask
                        # current_image = cv2.resize(cropped_image, dsize=(1280,480), interpolation=cv2.INTER_LINEAR)
                    else:
                        current_image = cv2.resize(current_image, dsize=(640,480), interpolation=cv2.INTER_LINEAR)
                    # current_image = rearrange(image_recorder.get_images()[cam_name], 'h w c -> c h w')
                    current_image = rearrange(current_image, 'h w c -> c h w')


                    if num_image_obs >1:
                        image_obs_history[cam_name][1:] =  image_obs_history[cam_name][:-1]
                        image_obs_history[cam_name][0] = current_image
                    else:
                        image_obs_history[cam_name][0] = current_image
                    
                # print('image obs', image_obs_history['head_camera'].shape)
                all_cam_images = []
                for cam_name in camera_names:
                    # print(cam_name, image_obs_history[cam_name].shape)
                    all_cam_images.append(np.array(image_obs_history[cam_name])[image_sampling])

                all_cam_images = np.stack(all_cam_images, axis=0)
                # print("all cam", all_cam_images.shape)
                # print('mask_obs_history', mask_obs_history['head_camera'].shape)
                if use_masks:
                    all_cam_masks = []
                    for cam_name in camera_names:
                        if cam_name == "head_camera":
                            all_cam_masks.append(np.array(mask_obs_history[cam_name])[image_sampling])
                            all_cam_masks = np.stack(all_cam_masks, axis =0)

                # move data into GPU
                cam_masks = None
                if use_gpu_for_inference:
                    if inference_batch == 1:
                        robot_obs_history_torch = torch.from_numpy(robot_obs_history_flat).float().cuda().unsqueeze(0)
                        cam_images = torch.from_numpy(all_cam_images / 255.0).float().cuda().unsqueeze(0)
                        if use_masks:
                            cam_masks = torch.from_numpy(all_cam_masks).float().cuda().unsqueeze(0)
                    else:
                        robot_obs_history_torch = torch.from_numpy(robot_obs_history_flat).float().cuda().unsqueeze(0).repeat_interleave(inference_batch, dim=0)
                        cam_images = torch.from_numpy(all_cam_images / 255.0).float().cuda().unsqueeze(0).repeat_interleave(inference_batch, dim=0)
                        if use_masks:
                            cam_masks = torch.from_numpy(all_cam_masks).float().cuda().unsqueeze(0).repeat_interleave(inference_batch, dim=0)
                else:
                    robot_obs_history_torch = torch.from_numpy(robot_obs_history_flat).float().cpu().unsqueeze(0)
                    cam_images = torch.from_numpy(all_cam_images / 255.0).float().cpu().unsqueeze(0)
                    if use_masks:
                        cam_masks = torch.from_numpy(all_cam_masks).float().cpu().unsqueeze(0)
                # print("image", cam_images.shape, "masks", cam_masks.shape)
                
                # if img_downsampling:
                #     transformations = [
                #         # transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                #         # transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1]/2 * ratio)]),
                #         transforms.Resize(size=img_downsampling_size),
                #         transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                #     ]
                #     for transform in transformations:
                #         cam_images = transform(cam_images)
                #     cam_masks = transforms.Resize(size=img_downsampling_size)(cam_masks)
                
                
                if img_downsampling:
                    # color_jitter = transforms.ColorJitter(brightness=0.3,contrast=0.4,saturation=0.5,hue=0.08)

                    # # Apply ColorJitter per image (expects [C, H, W] inputs)
                    # cam_images = torch.stack([color_jitter(img) for img in cam_images])

                    # Resize cam_images: shape [B, C, H, W]
                    target_h, target_w = img_downsampling_size
                    B, K, T, C, H, W = cam_images.shape
                    cam_images = cam_images.view(B * K * T, C, H, W)

                    cam_images = F.interpolate(
                        cam_images,
                        size=(target_h, target_w),  # New H, W
                        mode='bilinear',
                        align_corners=False
                    )

                    # Restore original shape
                    cam_images = cam_images.view(B, K, T, C, target_h, target_w)
                    if use_masks:
                        B, K, T, C, H, W = cam_masks.shape
                        cam_masks = cam_masks.view(B * K * T, C, H, W)

                        cam_masks = F.interpolate(
                            cam_masks,
                            size=(target_h, target_w),  # New H, W
                            mode='bilinear',
                            align_corners=False
                        )

                        # Restore original shape
                        cam_masks = cam_masks.view(B, K, T, C, target_h, target_w)


                t2 = time.time()
                # policy inference
                # print("shape", cam_images.shape, cam_masks.shape) # [1,3,2,3,240,640] [2,1,240,640]
                all_actions = policy(robot_obs_history_torch, cam_images, masks = cam_masks) # action dim: [1, chunk_size, action_dim]
                t3 = time.time()
                all_actions = all_actions.cpu().numpy()
                all_actions = all_actions[0]
                # all_actions = np.squeeze(all_actions, axis =0)
                all_actions = post_process(all_actions)

                
                # print('all_actions: \n', all_actions[:30])
                if relative_action_mode:
                    dsr_rotation = Rotation.from_euler("ZYZ", dsr_state_euler, degrees=True)
                    dsr_state_rotm = dsr_rotation.as_matrix()
                    rel_all_actions = all_actions
                    
                    rel_all_actions_rotation = Rotation.from_rotvec(rel_all_actions[:, 3:6])
                    all_actions_rotm = dsr_state_rotm * rel_all_actions_rotation.as_matrix()[:]
                    all_actions_rotation = Rotation.from_matrix(all_actions_rotm)
                    
                    for i in range(num_queries):
                        # position
                        all_actions[i, 0:3] = dsr_state_xpos + dsr_state_rotm.dot(rel_all_actions[i, 0:3])
                        # euler
                        all_actions[i, 3:6] = all_actions_rotation[i].as_euler("ZYZ", degrees=True)
                
                # print('all_actions post: \n', all_actions)
                 
                if temporal_ensemble:
                    all_time_actions[t, t:t+num_queries, :] = all_actions[:]
                    
                    ### for pose actions ###
                    TE_horizon_pose = num_queries - dsr_pose_action_skip
                    
                    pose_action_bottom_idx = max(t-TE_horizon_pose+1, 0)
                    pose_actions_for_ensemble = all_time_actions[ pose_action_bottom_idx:(t+1), t+dsr_pose_action_skip, :-num_robots] # (chunk_size, pose_action_dim)
                    

                    # actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                    # print('actions_populated: ', actions_populated)
                    # actions_for_curr_step = actions_for_curr_step[actions_populated]

                    pose_exp_weights = np.exp( esb_k * np.arange(len(pose_actions_for_ensemble)))
                    # exp_weights = np.exp( esb_k * np.arange(len(actions_for_nnext_step)))
                    pose_exp_weights = pose_exp_weights / pose_exp_weights.sum()
                    pose_exp_weights = np.expand_dims(pose_exp_weights, axis=1).transpose() # (1, chunk_size)

                    action[:-num_robots] = np.sum(  pose_exp_weights.dot(pose_actions_for_ensemble), axis=0, keepdims=False)
                    
                    ### for gripper actions ###
                    TE_horizon_gripper = num_queries - gripper_action_skip
                    
                    gripper_action_bottom_idx = max(t-TE_horizon_gripper+1, 0)
                    gripper_actions_for_ensemble = all_time_actions[ gripper_action_bottom_idx:(t+1), t+gripper_action_skip, -num_robots:] # (chunk_size, gripper action_dim(2) )
                    

                    # actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                    # print('actions_populated: ', actions_populated)
                    # actions_for_curr_step = actions_for_curr_step[actions_populated]

                    gripper_exp_weights = np.exp( esb_k * np.arange(len(gripper_actions_for_ensemble)))
                    # exp_weights = np.exp( esb_k * np.arange(len(actions_for_nnext_step)))
                    gripper_exp_weights = gripper_exp_weights / gripper_exp_weights.sum()
                    gripper_exp_weights = np.expand_dims(gripper_exp_weights, axis=1).transpose() # (1, chunk_size)
                    
                    action[-num_robots:] = np.sum(  gripper_exp_weights.dot(gripper_actions_for_ensemble), axis=0, keepdims=False)
                    
                else:
                    if t%policy_update_period == 0:
                        assert policy_update_period<num_queries
                        action_trajectory = all_actions.copy()
                    action = action_trajectory[t%policy_update_period, :]
                    # print('all_actions shape: ', all_actions.size())
                    # action = all_actions[0][:]
                    # print('action: ', action)
                
                # action = np.squeeze(action, axis=0)
                
                # print('all_action: ', all_actions)
            

                # print('action: ', action)
                # print('robot_state: ', robot_state)
                # for i in range(10):
                #     t1 = time.time()
                #     dsr.step()
                #     t2 = time.time()
                

                
        if rospy.is_shutdown():
            # dsr.stop()
            break
        
        # action = all_actions[t%10][:]
        if use_rotm6d is True:
            action_euler = np.zeros(0)
            for r in range(num_robots):
                action_rotm6d = action[3*num_robots+6*r:3*num_robots+6*(r+1)]
                action_rotm_x_axis = action_rotm6d[0:3]/np.linalg.norm(action_rotm6d[0:3])
                action_rotm_y_axis = action_rotm6d[3:6]
                action_rotm_y_axis = action_rotm_y_axis - np.dot(action_rotm_y_axis, action_rotm_x_axis) * action_rotm_x_axis
                action_rotm_y_axis = action_rotm_y_axis/np.linalg.norm(action_rotm_y_axis)
                action_rotm_z_axis = np.cross(action_rotm_x_axis, action_rotm_y_axis)
                action_rotm_z_axis = action_rotm_z_axis/np.linalg.norm(action_rotm_z_axis)
                action_rotm = np.concatenate([action_rotm_x_axis[:,None], action_rotm_y_axis[:,None], action_rotm_z_axis[:,None]], axis=1)
                action_rotation = Rotation.from_matrix(action_rotm)
                action_euler = np.concatenate( [action_euler, action_rotation.as_euler("ZYZ", degrees = True)], axis=-1)

            
            dsr_desired_pose = np.zeros(0)
            for r in range(num_robots):
                dsr_desired_pose = np.concatenate( [dsr_desired_pose, action[3*r:3*r+3], action_euler[3*r:3*r+3]])
        else:
            dsr_desired_pose = action[:6*num_robots]
        
        desired_gripper_pose = action[-num_robots:]

        # print('desired_gripper_pose: ', desired_gripper_pose)
        # print('dsr_desired_pose: ', dsr_desired_pose)
        # print('dsr_state_xpos: ', dsr_state_xpos)
        # print('dsr_state_euler: ', dsr_state_euler)
        
        ###### COMMAND ROBOT (IMPORTANT) ######
        pose_msg_l = PoseStamped()
        pose_msg_r = PoseStamped()
        pose_msg_l.header.stamp = rospy.Time.now()
        pose_msg_r.header.stamp = rospy.Time.now()
        pose_msg_l.header.frame_id = "base_link"
        pose_msg_r.header.frame_id = "base_link"

        # Left arm
        pos_l = dsr_desired_pose[0:3]
        action_rotation = Rotation.from_euler("ZYZ", dsr_desired_pose[3:6], degrees=True)
        quat_l = action_rotation.as_quat()
        
        pose_msg_l.pose.position.x = pos_l[0]
        pose_msg_l.pose.position.y = pos_l[1]
        pose_msg_l.pose.position.z = pos_l[2]
        pose_msg_l.pose.orientation.x = quat_l[0]
        pose_msg_l.pose.orientation.y = quat_l[1]
        pose_msg_l.pose.orientation.z = quat_l[2]
        pose_msg_l.pose.orientation.w = quat_l[3]

        # Right arm
        pos_r = dsr_desired_pose[6:9]
        action_rotation = Rotation.from_euler("ZYZ", dsr_desired_pose[9:12], degrees=True)
        quat_r = action_rotation.as_quat()

        pose_msg_r.pose.position.x = pos_r[0]
        pose_msg_r.pose.position.y = pos_r[1]
        pose_msg_r.pose.position.z = pos_r[2]
        pose_msg_r.pose.orientation.x = quat_r[0]
        pose_msg_r.pose.orientation.y = quat_r[1]
        pose_msg_r.pose.orientation.z = quat_r[2]
        pose_msg_r.pose.orientation.w = quat_r[3]
        pose_action_pub_l.publish(pose_msg_l)
        pose_action_pub_r.publish(pose_msg_r)
        
        gripper.set_action(desired_gripper_pose)
        gripper.set_action(desired_gripper_pose)
        #########################################
        
        # dsr.step() # for ROS topic publish
        

        # if t_end-t0>0.05:
        #     print("t3 - t2 (inference): ", t3-t2)
        #     print("t_end - t0 (total): ", t_end-t0)
        #     print(f"DSR commad duration {t_end-t0} is too long (over 50ms)")
            
        rate.sleep()
        
    
    # stop dsr
    # dsr.stop()
    # open gripper
    gripper.open()
    time.sleep(0.1)
    gripper.stop_control_loop = True
    # shutdown controllers
    # dsr.shutdown()
    # gripper.shutdown()
    # print(f'Avg Control Frequency [Hz]: {dsr.dsr_list[0].tick / (time.time() - time0)}')
    
    

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def rot6d2euler(rot6d):
    rotm_x_axis = rot6d[0:3]/np.linalg.norm(rot6d[0:3])
    rotm_y_axis = rot6d[3:6]
    rotm_y_axis = rotm_y_axis - np.dot(rotm_y_axis, rotm_x_axis) * rotm_x_axis
    rotm_y_axis = rotm_y_axis/np.linalg.norm(rotm_y_axis)
    rotm_z_axis = np.cross(rotm_x_axis, rotm_y_axis)
    rotm_z_axis = rotm_z_axis/np.linalg.norm(rotm_z_axis)
    rotm = np.concatenate([rotm_x_axis[:,None], rotm_y_axis[:,None], rotm_z_axis[:,None]], axis=1)
    rotation = Rotation.from_matrix(rotm)
    euler = rotation.as_euler("ZYZ", degrees = True)
    return euler

def rot6d2quat(rot6d):
    rot6d_size = rot6d.shape
    # print(f'rot6d_size: {rot6d_size}')
    if rot6d_size[0] == 6:
        rotm_x_axis = rot6d[0:3]/np.linalg.norm(rot6d[0:3])
        rotm_y_axis = rot6d[3:6]
        rotm_y_axis = rotm_y_axis - np.dot(rotm_y_axis, rotm_x_axis) * rotm_x_axis
        rotm_y_axis = rotm_y_axis/np.linalg.norm(rotm_y_axis)
        rotm_z_axis = np.cross(rotm_x_axis, rotm_y_axis)
        rotm_z_axis = rotm_z_axis/np.linalg.norm(rotm_z_axis)
        rotm = np.concatenate([rotm_x_axis[:,None], rotm_y_axis[:,None], rotm_z_axis[:,None]], axis=1)
        rotation = Rotation.from_matrix(rotm)
        quat = rotation.as_quat()
    elif rot6d_size[0] == 24:
        quat = np.zeros((24, 4), dtype=np.float32)
        for i in range(24):
            rotm_x_axis = rot6d[i, 0:3]/np.linalg.norm(rot6d[i, 0:3])
            rotm_y_axis = rot6d[i, 3:6]
            rotm_y_axis = rotm_y_axis - np.dot(rotm_y_axis, rotm_x_axis) * rotm_x_axis
            rotm_y_axis = rotm_y_axis/np.linalg.norm(rotm_y_axis)
            rotm_z_axis = np.cross(rotm_x_axis, rotm_y_axis)
            rotm_z_axis = rotm_z_axis/np.linalg.norm(rotm_z_axis)
            rotm = np.concatenate([rotm_x_axis[:,None], rotm_y_axis[:,None], rotm_z_axis[:,None]], axis=1)
            rotation = Rotation.from_matrix(rotm)
            quat[i] = rotation.as_quat()
    else:
        raise TypeError("Input must be a quaternion or quaternion array.") 
    return quat


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='Check Point Directory.', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='mask_demo', required=False)
    parser.add_argument('--use_masks', action='store_true')
    main(vars(parser.parse_args()))