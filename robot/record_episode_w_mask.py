#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ##
# @brief    [record_episode] 
# @author   Daegyu Lim (dglim@robros.co.kr)   

import rospy
import os
import time
import sys
import select
import termios
import threading
# from std_msgs.msg import Float32MultiArray, Float32
# from quest2ros.msg import OVR2ROSInputs, OVR2ROSHapticFeedback
# from geometry_msgs.msg import PoseStamped, Twist 
import numpy as np
# from scipy.spatial.transform import Rotation
import h5py
import argparse
from constants import *
import cv2
from tqdm import tqdm
import zmq
import json
import zlib

from metaquest_teleop import drlControl
from schunk_gripper_control import gripperControl
from robot_utils import ImageRecorder
from dxl_master_arm import dsrMasterArm

def write_nested_dataset(root, path, array):
    keys = path.strip('/').split('/')
    grp = root
    for k in keys[:-1]:
        if k not in grp:
            grp = grp.create_group(k)
        else:
            grp = grp[k]
    dset_name = keys[-1]
    if dset_name in grp:
        grp[dset_name][...] = array
    else:
        grp.create_dataset(dset_name, data=array)

def set_cbreak(fd):
    # Save old terminal settings
    old_settings = termios.tcgetattr(fd)
    # Put terminal into cbreak mode: unbuffered, no echo, etc.
    new_settings = old_settings[:]
    new_settings[3] = (new_settings[3] & ~termios.ICANON & ~termios.ECHO)
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    return old_settings

def restore_terminal(fd, old_settings):
    # Restore old terminal settings
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
# for single robot 
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:1113")
predictor_ready= False
init_masks=[]

no_obj_points = np.array([[0, 0]], dtype=np.float32)
no_obj_labels = np.array([-1], dtype=np.int32)

def send_array_recv_mask(array: np.ndarray, meta: dict = {}):
    # 메타정보: shape, dtype + 추가 사용자 정의
    meta.update({'dtype': str(array.dtype), 'shape': array.shape})
    socket.send_multipart([
        json.dumps(meta).encode('utf-8'),
        array.tobytes()
    ])
    meta_bytes, data_bytes = socket.recv_multipart()
    meta = json.loads(meta_bytes.decode('utf-8'))
    dtype = np.dtype(meta['dtype'])
    shape = tuple(meta['shape'])
    array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
    return array

def main(args):

    task_config = TASK_CONFIGS[args['task_name']]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']
    robot_id_list = task_config['robot_id_list']
    num_robots = len(robot_id_list)

    image_recorder = ImageRecorder(camera_names = task_config['camera_names'], init_node=False)
    dsr = drlControl(robot_id_list = robot_id_list, hz = HZ, init_node=True, teleop=True)
    gripper = gripperControl(robot_id_list = robot_id_list, hz = HZ, init_node=False, teleop=True)
    master_arms = dsrMasterArm(robot_id_list = robot_id_list, hz=HZ*2, init_node=False)

    # master_arms.set_init_q('dsr_l', [180.0,180.0,180.0,180.0,180.0,0.0,180.0])
    # master_arms.set_joint_axis('dsr_l', [1,1,-1,1,-1,1,1])

    # master_arms.set_init_q('dsr_r', [270.0,180.0,180.0,180.0,180.0,-180.0,180.0])
    # master_arms.set_joint_axis('dsr_r', [1,1,-1,1,-1,1,1])

    master_arms.set_init_q('dsr_l', [90.0,180.0,180.0,180.0,180.0,360.0,180.0])
    master_arms.set_joint_axis('dsr_l', [1,1,-1,1,-1,1,1])

    master_arms.set_init_q('dsr_r', [360.0,180.0,180.0,180.0,180.0,180.0,180.0])
    master_arms.set_joint_axis('dsr_r', [1,1,-1,1,-1,1,1])

    dsr.control_thread_start()
    gripper.control_thread_start()
    time.sleep(1.0)
    master_arms.thread_start()

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(args['task_name'] + ', ' + dataset_name + '\n' )

    rate = rospy.Rate(HZ)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
        exit()

    data_dict = {
        '/observations/xpos': [],
        '/observations/euler': [],
        '/observations/gripper_pos': [],
        '/actions/pose': [],
        '/actions/gripper_pos': [],
        '/rewards/task': [],
        '/labels/task_done': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        data_dict[f'/observations/masks/{cam_name}'] = []
        # data_dict[f'/observations/depth_images/{cam_name}'] = []


    images = dict()
    masks = dict()
    # depth_images = dict()
    actual_dt_history = []
    
    old_settings = set_cbreak(sys.stdin.fileno())
    last_input = None
    task_reward = 0
    task_done = False
    
    # check cameara streaming
    for cam_name in camera_names:
        images = image_recorder.get_images()
        # depth_images = image_recorder.get_depth_images()
        while images[cam_name] is None:
            print('waiting '+cam_name+' image streaming...')
            if rospy.is_shutdown():
                return
            time.sleep(1.0)
            images = image_recorder.get_images()
        # if cam_name == 'head_camera':
        #     first_image = images[cam_name][:,:640].copy()
        #     binary_mask = send_array_recv_mask(first_image)
        #     first_mask = (binary_mask > 0).astype(np.float32) * 255
        # while depth_images[cam_name] is None:
        #     print('waiting '+cam_name+' depth image streaming...')
        #     if rospy.is_shutdown():
        #         return
        #     time.sleep(1.0)
        #     images = image_recorder.get_images()
    print('Hold Master Arm Handle and wait for connection')
    master_arms.dsrConnect(connect_delay=5.0, connect_spline_duration= 5.0)
    time.sleep(5.0)

    print('Are you ready? Move the robot to the desired initial pose.')
    for cnt in range(2):
        print(2 - cnt, 'seconds before to start DATA COLLECTION!!!')
        time.sleep(1.0)
        if rospy.is_shutdown():
                return

    #ready gripper thread


    print('DATA COLLECTION Start!')
    time0 = time.time()
    for t in tqdm(range(max_timesteps)):
        t0 = time.time() #
        tdsr = time.time() #
        t1 = time.time() #
        
        dsr_action = dsr.get_action()
        dsr_state_xpos = dsr.get_xpos()
        dsr_state_euler = dsr.get_euler()
        gripper_action = gripper.get_action()
        gripper_state = gripper.get_state()

        
        ############# read foot switch input##############
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if rlist:  # If stdin is ready
            # read the line (non-blocking)
            key_input = sys.stdin.read(1)
            if key_input:
                if last_input != key_input:
                    if key_input == 'q':
                        task_done = True
                    elif key_input == 'r':
                        task_reward = 1
                    elif key_input == 'p':
                        task_reward = -1
                else:
                    task_reward = 0
                    task_done = False
                    
                last_input = key_input
                # print(f"Received input: {key_input}")
                # print(f"task_reward: {task_reward}")
                # print(f"task_done: {task_done}")
        else:
            last_input = None      
            task_reward = 0
            task_done = False  
        ##################################################

        # print('dsr_action: ', dsr_action, dsr_action.shape)
        # print('dsr_state_xpos: ', dsr_state_xpos, dsr_state_xpos.shape)
        # print('dsr_state_euler: ', dsr_state_euler, dsr_state_euler.shape)
        # print('gripper_action: ', gripper_action, gripper_action.shape)
        # print('gripper_state: ', gripper_state, gripper_state.shape)

        data_dict['/observations/xpos'].append(dsr_state_xpos)
        data_dict['/observations/euler'].append(dsr_state_euler)
        data_dict['/observations/gripper_pos'].append(gripper_state)
        data_dict['/actions/pose'].append(dsr_action)
        data_dict['/actions/gripper_pos'].append(gripper_action)
        
        for cam_name in camera_names:
            current_image = (image_recorder.get_images())[cam_name]
            
            data_dict[f'/observations/images/{cam_name}'].append(current_image)
            if cam_name == 'head_camera':
                # print('image shape',current_image.shape) # 480 1280 3
                current_input = current_image[:,:640].copy()
                binary_mask = send_array_recv_mask(current_input)
                # print(binary_mask.shape, binary_mask.min(), binary_mask.max(), np.count_nonzero(binary_mask))
                current_mask = (binary_mask > 0).astype(np.uint8) * 255
                # print(current_mask.shape, current_mask.min(), current_mask.max(), np.count_nonzero(current_mask))

                # print('mask shape', current_mask.shape, 'should be 0 640') # 1 480 640
                data_dict[f'/observations/masks/{cam_name}'].append(current_mask.squeeze(0))
            # data_dict[f'/observations/depth_images/{cam_name}'].append((image_recorder.get_depth_images())[cam_name])
            # print(f'{cam_name} size = {image_recorder.get_images()[cam_name].shape}')
        
        data_dict['/rewards/task'].append([task_reward])
        data_dict['/labels/task_done'].append([task_done])
        
        t2 = time.time() #
        actual_dt_history.append([t0, t1, t2])

        # print("tdsr - t0: ", tdsr-t0)
        # print("t1 - tdsr: ", t1-tdsr)
        # print("t2 - t1: ", t2-t1)
        if tdsr-t0>0.1:
            print("DSR commad duration is too long (over 100ms)")

        
        if rospy.is_shutdown() or task_done:
            # dsr.stop()
            # gripper.open()
            # dsr.control_thread_stop()
            # gripper.control_thread_stop()
            # return
            break

        rate.sleep()
    
    data_timesteps = len(data_dict['/observations/xpos'])
    print(f'data_length: {data_timesteps} steps ({data_timesteps/HZ}s)')
    print(f'Avg Control Frequency [Hz]: {data_timesteps / (time.time() - time0)}')
    master_arms.dsrDisconnect()
    # open gripper
    gripper.open()
    # stop dsr
    dsr.stop()

    time.sleep(0.5)

    dsr.control_thread_stop()
    gripper.control_thread_stop()
    master_arms.thread_stop()

    restore_terminal(sys.stdin.fileno(), old_settings)
    
    COMPRESS = True

    if COMPRESS:
        # JPEG compression
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
        compressed_image_len = []
        compressed_mask_len = []
        # compressed_depth_len = []
        for cam_name in camera_names:
            image_list = data_dict[f'/observations/images/{cam_name}']
            compressed_list = []
            compressed_image_len.append([])
            # depth_list = data_dict[f'/observations/depth_images/{cam_name}']
            # compressed_depth_list = []
            # compressed_depth_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_image_len[-1].append(len(encoded_image))
                if result == False:
                    print('Error during image compression')
            # for depth in depth_list:
            #     result, encoded_depth = cv2.imencode('.jpg', depth, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
            #     compressed_depth_list.append(encoded_depth)
            #     compressed_depth_len[-1].append(len(encoded_depth))
            #     if result == False:
            #         print('Error during depth image compression')
            data_dict[f'/observations/images/{cam_name}'] = compressed_list
            
            if cam_name == 'head_camera':
                mask_list = data_dict[f'/observations/masks/{cam_name}']
                compressed_mask_list = []
                compressed_mask_len.append([])
                for mask in mask_list:
                    compressed = zlib.compress(mask.tobytes())
                    compressed_mask_list.append(compressed)
                    compressed_mask_len[-1].append(len(compressed))
            
            # data_dict[f'/observations/depth_images/{cam_name}'] = compressed_depth_list
        print(f'compression: {time.time() - t0:.2f}s')

        # pad so it has same length
        t0 = time.time()
        compressed_image_len = np.array(compressed_image_len)
        padded_size = compressed_image_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list

        # compressed_mask_len = np.array(compressed_mask_len)
        # padded_mask_size = compressed_mask_len.max()
        # compressed_mask_list = data_dict[f'/observations/masks/head_camera']
        # padded_compressed_mask_list = []
        # for compressed_mask in compressed_mask_list:
        #     padded_compressed_mask = np.zeros(padded_mask_size, dtype='uint8')
        #     mask_len = len(compressed_mask)
        #     padded_compressed_mask[:mask_len] = compressed_mask
        #     padded_compressed_mask_list.append(padded_compressed_mask)
        # data_dict[f'/observations/masks/head_camera'] = padded_compressed_mask_list

        # compressed_depth_len = np.array(compressed_depth_len)
        # padded_size2 = compressed_depth_len.max()
        # for cam_name in camera_names:
        #     compressed_depth_list = data_dict[f'/observations/depth_images/{cam_name}']
        #     padded_compressed_depth_list = []
        #     for compressed_depth in compressed_depth_list:
        #         padded_compressed_depth = np.zeros(padded_size2, dtype='uint8')
        #         depth_len = len(compressed_depth)
        #         padded_compressed_depth[:depth_len] = compressed_depth
        #         padded_compressed_depth_list.append(padded_compressed_depth)
        #     data_dict[f'/observations/depth_images/{cam_name}'] = padded_compressed_depth_list
        print(f'padding: {time.time() - t0:.2f}s')

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        # root.attrs['sim'] = False
        root.attrs['compress'] = COMPRESS
        obs = root.create_group('observations')
        actions = root.create_group('actions')
        rewards = root.create_group('rewards')
        labels = root.create_group('labels')
        image = obs.create_group('images')
        masks = obs.create_group('masks')
        # depth = obs.create_group('depth_images')
        
        ### observations ###
        for cam_name in camera_names:
            if COMPRESS:
                _ = image.create_dataset(cam_name, (data_timesteps, padded_size), dtype='uint8',
                                         chunks=(1, padded_size), )
                # _ = depth.create_dataset(cam_name, (data_timesteps, padded_size2), dtype='uint8',
                #                          chunks=(1, padded_size2), )
            else:
                _ = image.create_dataset(cam_name, (data_timesteps, 360, 1280, 3), dtype='uint8',
                                         chunks=(1, 360, 1280, 3), )
                # _ = depth.create_dataset(cam_name, (data_timesteps, 360, 1280), dtype='uint8',
                #                         chunks=(1, 360, 1280, 1), )
            if cam_name == 'head_camera':
                # _ = masks.create_dataset(cam_name, (data_timesteps, padded_mask_size), dtype = 'uint8')
                vlen_uint8 = h5py.vlen_dtype(np.dtype('uint8'))
                compressed_bytes_array = np.empty(len(compressed_mask_list), dtype=object)
                for i, x in enumerate(compressed_mask_list):
                    compressed_bytes_array[i] = np.frombuffer(x, dtype=np.uint8)

                masks.create_dataset(cam_name, data=compressed_bytes_array, dtype=vlen_uint8)
        _ = obs.create_dataset('xpos', (data_timesteps, 3*num_robots))
        _ = obs.create_dataset('euler', (data_timesteps, 3*num_robots))
        _ = obs.create_dataset('gripper_pos', (data_timesteps, 1*num_robots))
        ### actions ###
        _ = actions.create_dataset('pose', (data_timesteps, 6*num_robots))
        _ = actions.create_dataset('gripper_pos', (data_timesteps, 1*num_robots))

        ### rewards ###
        _ = rewards.create_dataset('task', (data_timesteps, 1))
        ### labels ### 
        _ = labels.create_dataset('task_done', (data_timesteps, 1))
        
        for name, array in data_dict.items():
            # Avoid rewriting head_camera mask — already written manually above
            if name == '/observations/masks/head_camera':
                continue
            write_nested_dataset(root, name, array)            
            # root[name][...] = array
            #write_nested_dataset(root, name, array)

        if COMPRESS:
            _ = root.create_dataset('compressed_image_len', (len(camera_names), data_timesteps))
            root['/compressed_image_len'][...] = compressed_image_len
            # _ = root.create_dataset('compressed_depth_len', (len(camera_names), data_timesteps))
            # root['/compressed_depth_len'][...] = compressed_depth_len

    print(f'Saving: {time.time() - t0:.1f} secs')

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', default='mask_demo', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    main(vars(parser.parse_args())) # TODO
    # debug()