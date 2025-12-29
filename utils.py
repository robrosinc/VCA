import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
import zlib
from time import time
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation
import zlib

import IPython

e = IPython.embed


def flatten_list(l):
    return [item for sublist in l for item in sublist]


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path_list,
        camera_names,
        norm_stats,
        episode_ids,
        episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        img_obs_skip,
        policy_class,
        use_depth,
        use_masks,
        use_text
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.robot_obs_size = robot_obs_size
        self.img_obs_size = img_obs_size
        self.img_obs_skip = img_obs_skip
        self.cumulative_len = np.cumsum(self.episode_len)
        print("dataset size: ", self.cumulative_len[-1])
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class

        self.augment_images = True
        self.separate_left_right = False
        self.img_downsample = True
        self.img_downsample_size = (240, 640) # (180, 640) or (240, 640)

        self.img_debug = False

        self.use_depth = use_depth
        self.use_masks = use_masks
        self.use_text = use_text

        self.relative_action_mode = False
        self.relative_obs_mode = False
        self.relative_inter_gripper_proprio = True

        self.__getitem__(0)  # initialize self.is_sim and self.transformations
        self.is_sim = False

    def __len__(self):
        return self.cumulative_len[-1]

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(
            self.cumulative_len > index
        )  # argmax returns first True index
        start_ts = index - (
            self.cumulative_len[episode_index] - self.episode_len[episode_index]
        )
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        t0 = time()
        try:
            with h5py.File(dataset_path, "r") as root:
                try:  # some legacy data does not have this attribute
                    is_sim = root.attrs["sim"]
                except:
                    is_sim = False

                compressed = root.attrs.get("compress", False)

                num_robots = int(root["/actions/pose"].shape[1] / 6)

                action_xpos = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_euler = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_rotm6d = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_gripper = root["/actions/gripper_pos"][()]

                for r in range(num_robots):
                    action_xpos = np.concatenate(
                        (action_xpos, root["/actions/pose"][:, 6 * r : 6 * r + 3]),
                        axis=-1,
                    )
                    action_euler = np.concatenate(
                        (action_euler, root["/actions/pose"][:, 6 * r + 3 : 6 * r + 6]),
                        axis=-1,
                    )
                    action_rotation = Rotation.from_euler(
                        "ZYZ",
                        root["/actions/pose"][:, 6 * r + 3 : 6 * r + 6],
                        degrees=True,
                    )
                    action_rotm = action_rotation.as_matrix()
                    action_rotm6d = np.concatenate(
                        (action_rotm6d, action_rotm[:, :, 0], action_rotm[:, :, 1]),
                        axis=-1,
                    )

                if self.relative_action_mode:
                    action = np.concatenate(
                        [action_xpos, action_euler, action_gripper], axis=-1
                    )
                else:
                    action = np.concatenate(
                        [action_xpos, action_rotm6d, action_gripper], axis=-1
                    )

                # print('action[6]: ', action[:, 6])
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                # get observation at start_ts only
                xpos = root["/observations/xpos"][()]
                euler = root["/observations/euler"][()]

                rotm6d = np.zeros((euler.shape[0], 0))
                rel_xpos = np.zeros((euler.shape[0], 3))
                rel_rotm6d = np.zeros((euler.shape[0], 6))

                for r in range(num_robots):
                    rotation = Rotation.from_euler(
                        "ZYZ", euler[:, 3 * r : 3 * r + 3], degrees=True
                    )
                    rotm = rotation.as_matrix()
                    rotm6d = np.concatenate(
                        (rotm6d, rotm[:, :, 0], rotm[:, :, 1]), axis=-1
                    )

                gripper_pos = root["/observations/gripper_pos"][()]

                if self.relative_inter_gripper_proprio:
                    for t in range(euler.shape[0]):
                        reference_ee_pos = xpos[t, 0:3]
                        reference_ee_euler = euler[t, 0:3]
                        reference_ee_rotation = Rotation.from_euler(
                            "ZYZ", reference_ee_euler, degrees=True
                        )
                        reference_ee_rotm = reference_ee_rotation.as_matrix()

                        rel_xpos[t, :] = reference_ee_rotm.transpose().dot(
                            xpos[t, 3:6] - reference_ee_pos
                        )
                        right_hand_rotation = Rotation.from_euler(
                            "ZYZ", euler[t, 3:6], degrees=True
                        )
                        rel_rotm_temp = (
                            reference_ee_rotm.transpose()
                            * right_hand_rotation.as_matrix()
                        )
                        rel_rotm6d[t, :] = np.concatenate(
                            (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis=-1
                        )

                    if self.relative_obs_mode:
                        robot_state = np.concatenate(
                            [xpos, euler, rel_xpos, rel_rotm6d, gripper_pos], axis=-1
                        )
                    else:
                        robot_state = np.concatenate(
                            [xpos, rotm6d, rel_xpos, rel_rotm6d, gripper_pos], axis=-1
                        )
                else:
                    if self.relative_obs_mode:
                        robot_state = np.concatenate(
                            [xpos, euler, gripper_pos], axis=-1
                        )
                    else:
                        robot_state = np.concatenate(
                            [xpos, rotm6d, gripper_pos], axis=-1
                        )

                t1 = time()

                original_robot_shape = robot_state.shape
                t2 = time()
                # print('robot_state shape', original_robot_shape) # (2000, 7)
                image_dict = dict()

                if self.use_depth:
                    depth_image_dict = dict()
                if self.use_masks:
                    mask_dict = dict()

                img_sampling = np.clip(
                    range(
                        start_ts,
                        start_ts - self.img_obs_skip * self.img_obs_size,
                        -self.img_obs_skip,
                    ),
                    0,
                    self.max_episode_len,
                )

                img_sampling_np = np.array(img_sampling)
                unique_indices, inverse_indices = np.unique(img_sampling_np, return_inverse=True)

                for cam_name in self.camera_names:
                    image_dict[cam_name] = np.array(
                        root[f"/observations/images/{cam_name}"]
                    )[img_sampling]
                    if self.use_masks and cam_name == "head_camera":
                        if "/observations/masks" in root:
                            if f"{cam_name}_masks" in root["/observations/masks"]:
                                mask_dict[cam_name] = np.expand_dims(np.array(
                                    root[f"/observations/masks/{cam_name}_masks"]
                                )[img_sampling][:,:,:], axis=1)
                                # print("raw obs/masks/head_masks: ", mask_dict[cam_name].shape)
                            elif cam_name in root["/observations/masks"]:
                                unique_compressed = root[f"/observations/masks/{cam_name}"][unique_indices]
                                # print("raw obs/masks/head:", unique_compressed.shape)
                                decompressed_masks_unique = [
                                    np.frombuffer(zlib.decompress(entry), dtype=np.uint8).reshape(480, 640)
                                    for entry in unique_compressed
                                ]
                                decompressed_masks = [decompressed_masks_unique[i] for i in inverse_indices]
                                # print("resized obs/masks/head:", np.array(decompressed_masks).shape)
                                mask_dict[cam_name] = np.expand_dims(np.stack(decompressed_masks, axis=0), axis=1)
                        else:
                            if cam_name in root["/prompts/masks"]:
                                unique_compressed = root[f"/prompts/masks/{cam_name}"][unique_indices]
                                decompressed_masks_unique = [
                                    np.frombuffer(zlib.decompress(entry), dtype=np.uint8).reshape(480, 640)
                                    for entry in unique_compressed
                                ]
                                decompressed_masks = [decompressed_masks_unique[i] for i in inverse_indices]
                                # print("prompts/masks (resized):", np.array(decompressed_masks).shape)
                                mask_dict[cam_name] = np.expand_dims(np.stack(decompressed_masks, axis=0), axis=1)
                            else:
                                raise KeyError("No valid mask dataset found for head_camera")
                        resized_masks = []
                        for mask in mask_dict[cam_name]:
                            resized_mask = cv2.resize(mask[0], (self.img_downsample_size[1], self.img_downsample_size[0]), interpolation=cv2.INTER_LINEAR)
                            resized_masks.append(resized_mask)
            
                        # Stack the resized masks and add a new dimension
                        mask_dict[cam_name] = np.expand_dims(np.stack(resized_masks, axis=0), axis=1)

                # print("here", mask_dict['head_camera'].shape) # 2 1 240 640
                t3 = time()
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image_array = []
                        for i in range(len(image_dict[cam_name])):
                            decompressed_image = cv2.imdecode(
                                image_dict[cam_name][i], 1
                            )
                            # cv2.imshow('decoding', decompressed_image)
                            # cv2.waitKey()
                            decompressed_image_array.append(decompressed_image)
                        image_dict[cam_name] = np.array(decompressed_image_array)
                        # print('image_dict[cam_name].shape', image_dict[cam_name].shape)
                t4 = time()

                if self.use_masks:
                    all_cam_masks = []
                    for cam_name in self.camera_names:
                        if cam_name =='head_camera':
                            cropped_mask = mask_dict[cam_name][:,:,:,:640]
                            for t in range(len(mask_dict[cam_name])):
                                mask_dict[cam_name][t] = cropped_mask[t]
                            all_cam_masks.append(mask_dict[cam_name])
                    all_cam_masks = np.stack(all_cam_masks, axis=0)
                    # print("all_cam_masks", all_cam_masks.shape) # 1 T 1 240 640
                    mask_data = torch.from_numpy(all_cam_masks).float()
                    # mask_data = transforms.v2.Resize(size=self.img_downsample_size)(mask_data)
                else:
                    # Provide a placeholder tensor if masks are not used
                    mask_data = 0
                
                if self.use_depth:
                    for cam_name in self.camera_names:
                        depth_image_dict[cam_name] = np.array(
                            root[f"/observations/depth_images/{cam_name}"]
                        )[img_sampling]

                    if compressed:
                        for cam_name in depth_image_dict.keys():
                            decompressed_depth_array = []
                            for i in range(len(depth_image_dict[cam_name])):
                                decompressed_depth = cv2.imdecode(
                                    depth_image_dict[cam_name][i], 1
                                )
                                # cv2.imshow('decoding', decompressed_depth)
                                # cv2.waitKey()
                                decompressed_depth_array.append(decompressed_depth)
                            depth_image_dict[cam_name] = np.array(
                                decompressed_depth_array
                            )
                else:
                    # Provide a placeholder tensor when depth is not used
                    depth_data = 0

                if self.use_text:
                    input_ids = root['/prompts/text/input_ids'][unique_indices]
                    attention_mask = root['/prompts/text/attention_mask'][unique_indices]
                    input_ids = [input_ids[i] for i in inverse_indices]
                    attention_mask = [attention_mask[i] for i in inverse_indices]
                    # Convert the Python lists to PyTorch tensors
                    input_ids = torch.stack([torch.tensor(x) for x in input_ids])
                    attention_mask = torch.stack([torch.tensor(x) for x in attention_mask])

                else:
                    # Provide a placeholder tensor when text is not used
                    input_ids = torch.tensor(0)
                    attention_mask = torch.tensor(0)

                # get all actions after and including start_ts
                action = action[start_ts:]
                action_len = episode_len - start_ts
                # get all states before and including start_ts
                robot_state = robot_state[: start_ts + 1]
                robot_state_len = start_ts + 1

            # self.is_sim = is_sim
            padded_action = np.zeros(
                (self.max_episode_len, original_action_shape[1]), dtype=np.float32
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len, dtype=np.bool_)
            is_pad[action_len:] = 1
            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[: self.chunk_size]

            padded_robot_state = np.zeros(
                (self.max_episode_len, original_robot_shape[1]), dtype=np.float32
            )
            padded_robot_state[:] = robot_state[0]  # padding with the first state
            padded_robot_state[-robot_state_len:] = robot_state

            if self.relative_obs_mode == False:
                padded_robot_state = padded_robot_state[-self.robot_obs_size :]
            else:
                padded_robot_state = padded_robot_state[-(self.robot_obs_size + 1) :]
            padded_robot_state = padded_robot_state[::-1] # reverse the order of list; padded_robot_state[0] is the current state

            t5 = time()
            # print('padded_robot_state: ', padded_robot_state[0, 0:6])
            # print('padded_action pre: ', padded_action[:10, 0:6])

            if self.relative_action_mode:
                for r in range(num_robots):
                    reference_state_pos = padded_robot_state[0, 3*r:3*(r+1)]
                    reference_state_euler = padded_robot_state[0, 3*num_robots+3*r:3*num_robots+3*(r+1)]
                    reference_state_rotation = Rotation.from_euler(
                        "ZYZ", reference_state_euler, degrees=True
                    )

                    for t in range(self.chunk_size):
                        # orientation
                        action_rotation = Rotation.from_euler(
                            "ZYZ", padded_action[t, 3*num_robots+3*r:3*num_robots+3*(r+1)], degrees=True
                        )
                        action_rotm = (
                            reference_state_rotation.as_matrix().transpose()
                            * action_rotation.as_matrix()
                        )
                        rel_action_rotation = Rotation.from_matrix(action_rotm)
                        padded_action[t, 3*num_robots+3*r:3*num_robots+3*(r+1)] = rel_action_rotation.as_rotvec()

                        # position
                        padded_action[t, 3*r:3*(r+1)] = (
                            reference_state_rotation.as_matrix().transpose()
                        ).dot(padded_action[t, 3*r:3*(r+1)] - reference_state_pos)
                        # print('padded_action: ', t, ', ', padded_action[t, 0:3], ', ', padded_action[t, 3:6])
                        # print( (reference_state_rotation.as_matrix()))
                        # print( (reference_state_rotation.as_matrix().transpose()))

            if self.relative_obs_mode:
                for r in range(num_robots):
                    reference_state_pos = padded_robot_state[0, 3*r:3*(r+1)]
                    reference_state_euler = padded_robot_state[0, 3*num_robots+3*r:3*num_robots+3*(r+1)]
                    reference_state_rotation = Rotation.from_euler(
                        "ZYZ", reference_state_euler, degrees=True
                    )
                    for t in range(self.robot_obs_size):
                        # orientation
                        robot_rotation = Rotation.from_euler(
                            "ZYZ", padded_robot_state[t + 1, 3*num_robots+3*r:3*num_robots+3*(r+1)], degrees=True
                        )
                        robot_rotm = (
                            reference_state_rotation.as_matrix().transpose()
                            * robot_rotation.as_matrix()
                        )
                        rel_robot_rotation = Rotation.from_matrix(robot_rotm)
                        padded_robot_state[t + 1, 3*num_robots+3*r:3*num_robots+3*(r+1)] = rel_robot_rotation.as_rotvec()
                        # position
                        padded_robot_state[t + 1, 3*r:3*(r+1)] = (
                            reference_state_rotation.as_matrix().transpose()
                        ).dot(padded_robot_state[t + 1, 3*r:3*(r+1)] - reference_state_pos)
                        padded_robot_state[:-1, 0:6*num_robots] = padded_robot_state[1:, 0:6*num_robots]  # pop relative current state which is always be identity

                assert len(padded_robot_state) == self.robot_obs_size + 1
                padded_robot_state = padded_robot_state[: self.robot_obs_size]  # discard the last trash data

            t6 = time()
            # new axis for different cameras
            all_cam_images = []
            for cam_name in self.camera_names:
                if cam_name == 'head_camera':
                    resized_frames = []
                    for t in range(len(image_dict[cam_name])):
                        resized = cv2.resize(image_dict[cam_name][t][:,:640,:], (self.img_downsample_size[1], self.img_downsample_size[0]), interpolation=cv2.INTER_LINEAR)
                        resized_frames.append(resized)

                    image_dict[cam_name] = np.stack(resized_frames, axis=0)
                    # print("2",image_dict[cam_name].shape)

                else:
                    resized_frames = []
                    for t in range(len(image_dict[cam_name])):
                        resized = cv2.resize(
                            image_dict[cam_name][t],
                            (self.img_downsample_size[1], self.img_downsample_size[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                        resized_frames.append(resized)
                    # print("4",resized_frames[0].shape)
                    image_dict[cam_name] = np.stack(resized_frames, axis=0)
                    # print("5",image_dict[cam_name].shape)
                all_cam_images.append(image_dict[cam_name])

            if self.use_depth:
                for cam_name in self.camera_names:
                    all_cam_images.append(depth_image_dict[cam_name])

            all_cam_images = np.stack(all_cam_images, axis=0)

            # construct torch data
            action_data = torch.from_numpy(np.array(padded_action)).float()
            robot_state_data = torch.from_numpy(np.array(padded_robot_state)).float()
            image_data = torch.from_numpy(np.array(all_cam_images))
            # depth_data = torch.from_numpy(np.array(all_depth_images))
            is_pad = torch.from_numpy(is_pad).bool()
            # channel last
            image_data = torch.einsum("k t h w c -> k t c h w", image_data)

            if self.img_debug:
                image_data_for_show = torch.einsum(
                    "c h w -> h w c", image_data[2, 1, :]
                ).numpy()
                image_data_for_show = cv2.cvtColor(
                    image_data_for_show, cv2.COLOR_RGB2BGR
                )
                # print(image_data_for_show.shape)
                cv2.imshow("original", image_data_for_show)
                cv2.waitKey(0)

            # augmentation
            if self.augment_images:
                original_size = image_data.shape[3:]
                # single_camera_size = original_size.copy()
                # single_camera_size[1] *= 0.5
                # print(original_size, single_camera_size)
                # ratio = 0.95
                self.transformations = [
                    # transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    # transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1]/2 * ratio)]),
                    # transforms.v2.Resize(size=self.img_downsample_size),
                    transforms.v2.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                ]

                for transform in self.transformations:
                    image_data = transform(image_data)

            if self.img_debug:
                image_data_for_show = torch.einsum(
                    "c h w -> h w c", image_data[2, 1, :]
                ).numpy()
                image_data_for_show = cv2.cvtColor(
                    image_data_for_show, cv2.COLOR_RGB2BGR
                )
                cv2.imshow("finally transfomed", image_data_for_show)
                cv2.waitKey(0)

            # if self.separate_left_right:
            #     image_size = image_data.shape[3:]
            #     for k in range(image_data.shape[0]):
            #         for t in range(image_data.shape[1]):
            #             temp_left_image = image_data[k, t, :, :, :int(image_size[1]/2)].clone()
            #             print('temp_left_image.shape: ', temp_left_image.shape)
            #             temp_right_image = image_data[k, t, :, :, int(image_size[1]/2):].clone()
            #             print('temp_right_image.shape: ', temp_right_image.shape)

            #             image_data[k, t, :, :, :int(image_size[1]/2)] = temp_left_image.clone()
            #             image_data[k, t, :, :, int(image_size[1]/2):] = temp_right_image.clone()

            #             if self.img_debug:
            #                 image_data_for_show = torch.einsum('c h w -> h w c', image_data[k, t, :]).numpy()
            #                 cv2.imshow('finally transfomed', image_data_for_show)
            #                 cv2.waitKey(0)

            # normalize image and change dtype to float
            image_data = image_data / 255.0
            if self.policy_class == "Diffusion":
                # normalize to [-1, 1]
                action_data = (
                    (action_data - self.norm_stats["action_min"])
                    / (self.norm_stats["action_max"] - self.norm_stats["action_min"])
                ) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (
                    action_data - self.norm_stats["action_mean"]
                ) / self.norm_stats["action_std"]
            # print('self.norm_stats["action_mean"]: ', self.norm_stats["action_mean"])
            # print('self.norm_stats["action_std"]: ', self.norm_stats["action_std"])
            # print('self.norm_stats["state_mean"]: ', self.norm_stats["state_mean"])
            # print('self.norm_stats["state_std"]: ', self.norm_stats["state_std"])
            # print('robot_state_data.shape: ', robot_state_data.shape)
            # print('action_data.shape: ', action_data.shape)

            robot_state_data = (
                robot_state_data - self.norm_stats["state_mean"]
            ) / self.norm_stats["state_std"]
        except (FileNotFoundError, IOError, Exception) as e:
            print(f"Error loading {dataset_path} in __getitem__: {e}")
            raise
        t9 = time()
        # print("duration 1: ", t1-t0)
        # print("duration 2: ", t2-t1)
        # print("duration 3: ", t3-t2)
        # print("duration 4: ", t4-t3)
        # print("duration 5: ", t5-t4)
        # print("duration 6: ", t6-t5)
        # print(image_data.dtype, qpos_data.dtype, action_data.dtype, is_pad.dtype)
        # print("image", image_data.shape, "mask", mask_data.shape) # 3 2 3 240 640 / 1 2 1 240 640
        return image_data, robot_state_data, action_data, is_pad, depth_data, mask_data, input_ids, attention_mask


def get_norm_stats(dataset_path_list):
    all_state_data = []
    all_action_data = []
    all_episode_len = []
    relative_inter_gripper_proprio = True
    relative_action_mode = False
    relative_obs_mode = False

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, "r") as root:
                num_robots = int(root["/actions/pose"].shape[1] / 6)

                action_xpos = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_euler = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_rotm6d = np.zeros((root["/actions/gripper_pos"].shape[0], 0))
                action_gripper = root["/actions/gripper_pos"][()]

                for r in range(num_robots):
                    action_xpos = np.concatenate(
                        (action_xpos, root["/actions/pose"][:, 6 * r : 6 * r + 3]),
                        axis=-1,
                    )
                    action_euler = np.concatenate(
                        (action_euler, root["/actions/pose"][:, 6 * r + 3 : 6 * r + 6]),
                        axis=-1,
                    )
                    action_rotation = Rotation.from_euler(
                        "ZYZ",
                        root["/actions/pose"][:, 6 * r + 3 : 6 * r + 6],
                        degrees=True,
                    )
                    action_rotm = action_rotation.as_matrix()
                    action_rotm6d = np.concatenate(
                        (action_rotm6d, action_rotm[:, :, 0], action_rotm[:, :, 1]),
                        axis=-1,
                    )

                if relative_action_mode:
                    action = np.concatenate(
                        [action_xpos, action_euler, action_gripper], axis=-1
                    )
                else:
                    action = np.concatenate(
                        [action_xpos, action_rotm6d, action_gripper], axis=-1
                    )

                xpos = root["/observations/xpos"][()]
                euler = root["/observations/euler"][()]
                
                rotm6d = np.zeros((euler.shape[0], 0))
                rel_xpos = np.zeros((euler.shape[0], 3))
                rel_rotm6d = np.zeros((euler.shape[0], 6))

                for r in range(num_robots):
                    rotation = Rotation.from_euler(
                        "ZYZ", euler[:, 3 * r : 3 * r + 3], degrees=True
                    )
                    rotm = rotation.as_matrix()
                    rotm6d = np.concatenate(
                        (rotm6d, rotm[:, :, 0], rotm[:, :, 1]), axis=-1
                    )

                gripper_pos = root["/observations/gripper_pos"][()]
                t1 = time()

                if relative_inter_gripper_proprio:
                    for t in range(euler.shape[0]):
                        reference_ee_pos = xpos[t, 0:3]
                        reference_ee_euler = euler[t, 0:3]
                        reference_ee_rotation = Rotation.from_euler(
                            "ZYZ", reference_ee_euler, degrees=True
                        )
                        reference_ee_rotm = reference_ee_rotation.as_matrix()

                        rel_xpos[t, :] = reference_ee_rotm.transpose().dot(
                            xpos[t, 3:6] - reference_ee_pos
                        )
                        right_hand_rotation = Rotation.from_euler(
                            "ZYZ", euler[t, 3:6], degrees=True
                        )
                        rel_rotm_temp = (
                            reference_ee_rotm.transpose()
                            * right_hand_rotation.as_matrix()
                        )
                        rel_rotm6d[t, :] = np.concatenate(
                            (rel_rotm_temp[:, 0], rel_rotm_temp[:, 1]), axis=-1
                        )
                    if relative_obs_mode:
                        state = np.concatenate(
                            [xpos, euler, rel_xpos, rel_rotm6d, gripper_pos], axis=-1
                        )
                    else:
                        state = np.concatenate(
                            [xpos, rotm6d, rel_xpos, rel_rotm6d, gripper_pos], axis=-1
                        )
                else:
                    if relative_obs_mode:
                        state = np.concatenate([xpos, euler, gripper_pos], axis=-1)
                    else:
                        state = np.concatenate([xpos, rotm6d, gripper_pos], axis=-1)

        except Exception as e:
            print(f"Error loading {dataset_path} in get_norm_stats")
            print(e)
            quit()
        all_state_data.append(torch.from_numpy(state))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(state))
    all_state_data = torch.cat(all_state_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
    
    # if relative_action_mode:
    #     action_xpos_size = action_xpos.shape[1]
    #     action_euler_size = action_euler.shape[1]
    #     action_mean[: (action_xpos_size + action_euler_size)] = torch.zeros(
    #         action_xpos_size + action_euler_size
    #     ).clone()
    #     action_std[: (action_xpos_size + action_euler_size)] = torch.ones(
    #         action_xpos_size + action_euler_size
    #     ).clone()

    # normalize qpos data
    state_mean = all_state_data.mean(dim=[0]).float()
    state_std = all_state_data.std(dim=[0]).float()
    state_std = torch.clip(state_std, 1e-2, np.inf)  # clipping
    
    # if relative_obs_mode:
    #     xpos_size = xpos.shape[1]
    #     euler_size = euler.shape[1]
    #     state_mean[: (xpos_size + euler_size)] = torch.zeros(
    #         xpos_size + euler_size
    #     ).clone()
    #     state_std[: (xpos_size + euler_size)] = torch.ones(
    #         xpos_size + euler_size
    #     ).clone()

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "state_mean": state_mean.numpy(),
        "state_std": state_std.numpy(),
    }

    print(f"action_mean = {action_mean}")
    print(f"action_std = {action_std}")
    print(f"state_mean = {state_mean}")
    print(f"state_std = {state_std}")

    return stats, all_episode_len


def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = (
        np.array(sample_weights) / np.sum(sample_weights)
        if sample_weights is not None
        else None
    )
    sum_dataset_len_l = np.cumsum(
        [0] + [np.sum(episode_len) for episode_len in episode_len_l]
    )
    # print(f'len(episode_len_l) = {len(episode_len_l)}')
    # print(f'episode_len_l = {episode_len_l}')
    # print(f'sample_probs = {sample_probs}')
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(
                sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1]
            )
            # print(f'episode_idx = {episode_idx}, step_idx={step_idx}')
            batch.append(step_idx)
        yield batch


def load_data(
    dataset_dir_l,
    name_filter,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size,
    robot_obs_size,
    img_obs_size,
    img_obs_skip,
    skip_mirrored_data=False,
    load_pretrain=False,
    policy_class=None,
    stats_dir_l=None,
    sample_weights=None,
    train_ratio=0.95,
    use_depth=False,
    use_masks=False,
    use_text=False
):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [
        find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l
    ]
    dataset_path_list_list = [
        [n for n in dataset_path_list if name_filter(n)]
        for dataset_path_list in dataset_path_list_list
    ]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]

    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]

    # num_episodes_l = [len(dataset_path_list)]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[: int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0) :]
    train_episode_ids_l = [train_episode_ids_0] + [
        np.arange(num_episodes) + num_episodes_cumsum[idx]
        for idx, num_episodes in enumerate(num_episodes_l[1:])
    ]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(
        f"\n\nData from: {dataset_dir_l}\n - Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n"
    )

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [
        [all_episode_len[i] for i in train_episode_ids]
        for train_episode_ids in train_episode_ids_l
    ]
    val_episode_len_l = [
        [all_episode_len[i] for i in val_episode_ids]
        for val_episode_ids in val_episode_ids_l
    ]
    
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(
        flatten_list(
            [find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]
        )
    )
    # print(f'train_episode_len_l: {train_episode_len_l}')
    # print(f'length of train_episode_len_l: {len(train_episode_len_l)}')
    # print(f"Norm stats from: {stats_dir_l}")


    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        train_episode_ids,
        train_episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        img_obs_skip,
        policy_class,
        use_depth,
        use_masks,
        use_text
    )
    val_dataset = EpisodicDataset(
        dataset_path_list,
        camera_names,
        norm_stats,
        val_episode_ids,
        val_episode_len,
        chunk_size,
        robot_obs_size,
        img_obs_size,
        img_obs_skip,
        policy_class,
        use_depth,
        use_masks,
        use_text
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=16,
        prefetch_factor=2,
        persistent_workers=True, 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=16,
        prefetch_factor=2,
        persistent_workers=True, 
    )

    return train_loader, val_loader, train_sampler, val_sampler, norm_stats, train_dataset.is_sim

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
