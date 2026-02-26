# VCA: Vision-Click-Action framework for precise manipulation of segmented objects


#### Project Website: https://robrosinc.github.io/vca/

This repo contains the implementation of VCA.

This repo was initially forked from the [ACT repo](https://github.com/tonyzhaozh/act).

### Updates:


### Repo Structure
- ``train.py`` Train VCA via DDP
- ``train_one.py`` Train VCA on one GPU
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``robot/constants.py`` Constants shared across files
- ``robot/record_episode_w_mask.py`` Collect robot's state-action data for training
- ``utils.py`` Utils such as data loading and helper functions


### Installation

    conda create -n act python=3.8.10
    conda activate act
    pip install -e .

- also need to install realtime sam2 submodule https://github.com/robrosinc/REALTIME_SAM2 by `git submodule update --init --recursive --remote` and download checkpoints

    cd external/tamapp/checkpoints
    ./download_checkpoints.sh


### Example Usages

First create ros1 package doosan_robot_server here: https://github.com/robrosinc/doosan_robot_serverConnect

Data collection:

    roscore
    roslaunch ros_tcp_endpoint teleop_camera_rviz.launch
    python mask_server.py --use_masks
    python robot/record_episode_w_mask.py --use_masks --task_name hanoi_mask

Training:
train for DDP on multiple GPUs(need to add distributedsampler in utils.py first):

    PYTHONWARNINGS=ignore torchrun --nproc_per_node=2 --master_port=12355 train.py --ckpt_dir /root/checkpoints --policy_class ACT --task_name hanoi_mask --use_mask --save_every 20000 --batch_size 64 --seed 10 --num_steps 200100 --img_obs_size 1 --wandb --lr 1e-4

train_one.py for single GPU:

    python train_one.py --ckpt_dir /root/checkpoints --policy_class ACT --task_name hanoi_mask --use_masks --save_every 20000 --batch_size 128 --seed 10 --num_steps 200100 --img_obs_size 1 --wandb --lr 1e-4

Inference:

    roscore
    rosrun doosan_robot_server doosan_robot_server_node
    roslaunch ros_tcp_endpoint teleop_camera_rviz.launch
    python mask_server.py --use_masks
    python mask_client_cpp.py --use_masks --ckpt_dir checkpoint/hanoi