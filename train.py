import torch
import numpy as np
import os
import pickle
import argparse
from copy import deepcopy
from itertools import repeat
import logging
from tqdm import tqdm
import wandb
import time
import torchvision.transforms.v2 as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.init as init

from robot.constants import HZ
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed  # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

from detr.models.latent_model import Latent_Model_Transformer
from robot.constants import TASK_CONFIGS
import gc
import IPython

e = IPython.embed

def setup():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank, 
        world_size=world_size
        )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def expand_linear_weight_with_padding(model_layer, ckpt_weight):
    new_weight = model_layer.weight.data.clone()
    in_dim_ckpt = ckpt_weight.shape[1]
    new_weight[:,:in_dim_ckpt] = ckpt_weight
    new_weight[:, 20:] =0
    return new_weight

def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "Diffusion":
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def remove_module_prefix(state_dict):
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def train(rank, world_size, args):
    setup(rank, world_size)
    set_seed(args["seed"] + rank)

    is_wandb = args["wandb"]
    use_depth = args["use_depth"]
    use_masks = args["use_masks"]
    use_text = args["use_text"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_steps = args["num_steps"]
    eval_every = args["eval_every"]
    validate_every = args["validate_every"]
    save_every = args["save_every"]
    resume_ckpt_path = args["resume_ckpt_path"]

    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config["dataset_dir"]
    # num_episodes = task_config['num_episodes']
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    stats_dir = task_config.get("stats_dir", None)
    sample_weights = task_config.get("sample_weights", None)
    train_ratio = task_config.get("train_ratio", 0.99)
    name_filter = task_config.get("name_filter", lambda n: True)

    # fixed parameters
    state_dim = 29
    action_dim = 20
    lr_backbone = args["lr"]
    backbone = "resnet34"
    # backbone = "vit_b_16"
    # backbone = None
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "num_robot_observations": args["robot_obs_size"],
            "num_image_observations": args["img_obs_size"],
            "image_observation_skip": args["img_obs_every"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "vq": args["use_vq"],
            "vq_class": args["vq_class"],
            "vq_dim": args["vq_dim"],
            "action_dim": action_dim,
            "state_dim": state_dim,
            "no_encoder": args["no_encoder"],
            "use_depth": use_depth,
            "use_masks" : use_masks
        }
    else:
        raise NotImplementedError(f"policy class {policy_class} is not defined")

    actuator_config = {
        "actuator_network_dir": args["actuator_network_dir"],
        "history_len": args["history_len"],
        "future_len": args["future_len"],
        "prediction_len": args["prediction_len"],
    }


    train_loader, val_loader, train_sampler, val_sampler, norm_stats, is_sim = load_data(
        dataset_dir,
        name_filter,
        camera_names,
        batch_size_train,
        batch_size_val,
        args["chunk_size"],
        args["robot_obs_size"],
        args["img_obs_size"],
        args["img_obs_every"],
        args["skip_mirrored_data"],
        args["load_pretrain"],
        policy_class,
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
        use_depth=use_depth,
        use_masks=use_masks,
        use_text=use_text
    )

    config = {
        "num_steps": num_steps,
        "eval_every": eval_every,
        "validate_every": validate_every,
        "save_every": save_every,
        "ckpt_dir": ckpt_dir,
        "resume_ckpt_path": resume_ckpt_path,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "load_pretrain": args["load_pretrain"],
        "actuator_config": actuator_config,
        "is_wandb": is_wandb,
        "use_depth": use_depth,
        "use_masks": use_masks,
        "use_text": use_text,
        "batch_size": batch_size_train,
    }


    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    eval_every = config["eval_every"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]
    is_wandb = config["is_wandb"] and (rank == 0)

    policy = make_policy(policy_class, policy_config)
    policy.cuda(rank)
    for param in policy.model.parameters():
        param.requires_grad = True # False
    # for name, param in policy.model.named_parameters():
    #     if name.startswith("mask"):
    #         param.requires_grad = True
    #         if 'weight' in name:
    #             if param.dim() >= 2:
    #                 init.kaiming_normal_(param, mode ='fan_out', nonlinearity='relu')
    #         elif 'bias' in name:
    #             init.zeros_(param)
    policy = DDP(policy, device_ids=[rank], find_unused_parameters=True)
    optimizer = policy.module.configure_optimizers(lr_backbone, args['lr'], 1e-4)

    is_wandb = is_wandb and (rank==0)
    if is_wandb:
        expr_name = ckpt_dir.split("/")[-1]
        wandb.init(
            project="blocksort-headmono-without-mask",
            reinit=True,
            entity="donggunkim-kyung-hee-university",
            name=expr_name,
        )
        wandb.config.update(config)

    os.makedirs(ckpt_dir, exist_ok=True)
    config_path = os.path.join(ckpt_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)

    if config["resume_ckpt_path"] is not None:
        # map_location = lambda storage, loc: torch.device(f"cuda:{rank}")
        ckpt = torch.load(config["resume_ckpt_path"], map_location= f"cuda:{rank}")

        if 'model_state' in ckpt:
            loading_status = policy.module.deserialize(ckpt['model_state'])
        else:
            model_dict = remove_module_prefix(ckpt)
            status = policy.module.deserialize(model_dict)
            if rank == 0 and (status.missing_keys or status.unexpected_keys):
                print(f"Missing keys : {status.missing_keys} / Unexpected keys: {status.unexpected_keys}")


        start_step = ckpt.get('step', 0)
        if 'optim_state' in ckpt:
            optimizer.load_state_dict(ckpt['optim_state'])
            print(
                f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}, Step: {start_step}'
            )
        else:
            print("No optimizer found, starting fresh")

    else:
        start_step = 0
    step_per_epoch = len(train_loader)
    print(f'step_per_epoch: {step_per_epoch}')
    total_steps = args['num_steps']

    train_iter = iter(train_loader)
    current_epoch = 0
    
    progress_bar = tqdm(range(start_step,total_steps), desc=f"Training (GPU-{rank})")

    try:
        for step in progress_bar:       
            try:
                data = next(train_iter)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                train_iter = iter(train_loader)
                data = next(train_iter)    
                print(f'current_epoch: {current_epoch}')

            policy.train()
            image_data, robot_proprio_data, action_data, is_pad, depth_data, mask_data, input_ids, attention_mask = [d.cuda(rank) for d in data]

            forward_dict = policy(robot_proprio_data, image_data, depth_data, mask_data, input_ids, attention_mask, action_data, is_pad)
            loss = forward_dict["loss"]
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=loss.item())

            if is_wandb:
                wandb.log(forward_dict, step=step) 

            if step % save_every == 0 and rank ==0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
                torch.save({
                    'model_state': policy.module.serialize(),
                    'optim_state' : optimizer.state_dict(),
                    'step' : step,
                }, ckpt_path)

    finally:
        if is_wandb:
            wandb.finish()
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        torch.cuda.empty_cache()
        gc.collect()  # <<< 확실하게 garbage collection까지
        
        print('Finish the GPU multiprocessing')
        cleanup()



if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--ckpt_dir",
        action="store",
        type=str,
        default="/home/robros-ai/dg/robros_imitation_learning/ckpt/dsr_block_sort",
        help="ckpt_dir",
        required=True,
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        default="ACT",
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name",
        action="store",
        type=str,
        default="dsr_block_disassemble_and_sort",
        help="task_name",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        default=16,
        help="batch_size",
        required=True,
    )
    parser.add_argument(
        "--seed", action="store", type=int, default=0, help="seed", required=True
    )
    parser.add_argument(
        "--num_steps",
        action="store",
        type=int,
        default=50,
        help="num_steps",
        required=True,
    )

    parser.add_argument(
        "--lr", action="store", type=float, default=1e-4, help="lr", required=False
    )
    parser.add_argument("--load_pretrain", action="store_true", default=False)
    parser.add_argument(
        "--eval_every",
        action="store",
        type=int,
        default=1000,
        help="eval_every",
        required=False,
    )
    parser.add_argument(
        "--validate_every",
        action="store",
        type=int,
        default=1000,
        help="validate_every",
        required=False,
    )
    parser.add_argument(
        "--save_every",
        action="store",
        type=int,
        default=1000,
        help="save_every",
        required=False,
    )
    parser.add_argument(
        "--resume_ckpt_path",
        action="store",
        type=str,
        help="resume_ckpt_path",
        required=False,
    )
    parser.add_argument("--skip_mirrored_data", action="store_true")
    parser.add_argument(
        "--actuator_network_dir",
        action="store",
        type=str,
        help="actuator_network_dir",
        required=False,
    )
    parser.add_argument("--history_len", action="store", type=int)
    parser.add_argument("--future_len", action="store", type=int)
    parser.add_argument("--prediction_len", action="store", type=int)

    # for ACT
    parser.add_argument(
        "--kl_weight",
        action="store",
        type=int,
        default=10,
        help="KL Weight",
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        default=24,
        help="chunk_size",
        required=False,
    )
    parser.add_argument(
        "--robot_obs_size",
        action="store",
        type=int,
        default=2,
        help="robot state observation_size",
        required=False,
    )
    parser.add_argument(
        "--img_obs_size",
        action="store",
        type=int,
        default=2,
        help="image observation_size",
        required=False,
    )
    parser.add_argument(
        "--img_obs_every",
        action="store",
        type=int,
        default=10,
        help="image observation every n steps",
        required=False,
    )

    parser.add_argument("--use_depth", action="store_true", default=False)
    parser.add_argument("--use_masks", action="store_true", default=False)
    parser.add_argument("--use_text", action="store_true", default=False)

    parser.add_argument(
        "--hidden_dim",
        action="store",
        type=int,
        default=512,
        help="hidden_dim",
        required=False,
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        default=2048,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--use_vq", action="store_true")
    parser.add_argument("--vq_class", action="store", type=int, help="vq_class")
    parser.add_argument("--vq_dim", action="store", type=int, help="vq_dim")
    parser.add_argument("--no_encoder", action="store_true")

    args = vars(parser.parse_args())
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    train(rank, world_size, args)
