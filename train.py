import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
import logging
from tqdm import tqdm
from einops import rearrange
import wandb
import time
import torchvision.transforms.v2 as transforms
import GPUtil


from robot.constants import HZ
from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed  # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer


import IPython

e = IPython.embed


def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx + 1):
        if not os.path.isfile(os.path.join(dataset_dir, f"qpos_{i}.npy")):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def main(args):
    set_seed(1)
    # command line parameters
    is_wandb = args["wandb"]
    use_depth = args["use_depth"]
    use_masks = args["use_masks"]
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

    # get task parameters
    is_sim = task_name[:4] == "sim_"
    if is_sim or task_name == "all":
        from robot.constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from robot.constants import TASK_CONFIGS

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
        }
    elif policy_class == "Diffusion":

        policy_config = {
            "lr": args["lr"],
            "camera_names": camera_names,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "observation_horizon": 1,
            "action_horizon": 8,
            "prediction_horizon": args["chunk_size"],
            "num_queries": args["chunk_size"],
            "num_robot_observations": args["robot_obs_size"],
            "num_inference_timesteps": 10,
            "ema_power": 0.75,
            "vq": False,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError(f"policy class {policy_class} is not defined")

    actuator_config = {
        "actuator_network_dir": args["actuator_network_dir"],
        "history_len": args["history_len"],
        "future_len": args["future_len"],
        "prediction_len": args["prediction_len"],
    }

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
        "wandb": is_wandb,
        "use_masks": use_masks
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, "config.pkl")
    expr_name = ckpt_dir.split("/")[-1]

    if is_wandb:
        wandb.init(
            project="SAMIL",
            reinit=True,
            entity="donggunkim-kyung-hee-university",
            name=expr_name,
        )
        wandb.config.update(config)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    train_dataloader, val_dataloader, stats, _ = load_data(
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
        config["load_pretrain"],
        policy_class,
        stats_dir_l=stats_dir,
        sample_weights=sample_weights,
        train_ratio=train_ratio,
        use_depth=use_depth,
    )

    # GPUtil.showUtilization()
    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}")
    if is_wandb:
        wandb.finish()

    torch.cuda.empty_cache()


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


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    elif policy_class == "Diffusion":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print("rand crop resize is used!")
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[
            ...,
            int(original_size[0] * (1 - ratio) / 2) : int(
                original_size[0] * (1 + ratio) / 2
            ),
            int(original_size[1] * (1 - ratio) / 2) : int(
                original_size[1] * (1 + ratio) / 2
            ),
        ]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)

    return curr_image


def forward_pass(data, policy):
    image_data, robot_proprio_data, action_data, is_pad = data
    image_data, robot_proprio_data, action_data, is_pad = (
        image_data.cuda(),
        robot_proprio_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
    )
    return policy(
        robot_proprio_data, image_data, action_data, is_pad
    )  # TODO remove None

def forward_pass_with_masks(data, policy):
    image_data, robot_proprio_data, action_data, is_pad, mask_data = data
    image_data, robot_proprio_data, action_data, is_pad, mask_data = (
        image_data.cuda(),
        robot_proprio_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
        mask_data.cuda(),
    )

    mask_data_expanded = mask_data.expand(-1, 3, -1, 3, -1, -1)
    image_data = torch.cat((image_data, mask_data_expanded), dim=1)
    return policy(
        robot_proprio_data, image_data, action_data, is_pad
    )  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config["num_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    eval_every = config["eval_every"]
    validate_every = config["validate_every"]
    save_every = config["save_every"]
    is_wandb = config["wandb"]
    use_masks = config["use_masks"]

    set_seed(seed)
    validation_iteration = 50
    train_iteration = 5e2
    
    policy = make_policy(policy_class, policy_config)
    if config["load_pretrain"]:
        # loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f"loaded! {loading_status}")

    if config["resume_ckpt_path"] is not None:
        checkpoint = torch.load(config["resume_ckpt_path"], map_location= lambda storage, loc : storage.cuda(0))
        loading_status = policy.deserialize(checkpoint)
        print(
            f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}'
        )
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None

    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps), dynamic_ncols=True):
        # validation
        if False and step % validate_every == 0:
            print("validating")

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                for batch_idx, data in tqdm(
                    enumerate(val_dataloader),
                    total=validation_iteration,
                    dynamic_ncols=True,
                ):
                    # t0 = time.time()
                    if use_masks:
                        forward_dict = forward_pass_with_masks(data, policy)
                    else:
                        forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    # t1 = time.time()
                    # print(t1-t0)
                    if batch_idx >= validation_iteration:
                        break

                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary["loss"]
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f"val_{k}"] = validation_summary.pop(k)
            if is_wandb:
                wandb.log(validation_summary, step=step)
            print(f"Val loss:   {epoch_val_loss:.5f}")
            summary_string = ""
            for k, v in validation_summary.items():
                summary_string += f"{k}: {v.item():.3f} "
            print(summary_string)

        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f"policy_step_{step}_seed_{seed}.ckpt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
            # if is_wandb:
            #     wandb.log({'success': success}, step=step)

        # training
        policy.train()

        data = next(train_dataloader)
        if use_masks:
            forward_dict = forward_pass_with_masks(data, policy)
        else:
            forward_dict = forward_pass(data, policy)

        # backward
        loss = forward_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if is_wandb:
            wandb.log(forward_dict, step=step)  # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_step_{step}_seed_{seed}.ckpt")
            torch.save(policy.serialize(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_step_{best_step}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}"
    )

    return best_ckpt_info


def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f"Epoch {epoch} done")
        epoch += 1


if __name__ == "__main__":
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
        default=100000,
        help="num_steps",
        required=True,
    )

    parser.add_argument(
        "--lr", action="store", type=float, default=1e-5, help="lr", required=False
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
        default=40,
        help="chunk_size",
        required=False,
    )
    parser.add_argument(
        "--robot_obs_size",
        action="store",
        type=int,
        default=40,
        help="robot state observation_size",
        required=False,
    )
    parser.add_argument(
        "--img_obs_size",
        action="store",
        type=int,
        default=1,
        help="image observation_size",
        required=False,
    )
    parser.add_argument(
        "--img_obs_every",
        action="store",
        type=int,
        default=1,
        help="image observation every n steps",
        required=False,
    )
    parser.add_argument("--use_depth", action="store_true", default=False)
    parser.add_argument("--use_masks", action="store_true", default=False)
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

    main(vars(parser.parse_args()))
