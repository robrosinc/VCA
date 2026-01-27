# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone, build_mask_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import time
import numpy as np
import os, logging
import torch.distributed as dist

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, mask_backbones, text_encoder, transformer, encoder, args, log_file="nan_check.log"):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = args.num_queries
        self.num_robot_observations = args.num_robot_observations
        self.num_image_observations = args.num_image_observations
        self.image_observation_skip = args.image_observation_skip
        self.camera_names = args.camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = args.vq, args.vq_class, args.vq_dim
        self.state_dim, self.action_dim = args.state_dim, args.action_dim
        self.use_depth = args.use_depth
        self.use_masks = args.use_masks
        self.use_text = args.use_text
        self.use_slot_attention = False
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, args.action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(self.state_dim*self.num_robot_observations, hidden_dim)
            if mask_backbones is not None:
                self.input_proj_masks = nn.Conv2d(mask_backbones[0].num_channels, hidden_dim, kernel_size = 1)
                self.mask_backbones = nn.ModuleList(mask_backbones)
            elif text_encoder is not None:
                self.text_encoder = text_encoder
                self.input_proj_text = nn.Linear(text_encoder.output_dim, hidden_dim)
                self.text_pos_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
                if self.use_slot_attention: # reserve for slot attention
                    self.num_slots = 6 # TODO tune
                    self.slot_query = nn.Parameter(torch.randn(self.num_slots, hidden_dim))
                    self.slot_proj = nn.Linear(hidden_dim, hidden_dim)
                    self.slot_norm = nn.LayerNorm(hidden_dim)
                    self.slot_self_attn = nn.TransformerEncoderLayer(
                        d_model=hidden_dim,
                        nhead=4,
                        dim_feedforward=hidden_dim * 2,
                        dropout=0.1,
                        batch_first=True
                    )
                    self.slot_self_attn_layers = 2 # TODO tune
                    self.register_buffer("spatial_coords", None, persistent=False)
                    self.text_to_slot = nn.Linear(hidden_dim, hidden_dim)
                    self.rank_tau = 0.1
                    self.slot_pos_mlp = nn.Sequential(
                        nn.Linear(2, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.slot_rank_mlp = nn.Sequential(
                        nn.Linear(2, hidden_dim),  # (rank_x, rank_y)
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim)
                    )
                    self.slot_presence_head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim // 2, 1)
                    )


        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(self.state_dim*self.num_robot_observations, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
            print("backbones is None")

        # encoder extra parameters
        self.latent_dim = 64 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(self.action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(self.state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+self.num_robot_observations+self.num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Only rank 0 writes to log
        if self.rank == 0:
            os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
            self.logger = logging.getLogger("nan_check")
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:  # avoid duplicate handlers
                file_handler = logging.FileHandler(log_file)
                formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        else:
            self.logger = None

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs = qpos.shape[0] # batch size
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # CVAE's encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, num_obs, hidden_dim)
                # qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+num_obs, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+num_obs, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 1+self.num_robot_observations), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+num_obs)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+num_obs, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    # latent_sample = torch.ones([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    # latent_sample = latent_sample*1.0
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def extract_object_slots(self, feat_tokens, text_color=None):
        """
        feat_tokens: (S, B, C)
        text_color:  (B, C) or None
        returns:
            slots: (B, K, C)
            attn:  (B, K, S)
        """

        feat_tokens = feat_tokens
        S, B, C = feat_tokens.shape
        x = feat_tokens.permute(1, 0, 2)  # (B, S, C)

        # ---- slot query ----
        q = self.slot_query.unsqueeze(0).expand(B, -1, -1)  # (B, K, C)

        if text_color is not None:
            q = q + self.text_to_slot(text_color).unsqueeze(1)

        q = self.slot_proj(q)

        # ---- attention ----
        scale = C ** -0.5
        attn_logits = torch.einsum("bkc,bsc->bks", q, x) * scale
        attn = torch.softmax(attn_logits, dim=-1)

        slots = torch.einsum("bks,bsc->bkc", attn, x)
        slots = self.slot_norm(slots)
        presence_logits = self.slot_presence_head(slots).squeeze(-1)  # (B, K)
        presence = torch.sigmoid(presence_logits)

        return slots, attn, presence, presence_logits
    
    def soft_rank(self, coord, tau):
        """
        coord: (B, K)
        return: soft rank in [0, K-1], shape (B, K)
        """
        diff = coord.unsqueeze(-1) - coord.unsqueeze(-2)  # (B, K, K)
        prob = torch.sigmoid(diff / tau)
        rank = prob.sum(dim=-1)
        return rank

    # def check_nan(self, x, name):
    #     """
    #     Checks a tensor for NaN or Inf and logs details.
    #     Includes tensor shape and device.
    #     Only logs for rank 0 in DDP.
    #     """
    #     if x is None or not torch.is_tensor(x):
    #         return

    #     if self.logger is None:  # Only log on rank 0
    #         return

    #     shape_info = f"shape={tuple(x.shape)}, device={x.device}"

    #     if torch.isnan(x).any():
    #         msg = (
    #             f"[NaN DETECTED] {name} contains NaN ({shape_info})\n"
    #             f"  min={x.nanmin().item()}, max={x.nanmax().item()}"
    #         )
    #         self.logger.error(msg)

    #     if torch.isinf(x).any():
    #         finite_mask = ~torch.isinf(x)
    #         finite_min = x[finite_mask].min().item() if finite_mask.any() else "all inf"
    #         finite_max = x[finite_mask].max().item() if finite_mask.any() else "all inf"
    #         msg = (
    #             f"[INF DETECTED] {name} contains Inf ({shape_info})\n"
    #             f"  min={finite_min}, max={finite_max}"
    #         )
    #         self.logger.error(msg)

    def forward(self, qpos, image, env_state, depth = None, masks = None, input_ids = None, attention_mask = None, actions=None, is_pad=None, vq_sample=None, encoding_only=False):
        """
        qpos: batch, num_obs, robot_state_dim
        image: batch, num_cam, num_obs, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # print(f'qpos shape: {qpos.shape}')
        # start_time = time.time()
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        # self.check_nan(mu, "mu")
        # self.check_nan(logvar, "logvar")

        if encoding_only is True:
            return mu, logvar
        # print("qpos",qpos.shape,"image", image.shape,"input_ids", input_ids.shape)
        # print("latent_input", latent_input.shape)
        bs = qpos.shape[0]
        # cvae decoder
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            text_vec = None
            slot_tokens = []
            slot_pos_tokens = []
            for i in range(len(self.backbones)):
                for t in range(self.num_image_observations):
                    features, pos = self.backbones[i](image[:, i, t])
                    features = features[0]  # (B, C, H, W)
                    pos = pos[0]            # (B, C, H, W)

                    features = self.input_proj(features)

                    if self.use_text:
                        # ---- flatten vision immediately ----
                        B, C, H, W = features.shape

                        feat_tokens = features.flatten(2).permute(2, 0, 1)  # (HW, B, C)
                        pos_tokens  = pos.flatten(2).permute(2, 0, 1)       # (HW, B, C)

                        all_cam_features.append(feat_tokens)
                        all_cam_pos.append(pos_tokens)

                        if self.use_slot_attention:
                            if self.spatial_coords is None or self.spatial_coords.shape[0] != H * W:
                                y, x = torch.meshgrid(
                                    torch.linspace(-1, 1, H, device=features.device),
                                    torch.linspace(-1, 1, W, device=features.device),
                                    indexing="ij"
                                )
                            coords = torch.stack([x, y], dim=-1)      # (H, W, 2)
                            self.spatial_coords = coords.view(-1, 2)  # (S, 2)

                        if i == 0:
                            text_feat = self.text_encoder(
                                input_ids[:, t],
                                attention_mask=attention_mask[:, t]
                            )
                            # print("text_feat", text_feat.shape)
                            text_mask = attention_mask[:, t].unsqueeze(-1)
                            masked_text_feat = (text_feat * text_mask).sum(dim=1) / text_mask.sum(dim=1)
                            # print("text_vec", text_vec.shape)
                            text_vec = masked_text_feat
                            if self.use_slot_attention:
                                slots, attn, presence, presence_logits = self.extract_object_slots(feat_tokens, pos_tokens, text_feat[:, 1:3])  # slots: (B, K, C), attn: (B, K, S)
                                # self.check_nan(attn, "attn")
                                # self attention to make slots to go for different regions
                                for _ in range(self.slot_self_attn_layers):
                                    delta = self.slot_self_attn(slots)
                                    slots = slots + presence.unsqueeze(-1) * delta

                                # spatial centroid injection (S,2)
                                # bmm instead of einsum
                                spatial_coords_expanded = self.spatial_coords.unsqueeze(0).expand(B, -1, -1)
                                slot_centroids = torch.bmm(attn, spatial_coords_expanded) 
                                # slot_centroids = torch.einsum("bks,sd->bkd", attn, self.spatial_coords)  # (B, K, 2)
                                attn_mass = attn.sum(dim=-1, keepdim=True)  # (B,K,1)
                                slot_centroids = slot_centroids / (attn_mass + 1e-6)
                                slot_centroids = slot_centroids * presence.unsqueeze(-1)

                                x = slot_centroids[..., 0]  # (B, K)
                                y = slot_centroids[..., 1] 
                                NEG_INF = -1e4

                                x_masked = x + (1-presence) * NEG_INF
                                y_masked = y + (1-presence) * NEG_INF

                                rank_x = self.soft_rank(x_masked, self.rank_tau)
                                # self.check_nan(rank_x, "rank_x")
                                rank_y = self.soft_rank(y_masked, self.rank_tau)
                                # self.check_nan(rank_y, "rank_y")

                                rank_x = rank_x / (self.num_slots - 1 + 1e-6)
                                rank_y = rank_y / (self.num_slots - 1 + 1e-6)
                                rank_xy = torch.stack([rank_x, rank_y], dim=-1)  # (B, K, 2)

                                slot_pos_abs = self.slot_pos_mlp(slot_centroids)  # (B, K, C)
                                rank_embed = self.slot_rank_mlp(rank_xy)
                                slot_pos = slot_pos_abs + rank_embed
                                slots = slots + presence.unsqueeze(-1) * slot_pos

                                sim = torch.bmm(text_vec.unsqueeze(1), slot_pos.transpose(1, 2)).squeeze(1)  # (B, K)
                                # sim = torch.einsum("bc,bkc->bk", text_vec, slot_pos)
                                slot_weights = torch.softmax(sim, dim=-1)
                                selected_slot = torch.bmm(slot_weights.unsqueeze(1), slots).squeeze(1)  # (B, C)
                                selected_slot_pos = torch.bmm(slot_weights.unsqueeze(1), slot_pos).squeeze(1)  # (B, C)
                                # selected_slot = torch.einsum("bk,bkc->bc", slot_weights, slots)
                                # selected_slot_pos = torch.einsum("bk,bkc->bc", slot_weights, slot_pos)

                                # print("selected_slot", selected_slot.shape)
                                slot_token = selected_slot.unsqueeze(0)  # (1, B, C)
                                slot_pos = selected_slot_pos.unsqueeze(0)  # (1, B, C)

                                slot_tokens.append(slot_token)
                                slot_pos_tokens.append(slot_pos)
                                # print("slot_tokens", slot_token.shape, "slot_pos_tokens", slot_pos.shape)
                    else:
                        # ---- keep 4D ----
                        all_cam_features.append(features)
                        all_cam_pos.append(pos)

                    del features, pos

            if self.use_masks:
                for i in range(len(self.mask_backbones)):
                    for t in range(self.num_image_observations):
                        # print("1", masks.shape) # 1 1 2 1 240 640
                        # print("2", masks[:,i,t].shape) # 1 1 240 640
                        features, pos = self.mask_backbones[i](masks[:, i, t])
                        features = features[0]  # (B, C, H, W)
                        pos = pos[0]

                        features = self.input_proj_masks(features)

                        if self.use_text:
                            # ---- token mode ----
                            B, C, H, W = features.shape
                            feat_tokens = features.flatten(2).permute(2, 0, 1)  # (HW, B, C)
                            pos_tokens  = pos.flatten(2).permute(2, 0, 1)

                            all_cam_features.append(feat_tokens)
                            all_cam_pos.append(pos_tokens)
                        else:
                            # ---- spatial mode ----
                            all_cam_features.append(features)
                            all_cam_pos.append(pos)

                        del features, pos
            # print(all_cam_pos[0].shape)  # (B, hidden_dim, H, W)
            if self.use_text:
                # concatenate vision tokens
                vis_src = torch.cat(all_cam_features, dim=0)  # (S_vis, B, C)
                vis_pos = torch.cat(all_cam_pos, dim=0).repeat(1, bs, 1)  # (S_vis, B, C)

                if self.use_slot_attention:
                    # concatenate slot tokens across time
                    slot_src = torch.cat(slot_tokens, dim=0)  # (T, B, C)
                    slot_pos = torch.cat(slot_pos_tokens, dim=0) # (T, B, C)

                    # final src: [slots | vision]
                    src = torch.cat([slot_src, vis_src], dim=0)  # src.shape = (S_total, B, C)
                    pos = torch.cat([slot_pos, vis_pos], dim=0)  # pos.shape = (S_total, B, C)

                    print("src", src.shape, "pos", pos.shape)
                else:
                    src = torch.cat([text_vec, vis_src], dim=0)  # src.shape = (S_total, B, C)
                    pos = torch.cat([self.text_pos_embedding.repeat(1,bs,1), vis_pos], dim=0)

            else:
                src = torch.cat(all_cam_features, axis=3) # B, C, H, W
                pos = torch.cat(all_cam_pos, axis=3) # 1, C, H, W

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos.reshape(bs, -1))
            # fold camera dimension into width dimension

            # print(src.shape) # B 512 8 160  for each camera 40 if obs_img is 2, 20 if 1
            # print(pos.shape) # B 320*n + seq_len, 512 for n camera
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        # end_time = time.time()
        # print(f'DETR VAE forward time: {end_time - start_time:.4f} sec')
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries



class CNNMLP(nn.Module):

    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=self.action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    if args.use_depth:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)

    mask_backbones = None
    if args.use_masks:
        mask_backbones = []
        for name in args.camera_names:
            if name == "head_camera":
                mask_backbone = build_mask_backbone(args)
                mask_backbones.append(mask_backbone)
    text_encoder = None
    if args.use_text:
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-small')
        from .bert import build_bert
        text_encoder = build_bert(args)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        mask_backbones,
        text_encoder,
        transformer,
        encoder,
        args,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 7 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

