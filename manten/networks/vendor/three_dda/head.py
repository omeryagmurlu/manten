# ruff: noqa: PLR0913

import einops
import torch
from torch import nn

from .layers import (
    FFWRelativeCrossAttentionModule,
    FFWRelativeSelfAttentionModule,
    FFWRelativeSelfCrossAttentionModule,
    ParallelAttention,
)
from .position_encodings import (
    RotaryPositionEncoding3D,
    SinusoidalPosEmb,
)


class DiffusionHead(nn.Module):
    def __init__(
        self,
        embedding_dim=60,
        num_attn_heads=8,
        use_instruction=False,
        rotation_dim=4,
        nhist=3,
        lang_enhanced=False,
    ):
        super().__init__()
        self.use_instruction = use_instruction
        self.lang_enhanced = lang_enhanced

        # Encoders
        self.traj_encoder = nn.Linear(9, embedding_dim)
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        # self.dim_2d_3d_emb = nn.Embedding(2, embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = nn.ModuleList(
            [
                ParallelAttention(
                    num_layers=1,
                    d_model=embedding_dim,
                    n_heads=num_attn_heads,
                    self_attention1=False,
                    self_attention2=False,
                    cross_attention1=True,
                    cross_attention2=False,
                    rotary_pe=False,
                    apply_ffn=False,
                )
            ]
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = FFWRelativeCrossAttentionModule(
            embedding_dim, num_attn_heads, num_layers=2, use_adaln=True
        )

        # Shared attention layers
        if not self.lang_enhanced:
            self.self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, num_layers=4, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim,
                num_attn_heads,
                num_self_attn_layers=4,
                num_cross_attn_layers=3,
                use_adaln=True,
            )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.rotation_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.rotation_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rotation_dim),
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        if not self.lang_enhanced:
            self.position_self_attn = FFWRelativeSelfAttentionModule(
                embedding_dim, num_attn_heads, 2, use_adaln=True
            )
        else:  # interleave cross-attention to language
            self.position_self_attn = FFWRelativeSelfCrossAttentionModule(
                embedding_dim, num_attn_heads, 2, 1, use_adaln=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 3)
        )

        # # 3. Joint Position
        # self.joint_proj = nn.Linear(embedding_dim, embedding_dim)
        # if not self.lang_enhanced:
        #     self.joint_self_attn = FFWRelativeSelfAttentionModule(
        #         embedding_dim, num_attn_heads, 2, use_adaln=True
        #     )
        # else:  # interleave cross-attention to language
        #     self.joint_self_attn = FFWRelativeSelfCrossAttentionModule(
        #         embedding_dim, num_attn_heads, 2, 1, use_adaln=True
        #     )
        # self.joint_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 7),
        # )

        # 4. Openness
        self.openness_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1)
        )

    def forward(
        self,
        trajectory,
        timestep,
        context_feats,
        context,
        instr_feats,
        adaln_gripper_feats,
        fps_feats,
        fps_pos,
        # has_3d,
    ):
        """
        Arguments:
            trajectory: (B, trajectory_length, 3+6+X)
            timestep: (B, 1)
            context_feats: (B, N, F)
            context: (B, N, F, 2)
            instr_feats: (B, max_instruction_length, F)
            adaln_gripper_feats: (B, nhist, F)
            fps_feats: (B, N, F), N < context_feats.size(1)
            fps_pos: (B, N, F, 2)
            has_3d: (B,) indicating if the observation is 3D
        """
        # Trajectory features
        traj_feats = self.traj_encoder(trajectory)  # (B, L, F)

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_feats.size(1), device=traj_feats.device)
        )[None].repeat(len(traj_feats), 1, 1)
        if self.use_instruction:
            traj_feats, _ = self.traj_lang_attention[0](
                seq1=traj_feats,
                seq1_key_padding_mask=None,
                seq2=instr_feats,
                seq2_key_padding_mask=None,
                seq1_pos=None,
                seq2_pos=None,
                seq1_sem_pos=traj_time_pos,
                seq2_sem_pos=None,
            )
        traj_feats = traj_feats + traj_time_pos

        # Predict position, rotation, opening
        traj_feats = einops.rearrange(traj_feats, "b l c -> l b c")
        context_feats = einops.rearrange(context_feats, "b l c -> l b c")
        adaln_gripper_feats = einops.rearrange(adaln_gripper_feats, "b l c -> l b c")
        fps_feats = einops.rearrange(fps_feats, "b l c -> l b c")
        (pos_pred, rot_pred, openess_pred) = self.prediction_head(
            trajectory[..., :3],
            traj_feats,
            context,
            context_feats,
            timestep,
            adaln_gripper_feats,
            fps_feats,
            fps_pos,
            instr_feats,
            # has_3d,
        )
        return (pos_pred, rot_pred, openess_pred)

    def prediction_head(
        self,
        gripper_pcd,
        gripper_features,
        rel_context_pos,
        context_features,
        timesteps,
        curr_gripper_features,
        sampled_context_features,
        sampled_rel_context_pos,
        instr_feats,
        # has_3d,
    ):
        """
        Compute the predicted action (position, rotation, opening).

        Args:
            gripper_pcd: A tensor of shape (B, N, 3)
            gripper_features: A tensor of shape (N, B, F)
            context_features: A tensor of shape (N, B, F)
            rel_context_pos: A tensor of shape (B, N, F, 2)
            timesteps: A tensor of shape (B,) indicating the diffusion step
            curr_gripper_features: A tensor of shape (M, B, F)
            sampled_context_features: A tensor of shape (K, B, F)
            sampled_rel_context_pos: A tensor of shape (B, K, F, 2)
            instr_feats: (B, max_instruction_length, F)
            has_3d: (B,) indicating if the observation is 3D
        """
        # Diffusion timestep
        time_embs = self.encode_denoising_timestep(timesteps, curr_gripper_features)

        # Positional embeddings
        rel_gripper_pos = self.relative_pe_layer(gripper_pcd)
        # rel_context_pos = self.relative_pe_layer(context_pcd)

        # # 2D vs 3D marker
        # dim_2d_3d = self.dim_2d_3d_emb(has_3d.long())

        # Cross attention from gripper to full context
        gripper_features = self.cross_attn(
            query=gripper_features,
            value=context_features,
            # value=context_features + dim_2d_3d,
            query_pos=rel_gripper_pos,
            value_pos=rel_context_pos,
            diff_ts=time_embs,
        )[-1]

        # Self attention among gripper and sampled context
        # features = torch.cat([gripper_features, sampled_context_features + dim_2d_3d], 0)
        features = torch.cat([gripper_features, sampled_context_features], 0)
        rel_pos = torch.cat([rel_gripper_pos, sampled_rel_context_pos], 1)
        features = self.self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]

        num_gripper = gripper_features.shape[0]

        # Rotation head
        rotation, _ = self.predict_rot(features, rel_pos, time_embs, num_gripper, instr_feats)

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, num_gripper, instr_feats
        )

        # Openness head from position head
        openness = self.openness_predictor(position_features)

        # return position, rotation, joints, openness
        return (position, rotation, openness)

    def encode_denoising_timestep(self, timestep, curr_gripper_features):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)

        curr_gripper_features = einops.rearrange(
            curr_gripper_features, "npts b c -> b npts c"
        )
        curr_gripper_features = curr_gripper_features.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(curr_gripper_features)
        return time_feats + curr_gripper_feats

    def predict_pos(self, features, rel_pos, time_embs, num_gripper, instr_feats):
        position_features = self.position_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]
        position_features = einops.rearrange(
            position_features[:num_gripper], "npts b c -> b npts c"
        )
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, rel_pos, time_embs, num_gripper, instr_feats):
        rotation_features = self.rotation_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]
        rotation_features = einops.rearrange(
            rotation_features[:num_gripper], "npts b c -> b npts c"
        )
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation, rotation_features

    def predict_joints(self, features, rel_pos, time_embs, num_gripper, instr_feats):
        joint_features = self.joint_self_attn(
            query=features,
            query_pos=rel_pos,
            diff_ts=time_embs,
            context=instr_feats,
            context_pos=None,
        )[-1]
        joint_features = einops.rearrange(
            joint_features[:num_gripper], "npts b c -> b npts c"
        )
        joint_features = self.joint_proj(joint_features)
        joints = self.joint_predictor(joint_features)
        return joints, joint_features
