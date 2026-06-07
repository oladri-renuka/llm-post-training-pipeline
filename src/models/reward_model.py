"""
DistilBERT-based scalar reward model for Bradley-Terry preference learning.

Architecture: DistilBERT encoder + linear projection to scalar reward.
The scalar reward is used directly in the Bradley-Terry loss during training
and as the PPO reward signal during Phase 3.
"""

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Scalar reward model built on a DistilBERT backbone.

    The [CLS] token representation is projected to a single scalar.
    No sigmoid — raw logits are used in Bradley-Terry loss to avoid
    gradient saturation at the extremes.
    """

    def __init__(self, backbone_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.reward_head = nn.Linear(hidden_size, 1)

        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            rewards: (batch_size,) scalar reward per example.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)
        rewards = self.reward_head(cls_repr).squeeze(-1)
        return rewards


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss.

    Maximizes log P(chosen > rejected) = log sigmoid(r_chosen - r_rejected).

    Args:
        chosen_rewards: (batch_size,) rewards for preferred responses.
        rejected_rewards: (batch_size,) rewards for dispreferred responses.

    Returns:
        Scalar loss.
    """
    return -torch.nn.functional.logsigmoid(
        chosen_rewards - rejected_rewards
    ).mean()


def compute_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    """Fraction of pairs where chosen reward exceeds rejected reward."""
    return (chosen_rewards > rejected_rewards).float().mean().item()


def load_reward_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[RewardModel, PreTrainedTokenizerBase]:
    """
    Instantiate reward model and tokenizer from config.

    Args:
        cfg: Config with model fields.

    Returns:
        (reward_model, tokenizer) tuple.
    """
    logger.info("Initializing reward model: %s", cfg.model.backbone)
    model = RewardModel(
        backbone_name=cfg.model.backbone,
        dropout=cfg.model.dropout,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Reward model params — total: %s | trainable: %s",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    return model, tokenizer


def load_reward_model_from_checkpoint(
    checkpoint_path: str,
    backbone_name: str = "distilbert-base-uncased",
    dropout: float = 0.1,
) -> RewardModel:
    """
    Load a trained reward model from a saved state dict.

    Args:
        checkpoint_path: Path to directory containing reward_model.pt.
        backbone_name: DistilBERT variant used during training.
        dropout: Dropout value used during training.

    Returns:
        Reward model in eval mode.
    """
    import os
    model = RewardModel(backbone_name=backbone_name, dropout=dropout)
    state_dict_path = os.path.join(checkpoint_path, "reward_model.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded reward model from %s", state_dict_path)
    return model
