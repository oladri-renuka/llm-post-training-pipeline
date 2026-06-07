"""
LLaMA-3.2-1B-based scalar reward model for Bradley-Terry preference learning.

Architecture: LLaMA-3.2-1B encoder + linear projection on last token hidden state.
Using the same model family as the policy ensures the reward signal is coherent
with the policy's representation space.
"""

import logging
import os

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Scalar reward model built on LLaMA-3.2-1B backbone.

    The last token's hidden state is projected to a scalar reward.
    Using the final token (causal LM style) rather than [CLS] because
    LLaMA is a decoder-only model with no dedicated classification token.
    """

    def __init__(self, backbone_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            device_map=None,
        )
        # Freeze all backbone parameters except last transformer block
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        # Unfreeze last transformer block only
        for name, param in self.backbone.named_parameters():
            if "layers.15" in name:
                param.requires_grad = True

        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.01)

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
            output_hidden_states=True,
        )
        # Use last token hidden state (causal LM convention)
        last_hidden = outputs.hidden_states[-1]
        # Get the last non-padding token for each sequence
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_token_hidden = last_hidden[
            torch.arange(batch_size, device=input_ids.device), seq_lengths
        ]
        last_token_hidden = self.dropout(last_token_hidden.float())
        rewards = self.reward_head(last_token_hidden).squeeze(-1)
        return rewards


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise ranking loss."""
    return -torch.nn.functional.logsigmoid(
        chosen_rewards - rejected_rewards
    ).mean()


def compute_accuracy(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> float:
    return (chosen_rewards > rejected_rewards).float().mean().item()


def load_reward_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[RewardModel, PreTrainedTokenizerBase]:
    logger.info("Initializing LLaMA reward model: %s", cfg.model.backbone)
    model = RewardModel(
        backbone_name=cfg.model.backbone,
        dropout=cfg.model.dropout,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Reward model params — trainable: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )
    return model, tokenizer


def load_reward_model_from_checkpoint(
    checkpoint_path: str,
    backbone_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    dropout: float = 0.1,
) -> RewardModel:
    model = RewardModel(backbone_name=backbone_name, dropout=dropout)
    state_dict_path = os.path.join(checkpoint_path, "reward_model.pt")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded reward model from %s", state_dict_path)
    return model
