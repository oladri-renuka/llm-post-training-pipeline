"""
PPO policy and value model setup.

Loads the SFT checkpoint (LoRA merged into base weights) and wraps it
with TRL's AutoModelForCausalLMWithValueHead for PPO training.
The reference model (frozen SFT policy) is loaded separately for KL penalty.
"""

import logging
from typing import Tuple

import torch
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


def load_ppo_models(
    cfg: DictConfig,
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoModelForCausalLMWithValueHead, PreTrainedTokenizerBase]:
    """
    Load the PPO policy model, frozen reference model, and tokenizer.

    The SFT LoRA checkpoint is merged into the base weights before PPO.
    Merging is required because:
      1. PPOTrainer expects a standard CausalLM, not a PEFT wrapper.
      2. The value head is added on top of the merged model.
      3. The reference model must be an independent copy, not shared weights.

    Args:
        cfg: Config with model and ppo fields.

    Returns:
        (policy_model, ref_model, tokenizer)
        - policy_model: Trainable, has value head.
        - ref_model: Frozen, used for KL divergence computation.
    """
    torch_dtype = getattr(torch, cfg.model.torch_dtype)
    sft_checkpoint = cfg.model.sft_checkpoint

    logger.info("Loading SFT base model from: %s", sft_checkpoint)

    base_model = AutoModelForCausalLM.from_pretrained(
        sft_checkpoint,
        torch_dtype=torch_dtype,
        device_map=None,
    )

    logger.info("Merging LoRA weights into base model")
    peft_model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    merged_model = peft_model.merge_and_unload()
    merged_model.config.use_cache = False

    logger.info("Wrapping policy model with value head")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        merged_model,
        torch_dtype=torch_dtype,
    )

    logger.info("Loading frozen reference model")
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        merged_model,
        torch_dtype=torch_dtype,
    )
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    logger.info(
        "Policy model — trainable: %s / %s (%.1f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    return policy_model, ref_model, tokenizer
