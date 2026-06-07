"""
PPO policy and value model setup.
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
    torch_dtype = getattr(torch, cfg.model.torch_dtype)
    sft_checkpoint = cfg.model.sft_checkpoint

    logger.info("Loading base model and merging LoRA from: %s", sft_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch_dtype,
        device_map=None,
    )
    peft_model = PeftModel.from_pretrained(base_model, sft_checkpoint)
    merged_model = peft_model.merge_and_unload()
    merged_model.config.use_cache = False

    logger.info("Wrapping policy model with value head")
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_model)

    logger.info("Loading frozen reference model")
    base_ref = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch_dtype,
        device_map=None,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_ref)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    logger.info(
        "Policy model — trainable: %s / %s (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return policy_model, ref_model, tokenizer
