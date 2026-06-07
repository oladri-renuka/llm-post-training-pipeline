"""
PPO policy and value model setup.
"""

import logging
from copy import deepcopy
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

    logger.info("Creating reference model as deepcopy of merged SFT model")
    ref_base = deepcopy(merged_model)
    ref_base.config.use_cache = False
    for param in ref_base.parameters():
        param.requires_grad = False
    ref_base.eval()

    logger.info("Wrapping policy model with value head")
    policy_model = AutoModelForCausalLMWithValueHead(merged_model)
    policy_model.is_peft_model = False
    policy_model.pretrained_model.config.is_encoder_decoder = False
    for param in policy_model.parameters():
        param.requires_grad = True

    logger.info("Wrapping reference model with value head")
    ref_model = AutoModelForCausalLMWithValueHead(ref_base)
    ref_model.is_peft_model = False
    ref_model.pretrained_model.config.is_encoder_decoder = False
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

    # Verify models are identical at initialization
    logger.info("Verifying models are identical at initialization...")
    test_ids = tokenizer("Hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        ref_logits = ref_base(test_ids).logits.float()
        policy_logits = merged_model(test_ids).logits.float()
    max_diff = (ref_logits - policy_logits).abs().max().item()
    logger.info("Max logit difference at init: %.8f (should be ~0.0)", max_diff)
    assert max_diff < 1e-4, f"Models differ at init: max_diff={max_diff}"

    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy_model.parameters())
    logger.info(
        "Policy model — trainable: %s / %s (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return policy_model, ref_model, tokenizer
