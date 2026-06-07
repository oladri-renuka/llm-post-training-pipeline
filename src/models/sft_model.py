"""
LLaMA-3.2-1B-Instruct with LoRA adapter injection for SFT.

LoRA is applied after model load, before any device placement,
to avoid the multi-device dispatch issue that breaks gradient flow.
"""

import logging
from typing import Tuple

import torch
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(
    cfg: DictConfig,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """
    Load LLaMA-3.2-1B-Instruct in bfloat16.

    Device placement is deferred — model is loaded to CPU first,
    then moved to GPU by the Trainer. This prevents the multi-device
    issue when LoRA is applied before dispatch.

    Args:
        cfg: Config with model fields.

    Returns:
        (model, tokenizer) tuple. Model has no LoRA adapters yet.
    """
    torch_dtype = getattr(torch, cfg.model.torch_dtype)

    logger.info("Loading base model: %s", cfg.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch_dtype,
        attn_implementation=cfg.model.attn_implementation,
        device_map=None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    logger.info("Loading tokenizer: %s", cfg.model.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    cfg: DictConfig,
) -> AutoModelForCausalLM:
    """
    Inject LoRA adapters into the model's attention projections.

    LoRA is applied before any device_map dispatch. This is the correct
    sequencing — applying LoRA after device_map="auto" splits layers
    across devices and breaks the adapter's parameter registration.

    Args:
        model: Base causal LM (on CPU, no device map).
        cfg: Config with lora fields.

    Returns:
        PEFT-wrapped model with trainable LoRA parameters only.
    """
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied — trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return model
