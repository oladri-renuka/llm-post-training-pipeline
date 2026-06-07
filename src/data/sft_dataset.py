"""
Alpaca dataset loader and formatter for supervised fine-tuning.

Formats instruction-input-output triplets into the Alpaca prompt template
and tokenizes for causal language modeling.
"""

import logging
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task"
    "{input_section}"
    ". Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)

_ALPACA_INPUT_SECTION = ", paired with an input that provides further context"
_ALPACA_INPUT_BLOCK = "### Input:\n{input}\n\n"


def _format_alpaca_example(example: Dict) -> Dict:
    """Format a single Alpaca example into the prompt template."""
    has_input = bool(example.get("input", "").strip())

    input_section = _ALPACA_INPUT_SECTION if has_input else ""
    input_block = (
        _ALPACA_INPUT_BLOCK.format(input=example["input"]) if has_input else ""
    )

    text = _ALPACA_PROMPT_TEMPLATE.format(
        input_section=input_section,
        instruction=example["instruction"],
        input_block=input_block,
        output=example["output"],
    )
    return {"text": text}


def _tokenize(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict:
    """Tokenize a single formatted example with truncation."""
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_sft_dataset(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Load and preprocess the Alpaca dataset for SFT.

    Args:
        cfg: Hydra/OmegaConf config with data and training fields.
        tokenizer: Tokenizer matching the base model.

    Returns:
        Tokenized HuggingFace Dataset ready for SFTTrainer.
    """
    logger.info("Loading dataset: %s", cfg.data.dataset_name)
    raw = load_dataset(cfg.data.dataset_name, split=cfg.data.train_split)
    logger.info("Raw dataset size: %d examples", len(raw))

    formatted = raw.map(
        _format_alpaca_example,
        num_proc=cfg.data.num_proc,
        desc="Formatting Alpaca examples",
        remove_columns=raw.column_names,
    )

    tokenized = formatted.map(
        lambda ex: _tokenize(ex, tokenizer, cfg.data.max_length),
        num_proc=cfg.data.num_proc,
        desc="Tokenizing",
        remove_columns=["text"],
    )

    logger.info("Tokenized dataset size: %d examples", len(tokenized))
    return tokenized


def get_alpaca_prompt_only(example: Dict) -> str:
    """
    Return the prompt portion of an Alpaca example (no output).
    Used during PPO rollout to generate responses.
    """
    has_input = bool(example.get("input", "").strip())
    input_section = _ALPACA_INPUT_SECTION if has_input else ""
    input_block = (
        _ALPACA_INPUT_BLOCK.format(input=example["input"]) if has_input else ""
    )
    return (
        "Below is an instruction that describes a task"
        f"{input_section}"
        ". Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n"
        f"{input_block}"
        "### Response:\n"
    )
