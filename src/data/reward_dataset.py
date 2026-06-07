"""
UltraFeedback preference pair loader for reward model training.

Extracts chosen/rejected pairs from UltraFeedback and formats them
for Bradley-Terry contrastive training.
"""

import logging
from typing import Dict, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _extract_preference_pair(example: Dict) -> Optional[Dict]:
    """
    Extract a chosen/rejected pair from a single UltraFeedback example.

    UltraFeedback stores completions as a list of dicts with 'response'
    and 'overall_score' fields. We select the highest and lowest scored
    responses as the chosen/rejected pair.

    Returns None if the example cannot yield a valid pair.
    """
    instruction = example.get("instruction", "").strip()
    completions = example.get("completions", [])

    if not instruction or len(completions) < 2:
        return None

    scored = []
    for c in completions:
        try:
            score = float(c.get("overall_score", 0))
            response = c.get("response", "").strip()
            if response:
                scored.append((score, response))
        except (TypeError, ValueError):
            continue

    if len(scored) < 2:
        return None

    scored.sort(key=lambda x: x[0])
    rejected_score, rejected = scored[0]
    chosen_score, chosen = scored[-1]

    if chosen_score <= rejected_score:
        return None

    return {
        "instruction": instruction,
        "chosen": chosen,
        "rejected": rejected,
    }


def _tokenize_pair(
    example: Dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict:
    """Tokenize chosen and rejected responses independently."""
    chosen_text = f"{example['instruction']}\n\n{example['chosen']}"
    rejected_text = f"{example['instruction']}\n\n{example['rejected']}"

    chosen_enc = tokenizer(
        chosen_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    rejected_enc = tokenizer(
        rejected_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }


def load_reward_dataset(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetDict:
    """
    Load and preprocess UltraFeedback for reward model training.

    Args:
        cfg: Config with data and training fields.
        tokenizer: DistilBERT tokenizer.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    logger.info("Loading dataset: %s", cfg.data.dataset_name)
    raw = load_dataset(cfg.data.dataset_name, split="train")

    if cfg.data.num_samples < len(raw):
        raw = raw.select(range(cfg.data.num_samples))
        logger.info("Subsampled to %d examples", cfg.data.num_samples)

    logger.info("Extracting preference pairs...")
    pairs = raw.map(
        _extract_preference_pair,
        num_proc=cfg.data.num_proc,
        desc="Extracting pairs",
        remove_columns=raw.column_names,
    )
    pairs = pairs.filter(lambda ex: ex["instruction"] is not None)
    logger.info("Valid pairs extracted: %d", len(pairs))

    tokenized = pairs.map(
        lambda ex: _tokenize_pair(ex, tokenizer, cfg.data.max_length),
        num_proc=cfg.data.num_proc,
        desc="Tokenizing pairs",
        remove_columns=["instruction", "chosen", "rejected"],
    )

    split = tokenized.train_test_split(
        test_size=1 - cfg.data.train_val_split,
        seed=42,
    )
    logger.info(
        "Train: %d | Validation: %d",
        len(split["train"]),
        len(split["test"]),
    )
    return DatasetDict({"train": split["train"], "validation": split["test"]})
