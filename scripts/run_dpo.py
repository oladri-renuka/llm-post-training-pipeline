"""
Phase 3: Direct Preference Optimization (DPO)

Replaces PPO with DPO for preference learning. DPO reformulates RLHF
as a supervised learning problem — no reward model or rollout generation
needed. Trains directly on (prompt, chosen, rejected) triples from
UltraFeedback using the SFT model as both the policy and reference.

Usage:
    python scripts/run_dpo.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from datasets import load_dataset, DatasetDict
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_preference_pairs(example: dict) -> dict | None:
    """Extract chosen/rejected pair from UltraFeedback example."""
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
        "prompt": instruction,
        "chosen": chosen,
        "rejected": rejected,
    }


def load_dpo_dataset(cfg, tokenizer) -> DatasetDict:
    """Load and format UltraFeedback for DPO training."""
    logger.info("Loading dataset: %s", cfg.data.dataset_name)
    raw = load_dataset(cfg.data.dataset_name, split="train")

    if cfg.data.num_samples < len(raw):
        raw = raw.select(range(cfg.data.num_samples))

    pairs = []
    for ex in raw:
        pair = extract_preference_pairs(ex)
        if pair is not None:
            pairs.append(pair)

    logger.info("Valid preference pairs: %d", len(pairs))

    from datasets import Dataset
    dataset = Dataset.from_list(pairs)
    split = dataset.train_test_split(
        test_size=1 - cfg.data.train_val_split, seed=42
    )
    logger.info("Train: %d | Val: %d", len(split["train"]), len(split["test"]))
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def main() -> None:
    cfg = OmegaConf.load("configs/dpo_config.yaml")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    torch_dtype = getattr(torch, cfg.model.torch_dtype)

    logger.info("Loading SFT model from: %s", cfg.model.sft_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    peft_model = PeftModel.from_pretrained(base_model, cfg.model.sft_checkpoint)
    model = peft_model.merge_and_unload()
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset = load_dpo_dataset(cfg, tokenizer)

    os.makedirs(cfg.dpo.output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=cfg.dpo.output_dir,
        beta=cfg.dpo.beta,
        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        learning_rate=cfg.dpo.learning_rate,
        lr_scheduler_type=cfg.dpo.lr_scheduler_type,
        warmup_ratio=cfg.dpo.warmup_ratio,
        weight_decay=cfg.dpo.weight_decay,
        bf16=cfg.dpo.bf16,
        fp16=cfg.dpo.fp16,
        logging_steps=cfg.dpo.logging_steps,
        save_steps=cfg.dpo.save_steps,
        save_total_limit=cfg.dpo.save_total_limit,
        eval_strategy=cfg.dpo.eval_strategy,
        eval_steps=cfg.dpo.eval_steps,
        seed=cfg.dpo.seed,
        report_to=cfg.dpo.report_to,
        run_name=cfg.wandb.run_name,
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
        max_grad_norm=cfg.dpo.max_grad_norm,
        max_length=cfg.data.max_length,
        max_prompt_length=cfg.data.max_prompt_length,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    final_dir = os.path.join(cfg.dpo.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("DPO checkpoint saved to: %s", final_dir)

    wandb.finish()
    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
