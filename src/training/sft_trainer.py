"""
Supervised fine-tuning trainer configuration.

Wraps TRL's SFTTrainer with the config-driven hyperparameters.
All training arguments are sourced from the YAML config — nothing is
hardcoded here.
"""

import logging
import os

from omegaconf import DictConfig
from peft import PeftModel
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def build_sft_trainer(
    cfg: DictConfig,
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset,
) -> SFTTrainer:
    """
    Construct a TRL SFTTrainer from config.

    Args:
        cfg: Full config with training, wandb, and data fields.
        model: PEFT-wrapped model with LoRA adapters.
        tokenizer: Tokenizer with padding configured.
        train_dataset: Tokenized HuggingFace Dataset.

    Returns:
        Configured SFTTrainer, ready to call .train() on.
    """
    os.makedirs(cfg.training.output_dir, exist_ok=True)

    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        seed=cfg.training.seed,
        report_to=cfg.training.report_to,
        run_name=cfg.wandb.run_name,
        max_seq_length=cfg.data.max_length,
        dataset_text_field="text",
        packing=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info(
        "SFTTrainer configured — steps per epoch: ~%d | effective batch: %d",
        len(train_dataset) // (
            cfg.training.per_device_train_batch_size
            * cfg.training.gradient_accumulation_steps
        ),
        cfg.training.per_device_train_batch_size
        * cfg.training.gradient_accumulation_steps,
    )

    return trainer


def save_sft_checkpoint(
    trainer: SFTTrainer,
    output_dir: str,
) -> None:
    """
    Save the final LoRA adapter weights and tokenizer.

    Saves only the LoRA delta weights, not the full base model.
    The base model weights are downloaded separately when needed.

    Args:
        trainer: Trained SFTTrainer instance.
        output_dir: Directory to save adapter and tokenizer.
    """
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.model.save_pretrained(final_dir)
    trainer.tokenizer.save_pretrained(final_dir)
    logger.info("SFT checkpoint saved to: %s", final_dir)
