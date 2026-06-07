"""
Phase 1: Supervised Fine-Tuning

Entry point for SFT. Loads config, initializes W&B, runs training,
saves LoRA checkpoint.

Usage:
    python scripts/run_sft.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from omegaconf import OmegaConf

from src.data.sft_dataset import load_sft_dataset
from src.models.sft_model import apply_lora, load_base_model_and_tokenizer
from src.training.sft_trainer import build_sft_trainer, save_sft_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.load("configs/sft_config.yaml")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model, tokenizer = load_base_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    train_dataset = load_sft_dataset(cfg, tokenizer)

    trainer = build_sft_trainer(cfg, model, tokenizer, train_dataset)

    logger.info("Starting SFT training...")
    trainer.train()

    save_sft_checkpoint(trainer, cfg.training.output_dir)
    wandb.finish()
    logger.info("Phase 1 complete.")


if __name__ == "__main__":
    main()
