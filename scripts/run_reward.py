"""
Phase 2: Reward Model Training

Entry point for reward model training. Loads UltraFeedback preference pairs,
trains DistilBERT with Bradley-Terry loss, saves best checkpoint.

Usage:
    python scripts/run_reward.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from omegaconf import OmegaConf

from src.data.reward_dataset import load_reward_dataset
from src.models.reward_model import load_reward_model_and_tokenizer
from src.training.reward_trainer import save_reward_checkpoint, train_reward_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.load("configs/reward_config.yaml")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model, tokenizer = load_reward_model_and_tokenizer(cfg)
    dataset = load_reward_dataset(cfg, tokenizer)

    trained_model = train_reward_model(
        cfg=cfg,
        model=model,
        train_dataset=dataset["train"],
        val_dataset=dataset["validation"],
    )

    save_reward_checkpoint(trained_model, cfg.training.output_dir)
    wandb.finish()
    logger.info("Phase 2 complete.")


if __name__ == "__main__":
    main()
