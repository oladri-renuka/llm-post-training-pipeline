"""
Phase 3: PPO Fine-Tuning

Entry point for PPO. Loads SFT checkpoint and reward model,
runs 300 steps of PPO with adaptive KL penalty, saves final policy.

Usage:
    python scripts/run_ppo.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from trl import PPOTrainer

from src.data.sft_dataset import get_alpaca_prompt_only
from src.models.ppo_model import load_ppo_models
from src.models.reward_model import load_reward_model_from_checkpoint
from src.training.ppo_trainer import build_ppo_config, run_ppo_training
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = OmegaConf.load("configs/ppo_config.yaml")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    policy_model, ref_model, policy_tokenizer = load_ppo_models(cfg)

    reward_model = load_reward_model_from_checkpoint(
        checkpoint_path=cfg.reward.checkpoint,
        backbone_name="meta-llama/Llama-3.2-1B-Instruct",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", use_fast=True
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = reward_model.to(device)

    logger.info("Loading PPO prompts from: %s", cfg.data.prompt_dataset)
    raw = load_dataset(cfg.data.prompt_dataset, split="train")
    raw = raw.select(range(min(cfg.data.num_prompts, len(raw))))
    prompts = [get_alpaca_prompt_only(ex) for ex in raw]
    logger.info("Loaded %d prompts for PPO rollout", len(prompts))

    ppo_config = build_ppo_config(cfg)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=policy_tokenizer,
    )

    run_ppo_training(
        cfg=cfg,
        ppo_trainer=ppo_trainer,
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        prompts=prompts,
    )

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
