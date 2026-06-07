"""
PPO training loop using TRL's PPOTrainer.

Runs 300 steps of Proximal Policy Optimization. At each step:
  1. Sample a batch of prompts from the Alpaca dataset.
  2. Generate responses with the current policy.
  3. Score responses with the frozen reward model.
  4. Run PPO update with KL penalty against the SFT reference policy.

The KL penalty is adaptive — init_kl_coef is adjusted during training
to keep KL divergence near target_kl.
"""

import logging
import os
from typing import List

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from trl import PPOConfig, PPOTrainer

from src.models.reward_model import RewardModel

logger = logging.getLogger(__name__)


def build_ppo_config(cfg: DictConfig) -> PPOConfig:
    """Construct TRL PPOConfig from the YAML config."""
    return PPOConfig(
        
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        mini_batch_size=cfg.ppo.mini_batch_size,
        gradient_accumulation_steps=cfg.ppo.gradient_accumulation_steps,
        adap_kl_ctrl=cfg.ppo.adap_kl_ctrl,
        init_kl_coef=cfg.ppo.init_kl_coef,
        kl_penalty=cfg.ppo.kl_penalty,
        target=cfg.ppo.target_kl,
        gamma=cfg.ppo.gamma,
        lam=cfg.ppo.lam,
        cliprange=cfg.ppo.cliprange,
        cliprange_value=cfg.ppo.cliprange_value,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        seed=cfg.ppo.seed,
        log_with=cfg.ppo.log_with,
        tracker_project_name=cfg.wandb.project,
        tracker_kwargs={
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": cfg.wandb.run_name,
                "tags": list(cfg.wandb.tags),
            }
        },
    )


@torch.no_grad()
def score_responses(
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    responses: List[str],
    device: torch.device,
    max_length: int,
) -> List[torch.Tensor]:
    """
    Score a list of response strings with the reward model.

    Args:
        reward_model: Frozen RewardModel in eval mode.
        tokenizer: Reward model tokenizer (DistilBERT).
        responses: List of decoded response strings.
        device: Device for reward model inference.
        max_length: Max token length for reward model.

    Returns:
        List of scalar reward tensors, one per response.
    """
    reward_model.eval()
    encodings = tokenizer(
        responses,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    rewards = reward_model(
        encodings["input_ids"],
        encodings["attention_mask"],
    )
    return [r.detach() for r in rewards]


def run_ppo_training(
    cfg: DictConfig,
    ppo_trainer: PPOTrainer,
    reward_model: RewardModel,
    reward_tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
) -> None:
    """
    Execute 300 PPO steps.

    Each step samples a batch of prompts, generates responses,
    scores them with the reward model, and runs a PPO update.

    Args:
        cfg: Full config.
        ppo_trainer: Initialized TRL PPOTrainer.
        reward_model: Frozen reward model on GPU.
        reward_tokenizer: DistilBERT tokenizer for scoring.
        prompts: List of prompt strings sampled from Alpaca.
    """
    device = next(reward_model.parameters()).device
    reward_model.eval()

    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": ppo_trainer.tokenizer.pad_token_id,
        "max_new_tokens": 128,
    }

    n_prompts = len(prompts)
    batch_size = cfg.ppo.batch_size

    logger.info("Starting PPO training — %d steps", cfg.ppo.num_steps)

    for step in range(cfg.ppo.num_steps):
        start_idx = (step * batch_size) % n_prompts
        end_idx = start_idx + batch_size
        if end_idx <= n_prompts:
            batch_prompts = prompts[start_idx:end_idx]
        else:
            # wrap around to always get exactly batch_size prompts
            batch_prompts = (prompts[start_idx:] + prompts[:end_idx - n_prompts])[:batch_size]

        query_tensors = [
            ppo_trainer.tokenizer.encode(p, return_tensors="pt").squeeze(0)
            for p in batch_prompts
        ]

        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs,
        )

        response_strings = [
            ppo_trainer.tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]

        rewards = score_responses(
            reward_model=reward_model,
            tokenizer=reward_tokenizer,
            responses=response_strings,
            device=device,
            max_length=cfg.reward.max_length,
        )

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        if step % 10 == 0:
            mean_reward = torch.stack(rewards).mean().item()
            logger.info(
                "PPO step %d/%d | mean_reward: %.4f | kl: %.4f",
                step,
                cfg.ppo.num_steps,
                mean_reward,
                stats.get("objective/kl", float("nan")),
            )

    os.makedirs(cfg.ppo.output_dir, exist_ok=True)
    final_dir = os.path.join(cfg.ppo.output_dir, "final")
    ppo_trainer.save_pretrained(final_dir)
    logger.info("PPO checkpoint saved to: %s", final_dir)
