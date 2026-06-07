"""
Reward model trainer using Bradley-Terry pairwise loss.

Uses a standard PyTorch training loop rather than HuggingFace Trainer
because the contrastive loss requires simultaneous forward passes on
chosen and rejected sequences — Trainer's per-example abstraction
doesn't map cleanly onto this.
"""

import logging
import os
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

import wandb

from src.models.reward_model import RewardModel, bradley_terry_loss, compute_accuracy

logger = logging.getLogger(__name__)


class RewardModelCollator:
    """Collate chosen/rejected token pairs into batched tensors."""

    def __call__(self, batch):
        return {
            "chosen_input_ids": torch.tensor(
                [ex["chosen_input_ids"] for ex in batch], dtype=torch.long
            ),
            "chosen_attention_mask": torch.tensor(
                [ex["chosen_attention_mask"] for ex in batch], dtype=torch.long
            ),
            "rejected_input_ids": torch.tensor(
                [ex["rejected_input_ids"] for ex in batch], dtype=torch.long
            ),
            "rejected_attention_mask": torch.tensor(
                [ex["rejected_attention_mask"] for ex in batch], dtype=torch.long
            ),
        }


def train_reward_model(
    cfg: DictConfig,
    model: RewardModel,
    train_dataset,
    val_dataset,
) -> RewardModel:
    """
    Train the reward model with Bradley-Terry loss.

    Args:
        cfg: Config with training fields.
        model: RewardModel instance (on CPU, will be moved to device).
        train_dataset: Tokenized training split.
        val_dataset: Tokenized validation split.

    Returns:
        Best reward model (by validation accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model = model.to(device)

    collator = RewardModelCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.per_device_train_batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = len(train_loader) * cfg.training.num_train_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = None  # bfloat16 does not need GradScaler

    best_val_accuracy = 0.0
    best_state_dict = None
    global_step = 0

    os.makedirs(cfg.training.output_dir, exist_ok=True)

    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                chosen_rewards = model(chosen_ids, chosen_mask)
                rejected_rewards = model(rejected_ids, rejected_mask)
                loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

            if global_step % cfg.training.logging_steps == 0:
                acc = compute_accuracy(chosen_rewards.detach(), rejected_rewards.detach())
                wandb.log({
                    "train/loss": loss.item(),
                    "train/accuracy": acc,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                })
                logger.info(
                    "Step %d | loss: %.4f | acc: %.4f",
                    global_step, loss.item(), acc,
                )

            if global_step % cfg.training.eval_steps == 0:
                val_metrics = _evaluate(model, val_loader, device)
                wandb.log({
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "train/global_step": global_step,
                })
                logger.info(
                    "Val step %d | loss: %.4f | acc: %.4f",
                    global_step, val_metrics["loss"], val_metrics["accuracy"],
                )

                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    logger.info("New best val accuracy: %.4f", best_val_accuracy)

                model.train()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d complete | avg loss: %.4f", epoch + 1, avg_epoch_loss)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info("Restored best model (val accuracy: %.4f)", best_val_accuracy)

    return model


@torch.no_grad()
def _evaluate(
    model: RewardModel,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation loop and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in val_loader:
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)

        loss = bradley_terry_loss(chosen_rewards, rejected_rewards)
        acc = compute_accuracy(chosen_rewards, rejected_rewards)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_acc / n_batches,
    }


def save_reward_checkpoint(model: RewardModel, output_dir: str) -> None:
    """Save reward model state dict to output_dir/reward_model.pt."""
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "reward_model.pt"))
    logger.info("Reward model saved to: %s", final_dir)
