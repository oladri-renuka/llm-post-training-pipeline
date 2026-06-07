"""
Phase 4: Stratified A/B Evaluation

Entry point for evaluation. Builds the stratified held-out set,
generates responses from both models, runs statistical tests on
verifiable tasks, and calls GPT-4o-mini for open-ended judging.

Usage:
    OPENAI_API_KEY=<key> python scripts/run_eval.py
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import wandb
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.eval_dataset import build_eval_dataset
from src.evaluation.ab_test import run_ab_evaluation
from src.evaluation.metrics import format_results_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Required for GPT-4o-mini judging in Stratum 2."
        )

    cfg = OmegaConf.load("configs/eval_config.yaml")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    torch_dtype = getattr(torch, cfg.models.torch_dtype)

    logger.info("Loading baseline model: %s", cfg.models.baseline)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        cfg.models.baseline,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained(cfg.models.baseline, use_fast=True)
    if baseline_tokenizer.pad_token is None:
        baseline_tokenizer.pad_token = baseline_tokenizer.eos_token

    logger.info("Loading treatment model: %s", cfg.models.treatment)
    treatment_model = AutoModelForCausalLM.from_pretrained(
        cfg.models.treatment,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    treatment_tokenizer = AutoTokenizer.from_pretrained(cfg.models.treatment, use_fast=True)
    if treatment_tokenizer.pad_token is None:
        treatment_tokenizer.pad_token = treatment_tokenizer.eos_token

    verifiable_examples, open_ended_examples = build_eval_dataset(cfg)

    stratum_results, open_ended_result = run_ab_evaluation(
        cfg=cfg,
        baseline_model=baseline_model,
        baseline_tokenizer=baseline_tokenizer,
        treatment_model=treatment_model,
        treatment_tokenizer=treatment_tokenizer,
        verifiable_examples=verifiable_examples,
        open_ended_examples=open_ended_examples,
    )

    summary = format_results_table(stratum_results, open_ended_result)
    logger.info(summary)

    results_txt_path = os.path.join(cfg.output_dir, "eval_summary.txt")
    with open(results_txt_path, "w") as f:
        f.write(summary)
    logger.info("Summary written to: %s", results_txt_path)

    wandb.finish()
    logger.info("Phase 4 complete.")


if __name__ == "__main__":
    main()
