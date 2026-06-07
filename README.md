# LLM Post-Training Pipeline

A complete post-training pipeline for large language models demonstrating three training paradigms — supervised fine-tuning, reward modeling, and reinforcement learning from human feedback — evaluated with a statistically rigorous stratified A/B test.

## Pipeline Overview

```
Base Model (LLaMA-3.2-1B-Instruct)
        │
        ▼
Phase 1: SFT (LoRA, 26K Alpaca examples)
        │  3.4M trainable / 752M total params (220x reduction)
        ▼
Phase 2: Reward Model (DistilBERT + Bradley-Terry loss)
        │  Trained on 5K UltraFeedback preference pairs
        ▼
Phase 3: PPO (300 steps, adaptive KL penalty)
        │  Policy ← reward signal from Phase 2
        ▼
Phase 4: Stratified A/B Evaluation (600 held-out examples)
         Stratum 1: Verifiable (two-proportion z-test, p < 0.05)
         Stratum 2: Open-ended (GPT-4o-mini judge, directional only)
```

### Design decisions

**Distribution alignment.** SFT, reward model, and PPO all operate on instruction-following data from the same distribution (Alpaca/UltraFeedback). This ensures the reward signal is coherent with the fine-tuning objective — a common failure mode in RLHF pipelines is training the reward model on a different distribution than the policy.

**LoRA sequencing.** LoRA adapters are injected before any device placement (`device_map=None` at load time). Applying LoRA after `device_map="auto"` distributes layers across devices and breaks gradient flow through the adapter parameters.

**Stratified evaluation.** Verifiable tasks (format-constrained, factual, code) use binary correctness with a two-proportion z-test. Open-ended tasks use LLM-as-judge but are explicitly marked directional-only and not used for significance claims, to avoid conflating subjective preference with measurable improvement.

## Environment

Validated on RunPod PyTorch 2.4.0 template:

| Package | Version |
|---|---|
| torch | 2.12.0+cu130 |
| transformers | 4.44.2 |
| peft | 0.12.0 |
| trl | 0.10.1 |
| accelerate | 0.34.2 |

Minimum GPU: 24GB VRAM (RTX 3090, RTX 4090, A5000, A40).

## Setup

```bash
git clone https://github.com/oladri-renuka/llm-post-training-pipeline
cd llm-post-training-pipeline
pip install -r requirements.txt
wandb login
```

For Phase 4, set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Usage

Run phases individually:

```bash
make sft       # Phase 1: Supervised fine-tuning
make reward    # Phase 2: Reward model training
make ppo       # Phase 3: PPO fine-tuning
make eval      # Phase 4: Stratified A/B evaluation
```

Or run the full pipeline end-to-end:

```bash
make all
```

## Project Structure

```
├── configs/
│   ├── sft_config.yaml        # SFT hyperparameters and LoRA config
│   ├── reward_config.yaml     # Reward model training config
│   ├── ppo_config.yaml        # PPO hyperparameters
│   └── eval_config.yaml       # Evaluation strata and statistical config
├── src/
│   ├── data/
│   │   ├── sft_dataset.py     # Alpaca loader and Alpaca prompt formatter
│   │   ├── reward_dataset.py  # UltraFeedback preference pair extractor
│   │   └── eval_dataset.py    # Stratified eval set builder and correctness checks
│   ├── models/
│   │   ├── sft_model.py       # LLaMA load + LoRA injection
│   │   ├── reward_model.py    # DistilBERT reward head + Bradley-Terry loss
│   │   └── ppo_model.py       # PPO policy/reference model setup
│   ├── training/
│   │   ├── sft_trainer.py     # TRL SFTTrainer configuration
│   │   ├── reward_trainer.py  # Custom PyTorch training loop for reward model
│   │   └── ppo_trainer.py     # TRL PPOTrainer configuration and rollout loop
│   └── evaluation/
│       ├── ab_test.py         # A/B evaluation runner with LLM judge
│       └── metrics.py         # Two-proportion z-test and result formatting
├── scripts/
│   ├── run_sft.py             # Phase 1 entry point
│   ├── run_reward.py          # Phase 2 entry point
│   ├── run_ppo.py             # Phase 3 entry point
│   └── run_eval.py            # Phase 4 entry point
├── Makefile
└── requirements.txt
```

## Experiment Tracking

All runs are tracked in Weights & Biases under the `llm-post-training-pipeline` project. Each phase logs its own run with phase-specific metrics:

- **SFT**: training loss, learning rate schedule
- **Reward model**: Bradley-Terry loss, pairwise accuracy (train + val)
- **PPO**: mean reward, KL divergence, KL coefficient
- **Evaluation**: per-stratum accuracy, delta, p-value, open-ended win rates

## Evaluation Methodology

### Stratum 1: Verifiable tasks (primary claim)

| Subcategory | n | Correctness criterion |
|---|---|---|
| Format-constrained | 150 | Structural check (JSON parseable, bullet presence, numbered list) |
| Factual | 100 | Exact/substring match against reference |
| Code | 50 | AST parse of extracted code block |

Statistical test: two-proportion z-test, α = 0.05, two-tailed.

### Stratum 2: Open-ended tasks (directional only)

300 examples judged by GPT-4o-mini. Reported as win/loss/tie rates only. Known biases are surfaced explicitly in the evaluation report and these results are not used for significance claims.
