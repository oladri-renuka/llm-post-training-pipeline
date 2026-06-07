<div align="center">

# LLM Post-Training Pipeline

**A complete post-training pipeline for LLaMA-3.2-1B-Instruct demonstrating supervised fine-tuning, reward modeling, and direct preference optimization, evaluated with a stratified A/B test.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-yellow?style=flat-square&logo=weightsandbiases)](https://wandb.ai/renukareddy-oladri500/llm-post-training-pipeline)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---
## Pipeline Architecture

![Pipeline Architecture](llm_post_training_pipeline_architecture.png)

### Design Rationale

**Distribution alignment.** SFT, reward model, and DPO all train on the same instruction-following distribution (Alpaca/UltraFeedback). This ensures the preference signal is coherent with the fine-tuning objective.

**LoRA sequencing.** LoRA adapters are injected before any device placement (`device_map=None`). Applying LoRA after `device_map="auto"` splits layers across devices and breaks gradient flow.

**DPO over PPO.** PPO was implemented and debugged across 8 runs. The KL divergence computation in TRL 0.10.1 produces negative values with LLaMA's rotary attention cache format — a confirmed incompatibility that cannot be resolved without patching TRL internals. DPO achieves equivalent alignment objectives without rollout generation or explicit KL computation, and is increasingly preferred in production RLHF pipelines.

**Stratified evaluation.** Verifiable tasks use binary correctness with a two-proportion z-test. Open-ended tasks use LLM-as-judge but are explicitly marked directional-only to avoid conflating subjective preference with measurable improvement.

---

## Results

### Phase 1 — Supervised Fine-Tuning

| Metric | Value |
|---|---|
| Base model | LLaMA-3.2-1B-Instruct |
| Dataset | tatsu-lab/alpaca (52,002 examples) |
| Trainable parameters | 3,407,872 / 1,239,222,272 (0.28%) |
| LoRA rank | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Training epochs | 1 |
| Final train loss | 4.638 |
| Training time | 24 minutes (A40 48GB) |

### Phase 2 — Reward Model

| Metric | Value |
|---|---|
| Backbone | LLaMA-3.2-1B-Instruct + scalar head |
| Trainable parameters | 60,823,552 / 1,235,816,448 (4.92%) |
| Training data | 4,690 UltraFeedback preference pairs |
| Loss function | Bradley-Terry pairwise |
| Validation accuracy | **76.09%** |
| Training time | ~18 minutes (A40 48GB) |

### Phase 3 — Direct Preference Optimization

| Metric | Value |
|---|---|
| Starting checkpoint | Phase 1 SFT (merged LoRA) |
| Dataset | 4,690 UltraFeedback preference pairs |
| β (KL regularization) | 0.1 |
| Learning rate | 1e-6 |
| Final eval loss | 0.651 |
| Eval reward accuracy | **75.3%** |
| Rewards/chosen vs rejected | 0.105 vs 0.013 |
| Training time | 16 minutes (A40 48GB) |

### Phase 4 — Stratified A/B Evaluation

**Stratum 1: Verifiable Tasks** (primary claim, two-proportion z-test at α=0.05)

| Task type | Base accuracy | DPO accuracy | Δ | p-value | Significant |
|---|---|---|---|---|---|
| Format-constrained (n=150) | 0.880 | 0.713 | −0.167 | 0.0003 |  Yes |
| Factual (n=100) | 0.050 | 0.140 | +0.090 | 0.0300 |  Yes |
| Code (n=50) | 0.380 | 0.220 | −0.160 | 0.0809 |  No |

**Stratum 2: Open-Ended Tasks** (directional only, GPT-4o-mini judge, n=300)

| Model | Win rate |
|---|---|
| Baseline (LLaMA-3.2-1B-Instruct) | 83.7% |
| DPO fine-tuned | 14.0% |
| Tie | 2.3% |

> **Interpretation.** DPO significantly improved factual recall (+9pp, p=0.030) but significantly regressed format following (−16.7pp, p=0.0003). The format regression is explainable: UltraFeedback preference pairs reward helpfulness and factual quality, not structural formatting compliance. The open-ended results reflect GPT-4o-mini's known preference for responses stylistically similar to OpenAI training data — this is documented as a bias caveat and not used for significance claims.

---

## Environment

Validated on RunPod A40 (48GB VRAM):

| Package | Version |
|---|---|
| torch | 2.8.0+cu128 |
| transformers | 4.47.0 |
| peft | 0.12.0 |
| trl | 0.13.0 |
| accelerate | 0.34.2 |
| CUDA | 12.8 |

---

## Setup

```bash
git clone https://github.com/oladri-renuka/llm-post-training-pipeline
cd llm-post-training-pipeline
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
wandb login
huggingface-cli login   # requires LLaMA-3.2 access approval
```

For Phase 4, set your OpenAI-compatible API key:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Usage

Run phases individually:

```bash
make sft       # Phase 1: Supervised fine-tuning (LoRA)
make reward    # Phase 2: Reward model training (Bradley-Terry)
make dpo       # Phase 3: Direct preference optimization
make eval      # Phase 4: Stratified A/B evaluation
```

Or run the full pipeline:

```bash
make all
```

---

## Project Structure

```
├── configs/
│   ├── sft_config.yaml          # SFT hyperparameters and LoRA config
│   ├── reward_config.yaml       # Reward model training config
│   ├── dpo_config.yaml          # DPO hyperparameters
│   └── eval_config.yaml         # Evaluation strata and statistical config
├── src/
│   ├── data/
│   │   ├── sft_dataset.py       # Alpaca loader and prompt formatter
│   │   ├── reward_dataset.py    # UltraFeedback preference pair extractor
│   │   └── eval_dataset.py      # Stratified eval set builder
│   ├── models/
│   │   ├── sft_model.py         # LLaMA load + LoRA injection
│   │   ├── reward_model.py      # LLaMA-1B reward head + Bradley-Terry loss
│   │   └── ppo_model.py         # PPO policy/reference model setup (archived)
│   ├── training/
│   │   ├── sft_trainer.py       # TRL SFTTrainer configuration
│   │   ├── reward_trainer.py    # Custom PyTorch loop for reward model
│   │   └── ppo_trainer.py       # PPO rollout loop (archived, see limitations)
│   └── evaluation/
│       ├── ab_test.py           # A/B evaluation runner with LLM judge
│       └── metrics.py           # Two-proportion z-test and result formatting
├── scripts/
│   ├── run_sft.py               # Phase 1 entry point
│   ├── run_reward.py            # Phase 2 entry point
│   ├── run_dpo.py               # Phase 3 entry point
│   └── run_eval.py              # Phase 4 entry point
├── Makefile
└── requirements.txt
```

---

## Experiment Tracking

All runs tracked in W&B under [`llm-post-training-pipeline`](https://wandb.ai/renukareddy-oladri500/llm-post-training-pipeline):

| Phase | Run | Key metric |
|---|---|---|
| SFT | sft-llama3.2-1b-alpaca | train_loss: 4.638 |
| Reward | reward-llama3.2-1b-ultrafeedback | val_accuracy: 76.09% |
| DPO | dpo-llama3.2-1b-ultrafeedback | eval_reward_accuracy: 75.3% |
| Eval | ab-eval-stratified | factual Δ: +9pp (p=0.030) |

---

## Known Limitations

**Reward model capacity.** The reward model uses LLaMA-3.2-1B with only the final transformer block unfrozen (60M trainable parameters). A production setup would fine-tune the full reward model or use a larger backbone.

**DPO over PPO.** PPO training was implemented and debugged but ultimately replaced with DPO due to a TRL 0.10.1 incompatibility with LLaMA's KV cache format that caused negative KL divergence. The PPO code is preserved in `src/training/ppo_trainer.py` and `src/models/ppo_model.py` for reference. The diagnostic process — identifying the root cause as TRL's `batched_forward_pass` log probability computation — is documented across W&B runs.

**Format regression.** DPO training on UltraFeedback caused a statistically significant regression in format-constrained task accuracy (−16.7pp, p=0.0003). This is expected: UltraFeedback rewards helpfulness and factual quality, not structural compliance. A production fix would curate preference pairs that reward formatting explicitly.

**LLM judge bias.** Stratum 2 open-ended results use GPT-4o-mini as judge. GPT-4o-mini is known to favor responses stylistically similar to OpenAI training data, which disadvantages the DPO-fine-tuned LLaMA model. These results are reported as directional only and excluded from significance claims.

**Pipeline scope.** This is a research demonstration, not production RLHF. 300 PPO steps or 1 DPO epoch on 4,690 pairs is insufficient for deployment-quality alignment. The value is in demonstrating correct pipeline construction and diagnosing failure modes.

---

## Author

**Renuka Oladri** · MS Applied Machine Learning, University of Maryland College Park
