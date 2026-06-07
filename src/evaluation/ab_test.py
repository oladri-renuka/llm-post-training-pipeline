"""
Stratified A/B evaluation runner.

Generates responses from both models for all 600 held-out examples,
applies stratum-appropriate correctness checks, runs statistical tests
on the verifiable stratum, and calls GPT-4o-mini for open-ended judging.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import torch
import wandb
from omegaconf import DictConfig
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.eval_dataset import EvalExample, Stratum, check_correctness
from src.evaluation.metrics import (
    OpenEndedResult,
    StratumResult,
    compute_open_ended_result,
    compute_stratum_result,
    format_results_table,
)

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of two AI assistant responses "
    "to the same instruction. Your task is to determine which response better follows "
    "the instruction, is more accurate, and is more helpful.\n\n"
    "Respond with exactly one word: 'A', 'B', or 'tie'.\n"
    "A = Response A is better. B = Response B is better. tie = Both are equally good."
)

_JUDGE_USER_TEMPLATE = (
    "Instruction: {instruction}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better? Respond with 'A', 'B', or 'tie'."
)


@torch.no_grad()
def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> str:
    """Generate a single response from a model given a prompt string."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _judge_open_ended(
    client: OpenAI,
    judge_model: str,
    instruction: str,
    response_a: str,
    response_b: str,
) -> str:
    """Call GPT-4o-mini to judge between two open-ended responses."""
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _JUDGE_USER_TEMPLATE.format(
                        instruction=instruction,
                        response_a=response_a,
                        response_b=response_b,
                    ),
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = completion.choices[0].message.content.strip().upper()
        return verdict if verdict in {"A", "B", "TIE"} else "TIE"
    except Exception as exc:
        logger.warning("Judge call failed: %s — defaulting to TIE", exc)
        return "TIE"


def run_ab_evaluation(
    cfg: DictConfig,
    baseline_model: AutoModelForCausalLM,
    baseline_tokenizer: AutoTokenizer,
    treatment_model: AutoModelForCausalLM,
    treatment_tokenizer: AutoTokenizer,
    verifiable_examples: List[EvalExample],
    open_ended_examples: List[EvalExample],
) -> Tuple[List[StratumResult], OpenEndedResult]:
    """
    Run the full stratified A/B evaluation.

    Args:
        cfg: Eval config.
        baseline_model / baseline_tokenizer: Base LLaMA model.
        treatment_model / treatment_tokenizer: PPO fine-tuned model.
        verifiable_examples: 300 examples for significance testing.
        open_ended_examples: 300 examples for directional judging.

    Returns:
        (stratum_results, open_ended_result)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = baseline_model.to(device).eval()
    treatment_model = treatment_model.to(device).eval()

    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, "raw_results.jsonl")

    stratum_baseline: Dict[Stratum, List[bool]] = {
        s: [] for s in [Stratum.FORMAT_CONSTRAINED, Stratum.FACTUAL, Stratum.CODE]
    }
    stratum_treatment: Dict[Stratum, List[bool]] = {
        s: [] for s in [Stratum.FORMAT_CONSTRAINED, Stratum.FACTUAL, Stratum.CODE]
    }

    logger.info("Evaluating %d verifiable examples...", len(verifiable_examples))
    with open(results_path, "w") as f:
        for i, ex in enumerate(verifiable_examples):
            prompt = (
                f"Below is an instruction. Write a response that completes it.\n\n"
                f"### Instruction:\n{ex.instruction}"
                + (f"\n\nInput:\n{ex.input}" if ex.input else "")
                + "\n\n### Response:\n"
            )

            base_response = generate_response(
                baseline_model, baseline_tokenizer, prompt, device
            )
            treat_response = generate_response(
                treatment_model, treatment_tokenizer, prompt, device
            )

            base_correct = check_correctness(base_response, ex)
            treat_correct = check_correctness(treat_response, ex)

            stratum_baseline[ex.stratum].append(base_correct)
            stratum_treatment[ex.stratum].append(treat_correct)

            record = {
                "example_id": ex.example_id,
                "stratum": ex.stratum.value,
                "instruction": ex.instruction,
                "baseline_response": base_response,
                "treatment_response": treat_response,
                "baseline_correct": base_correct,
                "treatment_correct": treat_correct,
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("Verifiable progress: %d/%d", i + 1, len(verifiable_examples))

    stratum_results = []
    for stratum in [Stratum.FORMAT_CONSTRAINED, Stratum.FACTUAL, Stratum.CODE]:
        result = compute_stratum_result(
            stratum_name=stratum.value,
            baseline_correct=stratum_baseline[stratum],
            treatment_correct=stratum_treatment[stratum],
            alpha=cfg.statistics.alpha,
        )
        stratum_results.append(result)
        wandb.log({
            f"eval/{stratum.value}/accuracy_baseline": result.accuracy_baseline,
            f"eval/{stratum.value}/accuracy_treatment": result.accuracy_treatment,
            f"eval/{stratum.value}/delta": result.delta,
            f"eval/{stratum.value}/p_value": result.p_value,
            f"eval/{stratum.value}/significant": int(result.significant),
        })

    openai_client = OpenAI()
    judge_model = cfg.strata.open_ended.judge_model
    baseline_wins, treatment_wins, ties = 0, 0, 0

    logger.info("Judging %d open-ended examples with %s...", len(open_ended_examples), judge_model)
    with open(results_path, "a") as f:
        for i, ex in enumerate(open_ended_examples):
            prompt = (
                f"Below is an instruction. Write a response that completes it.\n\n"
                f"### Instruction:\n{ex.instruction}"
                + (f"\n\nInput:\n{ex.input}" if ex.input else "")
                + "\n\n### Response:\n"
            )

            base_response = generate_response(
                baseline_model, baseline_tokenizer, prompt, device
            )
            treat_response = generate_response(
                treatment_model, treatment_tokenizer, prompt, device
            )

            verdict = _judge_open_ended(
                openai_client, judge_model, ex.instruction, base_response, treat_response
            )

            if verdict == "A":
                baseline_wins += 1
            elif verdict == "B":
                treatment_wins += 1
            else:
                ties += 1

            record = {
                "example_id": ex.example_id,
                "stratum": "open_ended",
                "instruction": ex.instruction,
                "baseline_response": base_response,
                "treatment_response": treat_response,
                "judge_verdict": verdict,
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                logger.info("Open-ended progress: %d/%d", i + 1, len(open_ended_examples))

    open_ended_result = compute_open_ended_result(
        baseline_wins=baseline_wins,
        treatment_wins=treatment_wins,
        ties=ties,
        bias_caveats=list(cfg.strata.open_ended.bias_caveats),
    )

    wandb.log({
        "eval/open_ended/treatment_win_rate": open_ended_result.treatment_win_rate,
        "eval/open_ended/baseline_win_rate": open_ended_result.baseline_win_rate,
        "eval/open_ended/tie_rate": open_ended_result.tie_rate,
    })

    logger.info("Raw results written to: %s", results_path)
    return stratum_results, open_ended_result
