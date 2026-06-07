"""
Stratified evaluation dataset builder.

Constructs a 600-example held-out evaluation set split into:
  - Stratum 1 (300): verifiable tasks — format-constrained, factual, code
  - Stratum 2 (300): open-ended tasks for LLM-as-judge

Stratification is deterministic given a fixed seed.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Stratum(str, Enum):
    FORMAT_CONSTRAINED = "format_constrained"
    FACTUAL = "factual"
    CODE = "code"
    OPEN_ENDED = "open_ended"


@dataclass
class EvalExample:
    example_id: str
    instruction: str
    input: str
    reference_output: str
    stratum: Stratum


_FORMAT_KEYWORDS = [
    "list", "bullet", "numbered", "json", "table", "format",
    "steps", "summarize in", "outline", "enumerate",
]

_CODE_KEYWORDS = [
    "write a function", "implement", "code", "python", "javascript",
    "program", "script", "algorithm", "debug", "fix the",
]

_FACTUAL_KEYWORDS = [
    "what is", "who is", "when did", "where is", "how many",
    "define", "explain what", "what are the",
]


def _classify_example(instruction: str) -> Stratum:
    """
    Heuristically classify an instruction into one of four strata.
    Priority: code > format_constrained > factual > open_ended.
    """
    lower = instruction.lower()

    if any(kw in lower for kw in _CODE_KEYWORDS):
        return Stratum.CODE
    if any(kw in lower for kw in _FORMAT_KEYWORDS):
        return Stratum.FORMAT_CONSTRAINED
    if any(kw in lower for kw in _FACTUAL_KEYWORDS):
        return Stratum.FACTUAL
    return Stratum.OPEN_ENDED


def _verify_format(prediction: str, reference: str, instruction: str) -> bool:
    """
    Binary correctness check for format-constrained tasks.
    Checks structural properties rather than exact match.
    """
    lower_instruction = instruction.lower()

    if "json" in lower_instruction:
        try:
            import json
            json.loads(prediction)
            return True
        except (ValueError, TypeError):
            return False

    if any(kw in lower_instruction for kw in ["bullet", "list"]):
        lines = [l.strip() for l in prediction.strip().split("\n") if l.strip()]
        has_bullets = any(
            re.match(r"^[-*•]|^\d+\.", line) for line in lines
        )
        return has_bullets and len(lines) >= 2

    if "numbered" in lower_instruction or "steps" in lower_instruction:
        lines = [l.strip() for l in prediction.strip().split("\n") if l.strip()]
        return any(re.match(r"^\d+[.)]\s", line) for line in lines)

    return len(prediction.strip()) > 0


def _verify_code(prediction: str) -> bool:
    """
    Binary correctness check for code tasks.
    Checks that the prediction contains a syntactically parseable code block.
    """
    import ast

    code_block_pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    match = code_block_pattern.search(prediction)
    code = match.group(1) if match else prediction

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def build_eval_dataset(cfg: DictConfig) -> Tuple[List[EvalExample], List[EvalExample]]:
    """
    Build stratified held-out evaluation sets.

    Returns:
        Tuple of (verifiable_examples, open_ended_examples).
    """
    rng = np.random.default_rng(cfg.data.seed)

    logger.info("Loading evaluation source: %s", cfg.data.eval_dataset)
    raw = load_dataset(cfg.data.eval_dataset, split="train")

    buckets: Dict[Stratum, List[Dict]] = {s: [] for s in Stratum}
    for ex in raw:
        stratum = _classify_example(ex["instruction"])
        buckets[stratum].append(ex)

    targets = {
        Stratum.FORMAT_CONSTRAINED: cfg.strata.verifiable.subcategories.format_constrained.n,
        Stratum.FACTUAL: cfg.strata.verifiable.subcategories.factual.n,
        Stratum.CODE: cfg.strata.verifiable.subcategories.code.n,
        Stratum.OPEN_ENDED: cfg.strata.open_ended.n,
    }

    verifiable: List[EvalExample] = []
    open_ended: List[EvalExample] = []

    for stratum, target_n in targets.items():
        pool = buckets[stratum]
        if len(pool) < target_n:
            logger.warning(
                "Stratum %s: requested %d but only %d available",
                stratum.value, target_n, len(pool),
            )
            target_n = len(pool)

        indices = rng.choice(len(pool), size=target_n, replace=False)
        selected = [pool[i] for i in indices]

        for idx, ex in enumerate(selected):
            eval_ex = EvalExample(
                example_id=f"{stratum.value}_{idx:04d}",
                instruction=ex["instruction"],
                input=ex.get("input", ""),
                reference_output=ex["output"],
                stratum=stratum,
            )
            if stratum == Stratum.OPEN_ENDED:
                open_ended.append(eval_ex)
            else:
                verifiable.append(eval_ex)

    logger.info(
        "Eval set built — verifiable: %d | open_ended: %d",
        len(verifiable), len(open_ended),
    )
    return verifiable, open_ended


def check_correctness(
    prediction: str,
    example: EvalExample,
) -> bool:
    """
    Dispatch binary correctness check by stratum.

    Args:
        prediction: Model-generated response string.
        example: EvalExample with stratum and reference metadata.

    Returns:
        True if the prediction is judged correct, False otherwise.
    """
    if example.stratum == Stratum.FORMAT_CONSTRAINED:
        return _verify_format(prediction, example.reference_output, example.instruction)
    if example.stratum == Stratum.CODE:
        return _verify_code(prediction)
    if example.stratum == Stratum.FACTUAL:
        pred_norm = prediction.strip().lower()
        ref_norm = example.reference_output.strip().lower()
        return pred_norm == ref_norm or ref_norm in pred_norm
    return False
