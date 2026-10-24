"""Config dataclasses for the CMAD pipeline."""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class DebateConfig:
    M: int = 3
    N: int = 2
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.7


@dataclass
class SelectionConfig:
    # On L2-normalized embeddings, cosine similarity is a monotonic
    # reparametrization of the RBF kernel exp(-t * ||u - v||^2), so both
    # metrics select the same candidates for a matching threshold.
    metric: str = "cosine"
    t: float = 1.0          # RBF kernel temperature, only used when metric="rbf"
    # Cosine threshold equivalent to an RBF threshold of 0.9 at t=1.0:
    # exp(2cos(theta) - 2) = 0.9  =>  cos(theta) = 1 + ln(0.9) / 2 ~= 0.9473.
    tau: float = 0.9473
    apply_diversity: bool = True


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 16
    target_modules: str = "all-linear"
    lr: float = 1e-4
    num_train_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch size 16
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01


@dataclass
class SFTLoopConfig:
    L: int = 4
    train_pool_size: int = 300
    seed: int = 0


@dataclass
class DatasetConfig:
    """Everything that differs between GSM8K and MATH: the prompt wording
    (MATH asks for \\boxed{} since its answers are often non-numeric) and
    the matching answer extractor / scorer."""

    name: str
    round1_prompt: str
    round1_no_strategy_prompt: str
    debate_round_prompt: str
    single_prompt_templates: Dict[str, str]
    extractor: Callable
    scorer: Callable
    max_tokens: int = 1024


def _build_gsm8k_dataset_config() -> DatasetConfig:
    from cmad.answer_extract import compute_score, extract_final_numeric_answer
    from cmad.prompts import (
        AGENT_DEBATE_ROUND_PROMPT,
        AGENT_ROUND1_NO_STRATEGY_PROMPT,
        AGENT_ROUND1_PROMPT,
        COT_PROMPT,
        PROGRAM_OF_THOUGHT_PROMPT,
        SELF_CONTRAST_PROMPT,
        SELF_REFLECTION_PROMPT,
        STEP_BACK_PROMPT,
    )

    return DatasetConfig(
        name="gsm8k",
        round1_prompt=AGENT_ROUND1_PROMPT,
        round1_no_strategy_prompt=AGENT_ROUND1_NO_STRATEGY_PROMPT,
        debate_round_prompt=AGENT_DEBATE_ROUND_PROMPT,
        single_prompt_templates={
            "cot": COT_PROMPT,
            "step_back": STEP_BACK_PROMPT,
            "self_reflection": SELF_REFLECTION_PROMPT,
            "self_contrast": SELF_CONTRAST_PROMPT,
            "program_of_thought": PROGRAM_OF_THOUGHT_PROMPT,
        },
        extractor=extract_final_numeric_answer,
        scorer=compute_score,
    )


def _build_math_dataset_config() -> DatasetConfig:
    from cmad.answer_extract import compute_math_score, extract_boxed_answer
    from cmad.prompts import (
        AGENT_DEBATE_ROUND_PROMPT_MATH,
        AGENT_ROUND1_NO_STRATEGY_PROMPT_MATH,
        AGENT_ROUND1_PROMPT_MATH,
        COT_PROMPT_MATH,
        PROGRAM_OF_THOUGHT_PROMPT_MATH,
        SELF_CONTRAST_PROMPT_MATH,
        SELF_REFLECTION_PROMPT_MATH,
        STEP_BACK_PROMPT_MATH,
    )

    return DatasetConfig(
        name="math",
        round1_prompt=AGENT_ROUND1_PROMPT_MATH,
        round1_no_strategy_prompt=AGENT_ROUND1_NO_STRATEGY_PROMPT_MATH,
        debate_round_prompt=AGENT_DEBATE_ROUND_PROMPT_MATH,
        single_prompt_templates={
            "cot": COT_PROMPT_MATH,
            "step_back": STEP_BACK_PROMPT_MATH,
            "self_reflection": SELF_REFLECTION_PROMPT_MATH,
            "self_contrast": SELF_CONTRAST_PROMPT_MATH,
            "program_of_thought": PROGRAM_OF_THOUGHT_PROMPT_MATH,
        },
        extractor=extract_boxed_answer,
        scorer=compute_math_score,
        # MATH solutions are longer; 1024 tokens often truncates before the final \boxed{}.
        max_tokens=2048,
    )


# Factories rather than instances, to keep cmad.config free of import-time
# dependencies on cmad.prompts / cmad.answer_extract.
DATASET_CONFIGS: Dict[str, Callable[[], DatasetConfig]] = {
    "gsm8k": _build_gsm8k_dataset_config,
    "math": _build_math_dataset_config,
}
