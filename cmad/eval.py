"""Evaluation entrypoints for every condition: cot, vanilla_mad, step_back,
self_reflection, self_contrast, dmad, cmad (per-iteration), cmad_no_diversity
(per-iteration)."""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from cmad.agents import extract_answers, make_agent_fn
from cmad.answer_extract import compute_score, extract_final_numeric_answer
from cmad.debate import run_debate
from cmad.llm_client import LLMClient
from cmad.prompts import (
    COT_PROMPT,
    PROGRAM_OF_THOUGHT_PROMPT,
    SELF_CONTRAST_PROMPT,
    SELF_REFLECTION_PROMPT,
    STEP_BACK_PROMPT,
)
from cmad.strategies import generate_strategies, get_dmad_strategies
from cmad.voting import build_pooled_vote_input, majority_vote

DEFAULT_EVAL_CONCURRENCY = 16


def _single_prompt_condition(client: LLMClient, question: str, model: str, template: str, max_tokens: int = 1024) -> str:
    prompt = template.format(question=question)
    return client.generate(prompt, model=model, temperature=0.7, max_tokens=max_tokens)


def _score_one_question(name, client, item, model, debate_M, debate_N, generator_model,
                         single_prompt_templates, extractor, scorer, dataset_cfg):
    q, gt = item["question"], item["answer"]
    max_tokens = dataset_cfg.max_tokens if dataset_cfg is not None else 1024

    # A request error on one question (e.g. a long debate history exceeding
    # --max-model-len) is scored as wrong instead of aborting the whole run.
    try:
        if name in single_prompt_templates:
            raw = _single_prompt_condition(client, q, model, single_prompt_templates[name], max_tokens=max_tokens)
            extracted = extract_answers([raw], extractor=extractor)[0]
        elif name in ("vanilla_mad", "dmad", "cmad"):
            if name == "vanilla_mad":
                strategies = None
            elif name == "dmad":
                strategies = get_dmad_strategies(debate_M)
            else:  # cmad
                assert generator_model is not None, "cmad condition requires generator_model"
                strategies = generate_strategies(client, q, m=debate_M, model=generator_model,
                                                  subject=item.get("subject"))

            agent_fn = make_agent_fn(client, model=model, dataset_cfg=dataset_cfg)
            rounds = run_debate(q, strategies, M=debate_M, N=debate_N, agent_fn=agent_fn)
            # The predicted answer is the majority vote over all N rounds.
            all_rounds_extracted = [extract_answers(r, extractor=extractor) for r in rounds]
            pooled_vote_input = build_pooled_vote_input(all_rounds_extracted)
            pseudo_label, _ = majority_vote(pooled_vote_input)
            extracted = pseudo_label if not pseudo_label.startswith("__none__") else None
        else:
            raise ValueError(f"unknown condition: {name}")
    except Exception as e:
        print(f"[cmad] eval condition={name} question={q[:40]!r} failed, scoring as wrong: {e}")
        return {"question": q, "ground_truth": gt, "extracted": None, "score": 0.0, "error": str(e)}

    score = scorer(extracted, gt)
    return {"question": q, "ground_truth": gt, "extracted": extracted, "score": score}


def run_condition(
    name: str,
    client: LLMClient,
    questions: List[dict],  # [{"question": str, "answer": str}, ...]
    model: str,
    debate_M: int = 3,
    debate_N: int = 2,
    generator_model: Optional[str] = None,
    concurrency: int = DEFAULT_EVAL_CONCURRENCY,
    dataset_cfg=None,
) -> dict:
    """Run one baseline/CMAD condition over a question set and return a
    results dict: {"condition": name, "n": ..., "correct": ..., "accuracy": ...,
    "per_question": [...]}.

    name in {"cot", "step_back", "self_reflection", "self_contrast",
    "vanilla_mad", "dmad", "cmad"}. For "cmad", `generator_model` must be
    supplied (the base model or a LoRA adapter name) and is used to
    generate the M free-form strategies; for "dmad" and "vanilla_mad" no
    generator is needed.

    `dataset_cfg`: a cmad.config.DatasetConfig selecting prompt wording,
    answer extractor and scorer; defaults to GSM8K. Pass
    `cmad.config.DATASET_CONFIGS["math"]()` for MATH.

    Questions are scored concurrently with `concurrency` workers.
    """
    if dataset_cfg is not None:
        single_prompt_templates = dataset_cfg.single_prompt_templates
        extractor = dataset_cfg.extractor
        scorer = dataset_cfg.scorer
    else:
        single_prompt_templates = {
            "cot": COT_PROMPT,
            "step_back": STEP_BACK_PROMPT,
            "self_reflection": SELF_REFLECTION_PROMPT,
            "self_contrast": SELF_CONTRAST_PROMPT,
            "program_of_thought": PROGRAM_OF_THOUGHT_PROMPT,
        }
        extractor = extract_final_numeric_answer
        scorer = compute_score

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _score_one_question, name, client, item, model, debate_M, debate_N,
                generator_model, single_prompt_templates, extractor, scorer, dataset_cfg,
            )
            for item in questions
        ]
        per_question = [f.result() for f in futures]  # input order

    n = len(per_question)
    correct = sum(pq["score"] for pq in per_question)
    accuracy = correct / n if n > 0 else 0.0
    print(f"[cmad] eval condition={name} n={n} correct={correct} accuracy={accuracy:.4f}")
    return {"condition": name, "n": n, "correct": correct, "accuracy": accuracy, "per_question": per_question}
