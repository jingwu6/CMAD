"""Algorithm 1's outer loop: generate -> debate -> vote -> correctness filter
-> diversity select -> align -> accumulate -> LoRA SFT -> serve new adapter
-> repeat for L iterations.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from cmad.agents import extract_answers, make_agent_fn
from cmad.alignment import make_align_fn
from cmad.answer_extract import extract_final_numeric_answer
from cmad.config import DebateConfig, LoRAConfig, SelectionConfig, SFTLoopConfig
from cmad.debate import run_debate
from cmad.embeddings import embed_candidates
from cmad.llm_client import LLMClient
from cmad.sft_dataset import accumulate, build_sft_examples, select_candidates_for_question
from cmad.strategies import generate_strategies
from cmad.vllm_server import VLLMServerManager
from cmad.voting import build_pooled_vote_input, majority_vote

# Questions are independent, so they are debated concurrently against the
# same vLLM server; rounds within one question remain sequential.
DEFAULT_CONCURRENCY = 16


def _debate_one_question(
    client: LLMClient,
    question_item,
    generator_model: str,
    debate_cfg: DebateConfig,
    dataset_cfg=None,
) -> Optional[dict]:
    """Run strategy generation + debate + voting for one question, without
    applying selection yet (so iteration 1 can be shared by the CMAD and
    CMAD-w/o-diversity variants).

    `question_item`: a question string, or a dict {"question": str,
    "subject": str} for subject-aware strategy generation on MATH.

    Returns None if the debate hits an HTTP error (e.g. a prompt exceeding
    --max-model-len); the question is then left out of this iteration.
    """
    if isinstance(question_item, dict):
        question = question_item["question"]
        subject = question_item.get("subject")
    else:
        question = question_item
        subject = None

    extractor = dataset_cfg.extractor if dataset_cfg is not None else extract_final_numeric_answer
    try:
        strategies = generate_strategies(client, question, m=debate_cfg.M, model=generator_model, subject=subject)
        agent_fn = make_agent_fn(client, model=debate_cfg.model, temperature=debate_cfg.temperature, dataset_cfg=dataset_cfg)
        rounds = run_debate(question, strategies, M=debate_cfg.M, N=debate_cfg.N, agent_fn=agent_fn)
    except Exception as e:
        print(f"[cmad] debate failed for question {question[:40]!r} (generator={generator_model}), skipping: {e}")
        return None

    final_round = rounds[-1]
    final_answers_extracted = extract_answers(final_round, extractor=extractor)
    # Final-round answers drive the per-agent correctness check.
    vote_input = [a if a is not None else f"__none__{i}" for i, a in enumerate(final_answers_extracted)]

    # The pseudo-label itself is voted over all N rounds' answers (M*N votes).
    all_rounds_extracted = [extract_answers(r, extractor=extractor) for r in rounds[:-1]] + [final_answers_extracted]
    pooled_vote_input = build_pooled_vote_input(all_rounds_extracted)
    pseudo_label, _ = majority_vote(pooled_vote_input)

    embeddings = embed_candidates(strategies, final_round)
    return {
        "question": question,
        "strategies": strategies,
        "final_round": final_round,
        "vote_input": vote_input,
        "pseudo_label": pseudo_label,
        "embeddings": embeddings,
    }


def run_debate_and_vote(
    client: LLMClient,
    questions,
    generator_model: str,
    debate_cfg: DebateConfig,
    concurrency: int = DEFAULT_CONCURRENCY,
    dataset_cfg=None,
) -> List[dict]:
    """Run _debate_one_question over all `questions` concurrently. Returns
    the unfiltered per-question candidate dicts; failed questions are
    dropped."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_debate_one_question, client, q, generator_model, debate_cfg, dataset_cfg)
            for q in questions
        ]
        results = [f.result() for f in futures]
    dropped = sum(1 for r in results if r is None)
    if dropped:
        print(f"[cmad] run_debate_and_vote: dropped {dropped}/{len(questions)} questions due to debate failures")
    return [r for r in results if r is not None]


def select_and_build_rows(
    raw_candidates: List[dict],
    selection_cfg: SelectionConfig,
    align_fn,
) -> List[dict]:
    """Apply correctness filter + (optional) diversity selection + (optional)
    alignment to each question's candidates and return the SFT rows."""
    rows: List[dict] = []
    for cand in raw_candidates:
        selected = select_candidates_for_question(
            strategies=cand["strategies"],
            final_answers_extracted=cand["vote_input"],
            final_answers_raw=cand["final_round"],
            pseudo_label=cand["pseudo_label"],
            embeddings=cand["embeddings"],
            tau=selection_cfg.tau,
            metric=selection_cfg.metric,
            t=selection_cfg.t,
            apply_diversity=selection_cfg.apply_diversity,
        )
        rows.extend(build_sft_examples(cand["question"], cand["strategies"], cand["final_round"], selected,
                                        align_fn=align_fn))
    return rows


def run_one_iteration(
    client: LLMClient,
    questions,
    generator_model: str,
    debate_cfg: DebateConfig,
    selection_cfg: SelectionConfig,
    use_alignment: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
    dataset_cfg=None,
) -> List[dict]:
    """Run debate + selection over `questions` with G served as
    `generator_model`, and return this iteration's SFT rows."""
    # Alignment always uses the fixed base model, not the current G.
    align_fn = make_align_fn(client, model=debate_cfg.model) if use_alignment else None
    raw_candidates = run_debate_and_vote(client, questions, generator_model, debate_cfg, concurrency, dataset_cfg)
    iteration_rows = select_and_build_rows(raw_candidates, selection_cfg, align_fn)
    print(f"[cmad] sft_dataset size={len(iteration_rows)} (this iteration, over {len(questions)} questions)")
    return iteration_rows


def run_sft_loop(
    client: LLMClient,
    train_questions,
    base_model_path: str,
    debate_cfg: DebateConfig,
    selection_cfg: SelectionConfig,
    lora_cfg: LoRAConfig,
    loop_cfg: SFTLoopConfig,
    adapter_dir: str,
    variant_name: str,
    server_manager: VLLMServerManager,
    use_alignment: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
    precomputed_iter1_raw_candidates: Optional[List[dict]] = None,
    dataset_cfg=None,
) -> List[str]:
    """Run L SFT iterations and return the registered adapter names (one per
    iteration, in order), usable as `model=` values at eval time.

    Each new adapter is served by restarting vLLM with an updated
    --lora-modules list (see vllm_server.py).

    `precomputed_iter1_raw_candidates`: if given, iteration 1 reuses these
    debate results and only applies this variant's selection.
    """
    # Imported here so the rest of the pipeline (and its tests) does not
    # require torch/peft.
    from cmad.lora_sft import fit_lora_adapter

    dataset: List[dict] = []
    adapter_names: List[str] = []
    generator_model = base_model_path

    for l in range(1, loop_cfg.L + 1):
        print(f"[cmad] sft_loop variant={variant_name} iteration={l}/{loop_cfg.L} generator_model={generator_model}")
        if l == 1 and precomputed_iter1_raw_candidates is not None:
            align_fn = make_align_fn(client, model=debate_cfg.model) if use_alignment else None
            new_rows = select_and_build_rows(precomputed_iter1_raw_candidates, selection_cfg, align_fn)
            print(f"[cmad] sft_dataset size={len(new_rows)} (iteration 1, shared debate)")
        else:
            new_rows = run_one_iteration(
                client, train_questions, generator_model, debate_cfg, selection_cfg, use_alignment, concurrency, dataset_cfg,
            )
        dataset = accumulate(dataset, new_rows)

        iter_dir = os.path.abspath(os.path.join(adapter_dir, f"{variant_name}_iter{l}"))
        fit_lora_adapter(base_model_path, dataset, iter_dir, config=lora_cfg)

        # Save the accumulated training rows next to the adapter.
        dataset_path = os.path.join(iter_dir, "sft_dataset.jsonl")
        os.makedirs(iter_dir, exist_ok=True)
        with open(dataset_path, "w") as f:
            for row in dataset:
                f.write(json.dumps(row) + "\n")
        print(f"[cmad] wrote {len(dataset)} accumulated SFT rows to {dataset_path}")

        adapter_name = f"{variant_name}_iter{l}"
        server_manager.restart_with_adapter(adapter_name, iter_dir)
        adapter_names.append(adapter_name)
        generator_model = adapter_name  # next iteration's G is this iteration's adapter

    return adapter_names


def run_two_variant_sft_loops(
    client: LLMClient,
    train_questions,
    base_model_path: str,
    debate_cfg: DebateConfig,
    selection_cfg_with_diversity: SelectionConfig,
    selection_cfg_no_diversity: SelectionConfig,
    lora_cfg: LoRAConfig,
    loop_cfg: SFTLoopConfig,
    adapter_dir: str,
    server_manager: VLLMServerManager,
    use_alignment: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
    dataset_cfg=None,
):
    """Run the CMAD (with diversity) and CMAD-w/o-diversity SFT loops,
    sharing iteration 1's debate (same generator and questions; only
    selection differs). Returns (cmad_adapter_names, no_diversity_adapter_names)."""
    print("[cmad] running shared iteration-1 debate for cmad and cmad_no_diversity")
    shared_iter1_raw = run_debate_and_vote(client, train_questions, base_model_path, debate_cfg, concurrency, dataset_cfg)

    cmad_adapters = run_sft_loop(
        client, train_questions, base_model_path, debate_cfg, selection_cfg_with_diversity,
        lora_cfg, loop_cfg, adapter_dir, "cmad", server_manager, use_alignment, concurrency,
        precomputed_iter1_raw_candidates=shared_iter1_raw, dataset_cfg=dataset_cfg,
    )
    no_diversity_adapters = run_sft_loop(
        client, train_questions, base_model_path, debate_cfg, selection_cfg_no_diversity,
        lora_cfg, loop_cfg, adapter_dir, "cmad_no_diversity", server_manager, use_alignment, concurrency,
        precomputed_iter1_raw_candidates=shared_iter1_raw, dataset_cfg=dataset_cfg,
    )
    return cmad_adapters, no_diversity_adapters
