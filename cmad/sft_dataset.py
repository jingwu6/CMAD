"""Build the strategy-generator fine-tuning dataset from filtered debate candidates.

Combines correctness_filter + greedy_diversity_select (see selection.py)
with an optional alignment/refinement step, and accumulates examples across
SFT-loop iterations (Algorithm 1's D_f).
"""

from typing import Callable, List, Optional, Sequence

from cmad.selection import correctness_filter, greedy_diversity_select


def select_candidates_for_question(
    strategies: Sequence[str],
    final_answers_extracted: Sequence[str],
    final_answers_raw: Sequence[str],
    pseudo_label: str,
    embeddings,
    tau: float,
    apply_diversity: bool = True,
    metric: str = "cosine",
    t: float = 1.0,
) -> List[int]:
    """Return the indices (into strategies/final_answers) selected for SFT,
    for one question.

    `apply_diversity=False` gives the "CMAD w/o diversity selection"
    ablation: correctness filtering only.
    """
    correct_indices = correctness_filter(final_answers_extracted, pseudo_label)
    if not correct_indices:
        return []
    if not apply_diversity:
        return correct_indices
    # correct_indices is in agent-index order, the order greedy selection relies on.
    return greedy_diversity_select(correct_indices, embeddings, tau=tau, metric=metric, t=t)


def build_sft_examples(
    question: str,
    strategies: Sequence[str],
    final_answers_raw: Sequence[str],
    selected_indices: Sequence[int],
    align_fn: Optional[Callable[[str, str, str], str]] = None,
) -> List[dict]:
    """Turn the selected candidates for one question into ONE SFT training
    row (or zero, if nothing survived selection).

    `align_fn(question, strategy, final_answer) -> aligned_strategy` is the
    optional strategy-refinement agent (Eq 8), applied only to selected
    candidates.

    The row's `target_strategies` is the full list of kept strategies for
    this question, matching G's generation format (M strategies in a single
    numbered-list completion).
    """
    if not selected_indices:
        return []

    target_strategies = []
    for i in selected_indices:
        strategy = strategies[i]
        aligned = align_fn(question, strategy, final_answers_raw[i]) if align_fn is not None else strategy
        target_strategies.append(aligned)

    return [{"question": question, "target_strategies": target_strategies}]


def accumulate(dataset_so_far: List[dict], new_rows: List[dict]) -> List[dict]:
    """Algorithm 1's D_f <- D_f union {(x, S_hat)}: keep every row from every
    iteration, no de-duplication."""
    return dataset_so_far + new_rows
