"""Majority-vote pseudo-labeling over a set of agents' final-round answers.

The paper doesn't specify a tie-break rule (M agents can produce fewer than M
distinct answers with no strict majority, e.g. all M distinct, or an even
split). We pick a deterministic rule -- lowest agent index among the
tied-for-first group -- and test it explicitly rather than leaving vote
resolution to whatever a Counter/dict iteration order happens to produce.
"""

from collections import Counter
from typing import List, Optional, Tuple


def majority_vote(answers: List[str]) -> Tuple[str, dict]:
    """Return (pseudo_label, vote_counts).

    On an exact tie for the most votes, the winner is the answer belonging
    to the lowest-indexed agent among the tied answers (not e.g. the first
    key insertion order of a Counter, which is an implementation detail we
    don't want to depend on).
    """
    if not answers:
        raise ValueError("majority_vote requires at least one answer")

    counts = Counter(answers)
    max_votes = max(counts.values())
    tied = [a for a in counts if counts[a] == max_votes]

    if len(tied) == 1:
        pseudo_label = tied[0]
    else:
        # Lowest agent index among the tied answers.
        first_index_for_answer = {}
        for i, a in enumerate(answers):
            if a in tied and a not in first_index_for_answer:
                first_index_for_answer[a] = i
        pseudo_label = min(tied, key=lambda a: first_index_for_answer[a])

    return pseudo_label, dict(counts)


def build_pooled_vote_input(all_rounds_extracted: List[List[Optional[str]]]) -> List[str]:
    """Flatten every round's extracted answers (not just the final round)
    into one vote list for majority_vote, replacing None entries with
    per-position unique placeholder tokens so they never spuriously "win" a
    vote by accidentally matching each other.

    Pooling all N*M answers gives the vote more samples at no extra LLM
    cost, since every round's answers are already computed.

    `all_rounds_extracted`: List of N rounds, each a list of M
    already-extracted answers (i.e. debate.run_debate's rounds output,
    each round already passed through agents.extract_answers).
    """
    pooled: List[str] = []
    for round_idx, round_answers in enumerate(all_rounds_extracted):
        for agent_idx, a in enumerate(round_answers):
            pooled.append(a if a is not None else f"__none__r{round_idx}a{agent_idx}")
    return pooled
