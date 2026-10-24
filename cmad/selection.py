"""Correctness + diversity sample selection (paper Eq 4-7).

Correctness-only filtering lets a self-trained strategy generator converge
to homogeneous ("mental set") strategies over repeated iterations. Diversity
selection counters this with a greedy pass in embedding space: a candidate
is rejected if its similarity to an already-selected example exceeds a
threshold.
"""

from typing import List, Sequence

import numpy as np


def correctness_filter(answers: Sequence[str], pseudo_label: str) -> List[int]:
    """Return indices of answers that agree with the majority-vote pseudo_label,
    in original order."""
    return [i for i, a in enumerate(answers) if a == pseudo_label]


def rbf_similarity(u: np.ndarray, v: np.ndarray, t: float) -> float:
    """Gaussian potential kernel G_t(u, v) = exp(-t * ||u - v||^2) (Eq 5)."""
    diff = u - v
    sq_dist = float(np.dot(diff, diff))
    return float(np.exp(-t * sq_dist))


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """cos(theta) between u and v.

    For unit vectors ||u - v||^2 = 2(1 - cos(theta)), so on L2-normalized
    embeddings rbf_similarity is a strictly monotonic function of
    cos(theta): both metrics select the same candidates once tau is
    rescaled accordingly. Cosine is bounded in [-1, 1] and has no
    temperature, which makes tau easier to interpret.
    """
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return 0.0
    return float(np.dot(u, v) / denom)


def greedy_diversity_select(
    indices: Sequence[int],
    embeddings: np.ndarray,
    tau: float,
    metric: str = "cosine",
    t: float = 1.0,
) -> List[int]:
    """Greedily select a diverse subset from `indices`, in the given order (Eq 7).

    The first index is always kept. Each subsequent candidate is kept only
    if its max similarity to every already-selected member is strictly
    below `tau`. The result is order-dependent: callers pass candidates in
    agent-index order.

    Args:
        indices: candidate indices, in the order they should be considered.
        embeddings: (N, D) array indexed by the original indices.
        tau: similarity threshold; a candidate is rejected if its max
            similarity to the selected set is >= tau.
        metric: "cosine" (default) or "rbf" (Eq 5, uses `t`).
        t: RBF kernel temperature, only used when metric="rbf".

    Returns:
        Selected indices, in the order they were kept.
    """
    if metric == "cosine":
        sim_fn = cosine_similarity
    elif metric == "rbf":
        sim_fn = lambda u, v: rbf_similarity(u, v, t)
    else:
        raise ValueError(f"unknown metric: {metric!r}, expected 'cosine' or 'rbf'")

    if len(indices) == 0:
        return []

    selected: List[int] = [indices[0]]
    for idx in indices[1:]:
        max_sim = max(sim_fn(embeddings[idx], embeddings[s]) for s in selected)
        if max_sim < tau:
            selected.append(idx)
    return selected
