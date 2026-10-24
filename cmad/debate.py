"""Multi-agent debate orchestration (paper Eq 1-3, Du et al. 2023 protocol).

`run_debate` is a pure orchestration loop over an injectable `agent_fn`
callable, so it can be tested with fake agents (see tests/test_debate_orchestration_local.py).
"""

from typing import Callable, List, Optional


def build_shared_history(round_responses: List[str]) -> str:
    """Aggregate one round's responses (in agent-index order) into the
    shared history string handed to every agent for the *next* round.

    Per the paper, h_{n-1} is built from round n-1's responses ONLY -- this
    is not a running concatenation across all prior rounds. Round n's
    prompt sees exactly build_shared_history(round n-1's responses), never
    round n-2's or earlier.
    """
    lines = [f"Agent {i}: {resp}" for i, resp in enumerate(round_responses)]
    return "\n".join(lines)


def run_debate(
    question: str,
    strategies: Optional[List[str]],
    M: int,
    N: int,
    agent_fn: Callable,
) -> List[List[str]]:
    """Run M agents through N rounds of debate on `question`.

    Args:
        question: the question text.
        strategies: list of M per-agent strategy strings for round 1, or
            None for the vanilla-multi-agent-debate baseline (all agents
            get the same generic prompt, no per-agent strategy).
        M: number of agents.
        N: number of debate rounds (>= 1).
        agent_fn: callable with two possible signatures, dispatched by round:
            round 1:  agent_fn(agent_index, question, strategy_or_None) -> str
            round n>1: agent_fn(agent_index, question, history_str) -> str
            (see agents.make_agent_fn).

    Returns:
        List of N rounds, each a list of M response strings (agent-index
        order).
    """
    if strategies is not None:
        assert len(strategies) == M, f"expected {M} strategies, got {len(strategies)}"

    rounds: List[List[str]] = []

    round1 = [
        agent_fn(i, question, strategies[i] if strategies is not None else None)
        for i in range(M)
    ]
    rounds.append(round1)

    for _ in range(1, N):
        history = build_shared_history(rounds[-1])
        round_n = [agent_fn(i, question, history) for i in range(M)]
        rounds.append(round_n)

    return rounds
