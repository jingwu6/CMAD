"""Pure-CPU local test for cmad/voting.py. No network, no GPU.

Run: python3 tests/test_voting_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.voting import build_pooled_vote_input, majority_vote


def test_unique_plurality():
    label, counts = majority_vote(["42", "42", "7"])
    assert label == "42", label
    assert counts == {"42": 2, "7": 1}, counts
    print("test_unique_plurality: PASS", label, counts)


def test_three_way_tie_lowest_agent_index_wins():
    # M=3, all distinct -> 3-way tie -> agent 0's answer ("42") wins.
    label, counts = majority_vote(["42", "7", "13"])
    assert label == "42", label
    print("test_three_way_tie_lowest_agent_index_wins: PASS", label, counts)


def test_two_two_tie_lowest_agent_index_wins():
    # M=4, 2-2 split -> tie between "42" (agents 0,1) and "7" (agents 2,3)
    # -> lowest agent index among tied answers is agent 0 -> "42" wins.
    label, counts = majority_vote(["42", "42", "7", "7"])
    assert label == "42", label
    print("test_two_two_tie_lowest_agent_index_wins: PASS", label, counts)


def test_two_two_tie_other_direction():
    # Swap which answer's agents come first -> "7" should win this time,
    # proving the rule really tracks agent index, not e.g. string value or
    # Counter insertion order coincidentally matching the previous test.
    label, counts = majority_vote(["7", "7", "42", "42"])
    assert label == "7", label
    print("test_two_two_tie_other_direction: PASS", label, counts)


def test_single_answer():
    label, counts = majority_vote(["42"])
    assert label == "42", label
    print("test_single_answer: PASS", label, counts)


def test_build_pooled_vote_input_flattens_all_rounds():
    # 2 rounds, M=3 each -- must flatten to 6 entries, in round-then-agent order.
    all_rounds = [["42", "7", "13"], ["42", "42", "13"]]
    pooled = build_pooled_vote_input(all_rounds)
    assert pooled == ["42", "7", "13", "42", "42", "13"], pooled
    print("test_build_pooled_vote_input_flattens_all_rounds: PASS", pooled)


def test_build_pooled_vote_input_none_entries_get_unique_placeholders():
    # None must never spuriously "win" a vote by colliding with another
    # None from a different round/agent -- each gets a placeholder unique
    # to its (round, agent) position.
    all_rounds = [["42", None, "42"], [None, "42", "42"]]
    pooled = build_pooled_vote_input(all_rounds)
    assert pooled == ["42", "__none__r0a1", "42", "__none__r1a0", "42", "42"], pooled
    label, counts = majority_vote(pooled)
    assert label == "42", (label, counts)  # 4 real votes for "42" vs 2 unique None placeholders
    print("test_build_pooled_vote_input_none_entries_get_unique_placeholders: PASS", pooled)


def test_build_pooled_vote_input_more_votes_can_change_outcome():
    # Round 1 (M=3) alone would tie 1-1-1 -> lowest-agent-index rule picks
    # "42" (agent 0). Pooling round 2's votes too breaks the tie for real
    # with more evidence, picking "7" (2 votes) over "42" (1 vote) --
    # demonstrating pooling isn't a no-op, it can change the pseudo-label.
    round1 = ["42", "7", "13"]
    round2 = ["7", "7", "13"]
    final_round_only_label, _ = majority_vote(round1)
    assert final_round_only_label == "42", final_round_only_label  # baseline: final-round-only would pick this

    pooled = build_pooled_vote_input([round1, round2])
    pooled_label, pooled_counts = majority_vote(pooled)
    assert pooled_label == "7", (pooled_label, pooled_counts)
    print("test_build_pooled_vote_input_more_votes_can_change_outcome: PASS", pooled_label, pooled_counts)


if __name__ == "__main__":
    test_unique_plurality()
    test_three_way_tie_lowest_agent_index_wins()
    test_two_two_tie_lowest_agent_index_wins()
    test_two_two_tie_other_direction()
    test_single_answer()
    test_build_pooled_vote_input_flattens_all_rounds()
    test_build_pooled_vote_input_none_entries_get_unique_placeholders()
    test_build_pooled_vote_input_more_votes_can_change_outcome()
    print("\nAll cmad/voting.py local tests PASSED")
