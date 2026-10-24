"""Pure-CPU local test for cmad/debate.py's orchestration wiring, using a
fake/mocked agent_fn (no LLM/network/GPU calls). Also covers the
alignment-plumbing wiring in cmad/sft_dataset.py using a fake align_fn.

Run: python3 tests/test_debate_orchestration_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.debate import build_shared_history, run_debate
from cmad.sft_dataset import build_sft_examples


def test_round1_uses_per_agent_strategy():
    M = 3
    strategies = ["Sx", "Sy", "Sz"]
    captured = []

    def fake_agent_fn(agent_index, question, ctx):
        round_idx = len(captured) // M
        captured.append((round_idx, agent_index, ctx))
        if round_idx == 0:
            return f"R1_A{agent_index}_{ctx}"
        return f"R{round_idx + 1}_A{agent_index}"

    rounds = run_debate("Q", strategies, M=M, N=1, agent_fn=fake_agent_fn)
    assert rounds == [["R1_A0_Sx", "R1_A1_Sy", "R1_A2_Sz"]], rounds
    # Round-1 context passed to each agent is exactly its own assigned
    # strategy, not the question text or anything else.
    round1_calls = [c for c in captured if c[0] == 0]
    assert round1_calls == [(0, 0, "Sx"), (0, 1, "Sy"), (0, 2, "Sz")], round1_calls
    print("test_round1_uses_per_agent_strategy: PASS", rounds)


def test_round2_history_is_built_from_round1_only():
    M, N = 3, 2
    strategies = ["Sx", "Sy", "Sz"]
    captured = []

    def fake_agent_fn(agent_index, question, ctx):
        round_idx = len(captured) // M
        captured.append((round_idx, agent_index, ctx))
        if round_idx == 0:
            return f"R1_A{agent_index}_{ctx}"
        return f"R{round_idx + 1}_A{agent_index}"

    rounds = run_debate("Q", strategies, M=M, N=N, agent_fn=fake_agent_fn)
    round1_outputs = rounds[0]
    expected_history = build_shared_history(round1_outputs)

    round2_calls = [c for c in captured if c[0] == 1]
    assert len(round2_calls) == M, round2_calls
    # All three agents in round 2 see the identical history string, and it
    # is built from round 1's outputs (agent-index order), not round 2's
    # own outputs (which don't exist yet at the time it's built).
    for (_, agent_idx, ctx) in round2_calls:
        assert ctx == expected_history, (agent_idx, ctx, expected_history)
    print("test_round2_history_is_built_from_round1_only: PASS")


def test_round3_history_excludes_round1_not_cumulative():
    M, N = 3, 3
    strategies = ["Sx", "Sy", "Sz"]
    captured = []

    def fake_agent_fn(agent_index, question, ctx):
        round_idx = len(captured) // M
        captured.append((round_idx, agent_index, ctx))
        if round_idx == 0:
            return f"R1_A{agent_index}_{ctx}"
        return f"R{round_idx + 1}_A{agent_index}"

    rounds = run_debate("Q", strategies, M=M, N=N, agent_fn=fake_agent_fn)
    round2_outputs = rounds[1]
    expected_history_r3 = build_shared_history(round2_outputs)

    round3_calls = [c for c in captured if c[0] == 2]
    assert len(round3_calls) == M, round3_calls
    for (_, agent_idx, ctx) in round3_calls:
        assert ctx == expected_history_r3, (agent_idx, ctx, expected_history_r3)
        # Round 1's raw strategy text ("Sx") only ever appears inside round-1
        # output strings ("R1_A0_Sx") -- if round 3's history were a running
        # accumulation of rounds 1+2 (instead of round 2 only), "Sx" would
        # leak in via round 1's output text. It must not.
        assert "Sx" not in ctx, ctx
    print("test_round3_history_excludes_round1_not_cumulative: PASS")


def test_vanilla_mad_baseline_no_strategies():
    # strategies=None (vanilla multi-agent-debate baseline): every agent
    # gets None as its round-1 context, not a distinct strategy -- this is
    # the exact wiring that must differ from DMAD/CMAD's per-agent strategies.
    M = 3
    captured = []

    def fake_agent_fn(agent_index, question, ctx):
        captured.append(ctx)
        return f"A{agent_index}_generic"

    rounds = run_debate("Q", None, M=M, N=1, agent_fn=fake_agent_fn)
    assert captured == [None, None, None], captured
    assert rounds == [["A0_generic", "A1_generic", "A2_generic"]], rounds
    print("test_vanilla_mad_baseline_no_strategies: PASS")


def test_alignment_applied_only_to_selected_rows():
    strategies = ["S0", "S1", "S2", "S3"]
    final_answers_raw = ["ans0", "ans1", "ans2", "ans3"]
    # Only indices 0 and 2 survive correctness+diversity filtering (indices
    # 1 and 3 are rejected and must never be touched by align_fn).
    selected_indices = [0, 2]

    align_calls = []

    def fake_align_fn(question, strategy, final_answer):
        align_calls.append((question, strategy, final_answer))
        return f"{strategy}[ALIGNED]"

    rows = build_sft_examples(
        question="Q",
        strategies=strategies,
        final_answers_raw=final_answers_raw,
        selected_indices=selected_indices,
        align_fn=fake_align_fn,
    )

    # One row per QUESTION (not per candidate) -- the completion is the full
    # list of surviving aligned strategies, matching G's own "produce M
    # strategies in one numbered list" generation format (see lora_sft.py).
    assert len(rows) == 1, rows
    assert rows[0] == {"question": "Q", "target_strategies": ["S0[ALIGNED]", "S2[ALIGNED]"]}, rows[0]
    # Alignment was called exactly twice, only for the selected (strategy,
    # answer) pairs -- never for S1/ans1 or S3/ans3.
    assert align_calls == [("Q", "S0", "ans0"), ("Q", "S2", "ans2")], align_calls
    print("test_alignment_applied_only_to_selected_rows: PASS", rows)


def test_no_alignment_uses_raw_strategy():
    rows = build_sft_examples(
        question="Q",
        strategies=["S0", "S1"],
        final_answers_raw=["ans0", "ans1"],
        selected_indices=[1],
        align_fn=None,
    )
    assert rows == [{"question": "Q", "target_strategies": ["S1"]}], rows
    print("test_no_alignment_uses_raw_strategy: PASS", rows)


def test_all_three_selected_group_into_one_row_in_order():
    # All M=3 candidates survive filtering -- must produce exactly ONE row
    # whose target_strategies lists all three, in selected_indices order,
    # matching G's own "M strategies in one numbered-list completion"
    # generation format.
    rows = build_sft_examples(
        question="Q",
        strategies=["Sa", "Sb", "Sc"],
        final_answers_raw=["ansA", "ansB", "ansC"],
        selected_indices=[0, 1, 2],
        align_fn=None,
    )
    assert rows == [{"question": "Q", "target_strategies": ["Sa", "Sb", "Sc"]}], rows
    print("test_all_three_selected_group_into_one_row_in_order: PASS", rows)


def test_no_selected_candidates_produces_no_row():
    rows = build_sft_examples(
        question="Q",
        strategies=["S0", "S1"],
        final_answers_raw=["ans0", "ans1"],
        selected_indices=[],
        align_fn=None,
    )
    assert rows == [], rows
    print("test_no_selected_candidates_produces_no_row: PASS")


if __name__ == "__main__":
    test_round1_uses_per_agent_strategy()
    test_round2_history_is_built_from_round1_only()
    test_round3_history_excludes_round1_not_cumulative()
    test_vanilla_mad_baseline_no_strategies()
    test_alignment_applied_only_to_selected_rows()
    test_no_alignment_uses_raw_strategy()
    test_all_three_selected_group_into_one_row_in_order()
    test_no_selected_candidates_produces_no_row()
    print("\nAll debate-orchestration + alignment-plumbing local tests PASSED")
