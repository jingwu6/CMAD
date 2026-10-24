"""Pure-CPU local test that cmad/config.py's DatasetConfig actually changes
which prompts get sent and which extractor/scorer gets applied, using a
fake LLMClient (no network/GPU) that mirrors what a real model would return
for each prompt style. answer_extract.py's pure functions are covered
separately in test_math_answer_extract_local.py.

Run: python3 tests/test_dataset_config_integration_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.config import DATASET_CONFIGS
from cmad.eval import run_condition


class FakeClient:
    """Stands in for cmad.llm_client.LLMClient. Branches on prompt content
    the same way a real model would branch on its *instructions* -- if the
    prompt asks for \\boxed{}, "respond" with \\boxed{}; if it asks for
    "The answer is X", respond with that. This is what actually proves
    make_agent_fn/run_condition picked the right DatasetConfig prompts,
    rather than just asserting on the prompt *text* directly."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, model, temperature=0.7, max_tokens=1024, stop=None):
        self.calls.append(prompt)
        if "Propose" in prompt:  # STRATEGY_GENERATOR_PROMPT
            return "1. Strategy A\n2. Strategy B"
        if "\\boxed{}" in prompt:
            return "Working through it step by step. Final answer: \\boxed{\\frac{1}{2}}"
        if "The answer is" in prompt:
            return "Working through it step by step. The answer is 42"
        raise AssertionError(f"unrecognized prompt style (neither MATH nor GSM8K instruction found): {prompt!r}")


def test_run_condition_cot_uses_math_prompts_and_extractor():
    client = FakeClient()
    math_cfg = DATASET_CONFIGS["math"]()
    result = run_condition(
        "cot", client, [{"question": "What is 1/2 + 0?", "answer": "\\frac{1}{2}"}],
        model="fake-model", dataset_cfg=math_cfg,
    )
    assert result["accuracy"] == 1.0, result
    assert result["per_question"][0]["extracted"] == "\\frac{1}{2}", result["per_question"][0]
    assert any("\\boxed{}" in c for c in client.calls), "expected the MATH cot prompt (asks for \\boxed{}) to be sent"
    print("test_run_condition_cot_uses_math_prompts_and_extractor: PASS", result["per_question"][0])


def test_run_condition_cot_defaults_to_gsm8k_prompts_and_extractor():
    client = FakeClient()
    result = run_condition(
        "cot", client, [{"question": "What is 40 + 2?", "answer": "42"}],
        model="fake-model",  # no dataset_cfg -- must default to GSM8K
    )
    assert result["accuracy"] == 1.0, result
    assert result["per_question"][0]["extracted"] == "42", result["per_question"][0]
    assert any("The answer is" in c for c in client.calls), "expected the GSM8K cot prompt (asks for 'The answer is X') to be sent"
    print("test_run_condition_cot_defaults_to_gsm8k_prompts_and_extractor: PASS", result["per_question"][0])


def test_run_condition_cmad_uses_math_prompts_through_debate_and_strategy_gen():
    client = FakeClient()
    math_cfg = DATASET_CONFIGS["math"]()
    result = run_condition(
        "cmad", client, [{"question": "What is 1/2 + 0?", "answer": "\\frac{1}{2}"}],
        model="fake-debate-model", generator_model="fake-generator", debate_M=2, debate_N=1,
        dataset_cfg=math_cfg,
    )
    assert result["accuracy"] == 1.0, result
    # Confirms the FULL cmad path -- strategy generation, per-agent debate
    # prompts, and answer extraction -- all went through the MATH dataset_cfg,
    # not just the single-prompt baseline conditions covered above.
    assert any("Propose" in c for c in client.calls), "strategy generator was never called"
    assert any("\\boxed{}" in c for c in client.calls), "MATH-style debate prompt was never sent"
    print("test_run_condition_cmad_uses_math_prompts_through_debate_and_strategy_gen: PASS", result["per_question"][0])


def test_run_condition_mismatched_dataset_cfg_produces_wrong_score_not_crash():
    # If a MATH ground truth
    # ("\frac{1}{2}") were scored with GSM8K's numeric extractor+scorer, it
    # must fail cleanly (score 0, no answer extracted) rather than crash or
    # silently "match" -- proving the extractor/scorer really are coupled
    # to dataset_cfg, not accidentally still using GSM8K's regardless.
    client = FakeClient()
    math_cfg = DATASET_CONFIGS["math"]()
    # MATH prompts/extractor, but the response is GSM8K-style (this
    # shouldn't happen in real use since the prompt itself asks for \boxed{},
    # but simulates a model ignoring instructions) -- extractor should
    # correctly find no \boxed{} and return None, not crash.
    client.generate = lambda prompt, model, temperature=0.7, max_tokens=1024, stop=None: "The answer is 42"
    result = run_condition(
        "cot", client, [{"question": "Q", "answer": "42"}], model="fake-model", dataset_cfg=math_cfg,
    )
    assert result["per_question"][0]["extracted"] is None, result["per_question"][0]
    assert result["accuracy"] == 0.0, result
    print("test_run_condition_mismatched_dataset_cfg_produces_wrong_score_not_crash: PASS", result["per_question"][0])


if __name__ == "__main__":
    test_run_condition_cot_uses_math_prompts_and_extractor()
    test_run_condition_cot_defaults_to_gsm8k_prompts_and_extractor()
    test_run_condition_cmad_uses_math_prompts_through_debate_and_strategy_gen()
    test_run_condition_mismatched_dataset_cfg_produces_wrong_score_not_crash()
    print("\nAll dataset_cfg (GSM8K/MATH) integration wiring local tests PASSED")
