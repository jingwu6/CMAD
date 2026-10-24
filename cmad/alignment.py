"""Strategy refinement / alignment agent (paper Eq 8): rewrites a strategy's
description to match the reasoning actually used in the final answer."""

from cmad.llm_client import LLMClient
from cmad.prompts import ALIGNMENT_PROMPT


def make_align_fn(client: LLMClient, model: str, temperature: float = 0.3):
    """Return align_fn(question, strategy, final_answer) -> aligned_strategy,
    as expected by sft_dataset.build_sft_examples."""

    def align_fn(question: str, strategy: str, final_answer: str) -> str:
        prompt = ALIGNMENT_PROMPT.format(question=question, strategy=strategy, final_answer=final_answer)
        return client.generate(prompt, model=model, temperature=temperature, max_tokens=256).strip()

    return align_fn
