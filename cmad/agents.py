"""Debate-agent wrappers over LLMClient. Orchestration lives in debate.py."""

from cmad.answer_extract import extract_final_numeric_answer
from cmad.llm_client import LLMClient
from cmad.prompts import (
    AGENT_DEBATE_ROUND_PROMPT,
    AGENT_ROUND1_NO_STRATEGY_PROMPT,
    AGENT_ROUND1_PROMPT,
)


def make_agent_fn(client: LLMClient, model: str, temperature: float = 0.7, dataset_cfg=None):
    """Return an agent_fn(agent_index, question, ctx) -> raw_text for
    debate.run_debate.

    `ctx` is None (round 1, vanilla MAD), a strategy string (round 1,
    DMAD/CMAD), or a shared-history string from build_shared_history
    (rounds 2+, always starting with "Agent 0:").

    `dataset_cfg` (cmad.config.DatasetConfig) selects the prompt wording;
    defaults to GSM8K.
    """
    round1_prompt = dataset_cfg.round1_prompt if dataset_cfg is not None else AGENT_ROUND1_PROMPT
    round1_no_strategy_prompt = dataset_cfg.round1_no_strategy_prompt if dataset_cfg is not None else AGENT_ROUND1_NO_STRATEGY_PROMPT
    debate_round_prompt = dataset_cfg.debate_round_prompt if dataset_cfg is not None else AGENT_DEBATE_ROUND_PROMPT
    max_tokens = dataset_cfg.max_tokens if dataset_cfg is not None else 1024

    def agent_fn(agent_index: int, question: str, ctx):
        if ctx is None:
            prompt = round1_no_strategy_prompt.format(question=question)
        elif isinstance(ctx, str) and ctx.startswith("Agent 0:"):
            prompt = debate_round_prompt.format(question=question, history=ctx)
        else:
            prompt = round1_prompt.format(question=question, strategy=ctx)
        return client.generate(prompt, model=model, temperature=temperature, max_tokens=max_tokens)

    return agent_fn


def extract_answers(raw_texts, extractor=extract_final_numeric_answer):
    """Apply `extractor` (GSM8K's numeric extractor by default) to each raw agent output."""
    return [extractor(t) for t in raw_texts]
