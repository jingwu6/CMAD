"""Strategy generation for CMAD's learned strategy generator G."""

import re
from typing import List, Optional

from cmad.llm_client import LLMClient
from cmad.prompts import DMAD_FIXED_STRATEGIES, STRATEGY_GENERATOR_PROMPT, STRATEGY_GENERATOR_PROMPT_WITH_SUBJECT

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")


def parse_strategies(raw_text: str, m: int) -> List[str]:
    """Parse G's numbered-list output into exactly m strategy strings.

    Falls back to treating each non-empty line as a strategy if the numbered
    format wasn't followed, and pads with a generic strategy if fewer than m
    were recovered."""
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    strategies = []
    for line in lines:
        match = _NUMBERED_LINE_RE.match(line)
        strategies.append(match.group(1).strip() if match else line)

    strategies = strategies[:m]
    while len(strategies) < m:
        strategies.append("Solve the problem using straightforward step-by-step reasoning.")
    return strategies


def generate_strategies(client: LLMClient, question: str, m: int, model: str, temperature: float = 1.0,
                         subject: Optional[str] = None) -> List[str]:
    """Call G to produce m strategies for `question`.

    `subject`: MATH's subject label (e.g. "Algebra"); when given, G is told
    the subject so it can tailor its strategies. None (always the case for
    GSM8K) uses the generic prompt."""
    if subject is not None:
        prompt = STRATEGY_GENERATOR_PROMPT_WITH_SUBJECT.format(question=question, m=m, subject=subject)
    else:
        prompt = STRATEGY_GENERATOR_PROMPT.format(question=question, m=m)
    raw = client.generate(prompt, model=model, temperature=temperature, max_tokens=512)
    return parse_strategies(raw, m)


def get_dmad_strategies(m: int) -> List[str]:
    """DMAD baseline's fixed strategy set, cycled if m exceeds its length."""
    return [DMAD_FIXED_STRATEGIES[i % len(DMAD_FIXED_STRATEGIES)] for i in range(m)]
