"""Prompt templates for every role in the CMAD pipeline and its baselines."""

STRATEGY_GENERATOR_PROMPT = """You are designing distinct problem-solving strategies for a math question.
Question: {question}

Propose {m} different high-level strategies for approaching this question. Each strategy should describe a distinct way of THINKING about the problem (e.g. what to focus on first, what kind of reasoning to use), not the final answer itself. Make the {m} strategies as different from each other as possible.

Respond with exactly {m} strategies, one per line, formatted as:
1. <strategy 1>
2. <strategy 2>
...
"""

# Used for MATH, whose problems carry a subject label (Algebra, Geometry, ...).
STRATEGY_GENERATOR_PROMPT_WITH_SUBJECT = """You are designing distinct problem-solving strategies for a {subject} problem.
Question: {question}

Propose {m} different high-level strategies for approaching this question, each one well-suited to {subject} problems specifically (e.g. a strategy that works well for {subject} may differ from what works for other math subjects). Each strategy should describe a distinct way of THINKING about the problem (e.g. what to focus on first, what kind of reasoning to use), not the final answer itself. Make the {m} strategies as different from each other as possible.

Respond with exactly {m} strategies, one per line, formatted as:
1. <strategy 1>
2. <strategy 2>
...
"""

AGENT_ROUND1_PROMPT = """Question: {question}

Solve this question using the following strategy:
{strategy}

Show your reasoning step by step, then clearly state your final numeric answer at the end as "The answer is X"."""

AGENT_ROUND1_NO_STRATEGY_PROMPT = """Question: {question}

Solve this question step by step, then clearly state your final numeric answer at the end as "The answer is X"."""

AGENT_DEBATE_ROUND_PROMPT = """Question: {question}

Here are the solutions from all agents in the previous round:
{history}

Review these solutions. If you find a mistake in your own or another agent's reasoning, correct it. Provide your own updated, step-by-step solution to the question, then clearly state your final numeric answer at the end as "The answer is X"."""

COT_PROMPT = """Question: {question}

Let's think step by step, then clearly state the final numeric answer at the end as "The answer is X"."""

STEP_BACK_PROMPT = """Question: {question}

First, step back and identify the general mathematical principle or concept underlying this question. Then apply that principle to solve the question step by step, and clearly state the final numeric answer at the end as "The answer is X"."""

PROGRAM_OF_THOUGHT_PROMPT = """Question: {question}

Solve this question by writing out the calculation as a short sequence of explicit arithmetic expressions (as if writing lines of code), one operation per line, rather than prose. Then clearly state the final numeric answer at the end as "The answer is X"."""

SELF_REFLECTION_PROMPT = """Question: {question}

First, solve the question step by step and state a final numeric answer.
Then, critically review your own solution: check every arithmetic step and every assumption for mistakes.
If you find any mistake, correct it and provide a revised final answer.
Clearly state your final numeric answer at the end as "The answer is X"."""

SELF_CONTRAST_PROMPT = """Question: {question}

Solve this question using two different reasoning approaches independently. Compare the two solutions: if they agree, state that answer as final. If they disagree, identify which approach made an error and why, then give the corrected final answer.
Clearly state your final numeric answer at the end as "The answer is X"."""

ALIGNMENT_PROMPT = """Question: {question}

A strategy was proposed for solving this question:
{strategy}

The following solution was ultimately judged correct:
{final_answer}

Rewrite the strategy description so that it accurately describes the reasoning approach actually used in the correct solution above. Keep it as a single short paragraph describing the approach, not the solution itself.

Rewritten strategy:"""

DMAD_FIXED_STRATEGIES = [
    "Use chain-of-thought reasoning: work through the problem step by step in plain language.",
    "Use step-back prompting: first identify the general principle underlying the problem, then apply it.",
    "Use program-of-thought reasoning: express the solution as a short sequence of explicit arithmetic expressions.",
]

# --- MATH variants: same structure, but ask for the final answer in \boxed{}
# since MATH answers are often non-numeric. ---

AGENT_ROUND1_PROMPT_MATH = """Question: {question}

Solve this question using the following strategy:
{strategy}

Show your reasoning step by step, then clearly state your final answer at the end wrapped in \\boxed{{}}, e.g. \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."""

AGENT_ROUND1_NO_STRATEGY_PROMPT_MATH = """Question: {question}

Solve this question step by step, then clearly state your final answer at the end wrapped in \\boxed{{}}, e.g. \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."""

AGENT_DEBATE_ROUND_PROMPT_MATH = """Question: {question}

Here are the solutions from all agents in the previous round:
{history}

Review these solutions. If you find a mistake in your own or another agent's reasoning, correct it. Provide your own updated, step-by-step solution to the question, then clearly state your final answer at the end wrapped in \\boxed{{}}."""

COT_PROMPT_MATH = """Question: {question}

Let's think step by step, then clearly state the final answer at the end wrapped in \\boxed{{}}."""

STEP_BACK_PROMPT_MATH = """Question: {question}

First, step back and identify the general mathematical principle or concept underlying this question. Then apply that principle to solve the question step by step, and clearly state the final answer at the end wrapped in \\boxed{{}}."""

PROGRAM_OF_THOUGHT_PROMPT_MATH = """Question: {question}

Solve this question by writing out the calculation as a short sequence of explicit arithmetic/symbolic expressions (as if writing lines of code), one operation per line, rather than prose. Then clearly state the final answer at the end wrapped in \\boxed{{}}."""

SELF_REFLECTION_PROMPT_MATH = """Question: {question}

First, solve the question step by step and state a final answer.
Then, critically review your own solution: check every step and every assumption for mistakes.
If you find any mistake, correct it and provide a revised final answer.
Clearly state your final answer at the end wrapped in \\boxed{{}}."""

SELF_CONTRAST_PROMPT_MATH = """Question: {question}

Solve this question using two different reasoning approaches independently. Compare the two solutions: if they agree, state that answer as final. If they disagree, identify which approach made an error and why, then give the corrected final answer.
Clearly state your final answer at the end wrapped in \\boxed{{}}."""
