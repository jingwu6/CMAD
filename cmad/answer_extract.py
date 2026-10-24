"""Numeric answer extraction and scoring for GSM8K-style free-form chat answers.

CMAD agents are not trained to emit GSM8K's canonical "#### N" format -- they
just write out a chat-style solution and (ideally) state the final number
somewhere near the end, optionally via "the answer is N" or "\\boxed{N}".
Adapted from verl's own gsm8k "flexible" extractor (last number found in the
text), extended to prefer a number that follows an explicit answer cue over
a bare last-number match, since free-form chat text is noisier than GSM8K's
own step-by-step solution format.
"""

import re

_NUMBER_RE = re.compile(r"-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?")
_CUED_RE = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer\s*:?|answer\s*:)\s*\$?\\?boxed\{([^}]*)\}"
    r"|(?:the\s+answer\s+is|final\s+answer\s*:?|answer\s*:)\s*\$?(-?[0-9,]+(?:\.[0-9]+)?)"
    r"|\\boxed\{([^}]*)\}",
    re.IGNORECASE,
)


def _normalize_number(raw: str):
    if raw is None:
        return None
    cleaned = raw.replace(",", "").replace("$", "").strip().rstrip(".")
    if cleaned in ("", "-", "."):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    # Normalize to int-like string when it is a whole number, matching how
    # GSM8K ground-truth answers are always written (no ".0" suffix).
    if value == int(value):
        return str(int(value))
    return str(value)


def extract_final_numeric_answer(solution_str: str):
    """Return the model's final numeric answer as a normalized string, or None.

    Preference order: (1) a number following an explicit answer cue
    ("the answer is", "final answer:", "\\boxed{...}") -- last such cue wins
    if there are several; (2) otherwise, the last bare number anywhere in the
    text (verl's "flexible" GSM8K convention).
    """
    if not solution_str:
        return None

    cued_matches = list(_CUED_RE.finditer(solution_str))
    if cued_matches:
        last = cued_matches[-1]
        raw = last.group(1) or last.group(2) or last.group(3)
        normalized = _normalize_number(raw)
        if normalized is not None:
            return normalized

    all_numbers = _NUMBER_RE.findall(solution_str)
    for raw in reversed(all_numbers):
        normalized = _normalize_number(raw)
        if normalized is not None:
            return normalized
    return None


def compute_score(extracted_answer, ground_truth) -> float:
    """1.0 if extracted_answer matches ground_truth (both normalized), else 0.0."""
    if extracted_answer is None:
        return 0.0
    normalized_gt = _normalize_number(str(ground_truth))
    return 1.0 if extracted_answer == normalized_gt else 0.0


# --- MATH support. MATH answers are often non-numeric (fractions, radicals,
# intervals, sets), so answers are read from the last \boxed{...} with
# brace-aware matching (handles nesting such as \boxed{\frac{1}{2}}). ---

_LATEX_STRIP_TOKENS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "~")


def extract_boxed_answer(solution_str: str):
    """Return the contents of the LAST \\boxed{...} in solution_str (brace
    matching handles nesting), or None if no \\boxed{} is present.

    Unlike GSM8K's extractor, there is no "last bare number" fallback --
    MATH answers are frequently non-numeric, so a bare-number fallback would
    silently produce nonsense on e.g. answers like "\\frac{1}{2}" or
    "(3, 5)". An unboxed model response is treated as no-answer.
    """
    if not solution_str:
        return None

    marker = "\\boxed{"
    boxed_contents = []
    search_start = 0
    while True:
        start = solution_str.find(marker, search_start)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(solution_str) and depth > 0:
            if solution_str[i] == "{":
                depth += 1
            elif solution_str[i] == "}":
                depth -= 1
            i += 1
        if depth != 0:
            break  # unbalanced braces from here on -- stop rather than misparse
        boxed_contents.append(solution_str[content_start : i - 1])
        search_start = i

    return boxed_contents[-1].strip() if boxed_contents else None


def _normalize_math_answer(raw):
    """Best-effort normalization for comparing two boxed MATH answers as
    strings. This is NOT a CAS -- it will not recognize e.g. "\\frac{1}{2}"
    and "0.5" as equivalent, only textually-equivalent-up-to-LaTeX-noise
    answers.
    """
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for tok in _LATEX_STRIP_TOKENS:
        s = s.replace(tok, "")
    s = s.replace(" ", "")
    s = s.strip("$")
    s = s.rstrip(".")
    s = s.replace("\\%", "%")
    if s.startswith("+"):
        s = s[1:]
    return s if s != "" else None


def compute_math_score(extracted_answer, ground_truth) -> float:
    """1.0 if extracted_answer and ground_truth normalize to the same
    string, else 0.0. `ground_truth` here is already the bare boxed answer
    (e.g. from scripts/prepare_math.py extracting it out of the dataset's
    "solution" field), not the full solution text."""
    a = _normalize_math_answer(extracted_answer)
    b = _normalize_math_answer(str(ground_truth) if ground_truth is not None else None)
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0
