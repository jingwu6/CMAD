"""Pure-CPU local test for cmad/answer_extract.py's MATH-dataset support
(extract_boxed_answer, compute_math_score). No network, no GPU.

Run: python3 tests/test_math_answer_extract_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.answer_extract import compute_math_score, extract_boxed_answer


def test_simple_integer():
    got = extract_boxed_answer("Working through it step by step... The final answer is \\boxed{42}.")
    assert got == "42", got
    print("test_simple_integer: PASS", got)


def test_nested_braces_fraction():
    got = extract_boxed_answer("So the answer is \\boxed{\\frac{1}{2}}.")
    assert got == "\\frac{1}{2}", got
    print("test_nested_braces_fraction: PASS", got)


def test_multiple_boxed_takes_last():
    text = "First I thought \\boxed{3} but after checking, the correct answer is \\boxed{5}."
    got = extract_boxed_answer(text)
    assert got == "5", got
    print("test_multiple_boxed_takes_last: PASS", got)


def test_no_boxed_present_returns_none():
    got = extract_boxed_answer("I solved it and got 42 but forgot to box it.")
    assert got is None, got
    print("test_no_boxed_present_returns_none: PASS", got)


def test_none_input_returns_none():
    assert extract_boxed_answer(None) is None
    assert extract_boxed_answer("") is None
    print("test_none_input_returns_none: PASS")


def test_unbalanced_braces_does_not_crash():
    # A malformed model output missing the closing brace -- must not raise,
    # and since it never closes, there's no complete boxed content to
    # extract (matches the "unboxed response = no answer" contract).
    got = extract_boxed_answer("The answer is \\boxed{42")
    assert got is None, got
    print("test_unbalanced_braces_does_not_crash: PASS", got)


def test_deeply_nested_braces():
    # \boxed{\left(\frac{1}{2}, \frac{3}{4}\right)} -- 3 levels of nesting,
    # naive non-brace-aware regex ([^}]*) would truncate at the first inner
    # "}" and return "\left(\frac{1", which is exactly the bug this
    # brace-counting implementation exists to avoid.
    text = "\\boxed{\\left(\\frac{1}{2}, \\frac{3}{4}\\right)}"
    got = extract_boxed_answer(text)
    assert got == "\\left(\\frac{1}{2}, \\frac{3}{4}\\right)", got
    print("test_deeply_nested_braces: PASS", got)


def test_score_exact_match():
    assert compute_math_score("42", "42") == 1.0
    assert compute_math_score("42", "43") == 0.0
    print("test_score_exact_match: PASS")


def test_score_none_extracted_is_wrong():
    assert compute_math_score(None, "42") == 0.0
    print("test_score_none_extracted_is_wrong: PASS")


def test_score_dfrac_tfrac_normalize_to_frac():
    assert compute_math_score("\\dfrac{1}{2}", "\\frac{1}{2}") == 1.0
    assert compute_math_score("\\tfrac{1}{2}", "\\frac{1}{2}") == 1.0
    print("test_score_dfrac_tfrac_normalize_to_frac: PASS")


def test_score_left_right_and_spacing_stripped():
    a = "\\left(\\frac{1}{2}, \\frac{3}{4}\\right)"
    b = "(\\frac{1}{2},\\frac{3}{4})"
    assert compute_math_score(a, b) == 1.0, (a, b)
    print("test_score_left_right_and_spacing_stripped: PASS")


def test_score_does_not_falsely_equate_different_values():
    # Deliberately checking the normalizer isn't so aggressive it collapses
    # genuinely different answers -- e.g. 1/2 vs 1/3 must NOT match even
    # though both are "\frac{1}{N}" shaped.
    assert compute_math_score("\\frac{1}{2}", "\\frac{1}{3}") == 0.0
    print("test_score_does_not_falsely_equate_different_values: PASS")


def test_score_percent_and_trailing_period():
    assert compute_math_score("50\\%.", "50%") == 1.0
    print("test_score_percent_and_trailing_period: PASS")


if __name__ == "__main__":
    test_simple_integer()
    test_nested_braces_fraction()
    test_multiple_boxed_takes_last()
    test_no_boxed_present_returns_none()
    test_none_input_returns_none()
    test_unbalanced_braces_does_not_crash()
    test_deeply_nested_braces()
    test_score_exact_match()
    test_score_none_extracted_is_wrong()
    test_score_dfrac_tfrac_normalize_to_frac()
    test_score_left_right_and_spacing_stripped()
    test_score_does_not_falsely_equate_different_values()
    test_score_percent_and_trailing_period()
    print("\nAll cmad/answer_extract.py MATH-support local tests PASSED")
