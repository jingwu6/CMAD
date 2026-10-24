"""Pure-CPU local test for cmad/selection.py (correctness + diversity sample
selection), using synthetic examples with hand-computed expected outputs.
No network, no GPU.

Run: python3 tests/test_selection_local.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np

from cmad.selection import correctness_filter, cosine_similarity, greedy_diversity_select, rbf_similarity


def test_correctness_filter_drops_disagreeing_answer():
    answers = ["42", "42", "42", "7", "42"]
    pseudo_label = "42"
    kept = correctness_filter(answers, pseudo_label)
    assert kept == [0, 1, 2, 4], kept
    print("test_correctness_filter_drops_disagreeing_answer: PASS", kept)


def test_rbf_similarity_matches_hand_computed():
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 0.0])
    p3 = np.array([5.0, 5.0])
    t = 1.0
    sim_01 = rbf_similarity(p0, p1, t)
    sim_00 = rbf_similarity(p0, p0, t)
    sim_03 = rbf_similarity(p0, p3, t)
    assert math.isclose(sim_01, math.exp(-1.0), rel_tol=1e-6), sim_01
    assert math.isclose(sim_00, 1.0, rel_tol=1e-6), sim_00
    assert sim_03 < 1e-10, sim_03
    print("test_rbf_similarity_matches_hand_computed: PASS", sim_01, sim_00, sim_03)


def test_cosine_similarity_matches_hand_computed():
    # Clean angles: 0 deg (identical), 90 deg (orthogonal), 180 deg (opposite).
    p0 = np.array([1.0, 0.0])
    p90 = np.array([0.0, 1.0])
    p180 = np.array([-1.0, 0.0])
    assert math.isclose(cosine_similarity(p0, p0), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine_similarity(p0, p90), 0.0, abs_tol=1e-9)
    assert math.isclose(cosine_similarity(p0, p180), -1.0, rel_tol=1e-6)
    # Scale-invariance: cosine similarity depends only on direction.
    assert math.isclose(cosine_similarity(p0, p0 * 5.0), 1.0, rel_tol=1e-6)
    print("test_cosine_similarity_matches_hand_computed: PASS")


def test_cosine_similarity_zero_vector_does_not_crash():
    zero = np.array([0.0, 0.0])
    p0 = np.array([1.0, 0.0])
    assert cosine_similarity(zero, p0) == 0.0
    assert cosine_similarity(zero, zero) == 0.0
    print("test_cosine_similarity_zero_vector_does_not_crash: PASS")


def test_greedy_diversity_select_defaults_to_cosine_metric():
    # metric="cosine" is greedy_diversity_select's default (no metric= kwarg
    # passed here) -- confirms the actual default, not just that cosine
    # works when explicitly requested.
    p0 = np.array([1.0, 0.0])
    p90 = np.array([0.0, 1.0])
    embeddings = np.array([p0, p90])
    selected = greedy_diversity_select([0, 1], embeddings, tau=0.5)
    assert selected == [0, 1], selected  # cos(90deg)=0 < 0.5 -> kept
    print("test_greedy_diversity_select_defaults_to_cosine_metric: PASS", selected)


def test_greedy_diversity_select_cosine_hand_computed_forward_order():
    # 4 candidates at clean angles from p0=(1,0): p1 at 90deg (orthogonal),
    # p2 == p0 (exact duplicate, 0deg), p3 at 180deg (opposite).
    embeddings = np.array([
        [1.0, 0.0],   # 0
        [0.0, 1.0],   # 1 (90deg from 0)
        [1.0, 0.0],   # 2 (duplicate of 0)
        [-1.0, 0.0],  # 3 (180deg from 0)
    ])
    tau = 0.5

    # Hand-computed: cos(0,1)=0<0.5 -> keep 1.
    # cos(2,0)=1.0, cos(2,1)=0 -> max=1.0>=0.5 -> reject 2.
    # cos(3,0)=-1.0, cos(3,1)=0 -> max=0<0.5 -> keep 3.
    selected = greedy_diversity_select([0, 1, 2, 3], embeddings, tau=tau)
    assert selected == [0, 1, 3], selected
    print("test_greedy_diversity_select_cosine_hand_computed_forward_order: PASS", selected)


def test_greedy_diversity_select_cosine_is_order_dependent():
    embeddings = np.array([
        [1.0, 0.0],   # 0
        [0.0, 1.0],   # 1
        [1.0, 0.0],   # 2 (duplicate of 0)
        [-1.0, 0.0],  # 3
    ])
    tau = 0.5

    # Order [2,0,1,3]: seed 2. cos(0,2)=1.0>=0.5 -> reject 0 this time
    # (whichever duplicate comes first survives). cos(1,2)=0<0.5 -> keep 1.
    # cos(3,{2,1})=max(-1.0,0)=0<0.5 -> keep 3.
    selected = greedy_diversity_select([2, 0, 1, 3], embeddings, tau=tau)
    assert selected == [2, 1, 3], selected

    forward_selected = greedy_diversity_select([0, 1, 2, 3], embeddings, tau=tau)
    assert selected != forward_selected, (selected, forward_selected)
    print("test_greedy_diversity_select_cosine_is_order_dependent: PASS", selected, "vs forward:", forward_selected)


def test_greedy_diversity_select_rbf_metric_still_available():
    # metric="rbf" (Eq 5 kernel).
    embeddings = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [5.0, 5.0],
    ])
    selected = greedy_diversity_select([0, 1, 2, 3], embeddings, tau=0.5, metric="rbf", t=1.0)
    assert selected == [0, 1, 3], selected
    print("test_greedy_diversity_select_rbf_metric_still_available: PASS", selected)


def test_greedy_diversity_select_unknown_metric_raises():
    try:
        greedy_diversity_select([0, 1], np.array([[1.0, 0.0], [0.0, 1.0]]), tau=0.5, metric="euclidean")
        raise AssertionError("expected ValueError for unknown metric")
    except ValueError:
        print("test_greedy_diversity_select_unknown_metric_raises: PASS")


def test_greedy_diversity_select_single_candidate():
    embeddings = np.array([[1.0, 2.0]])
    selected = greedy_diversity_select([0], embeddings, tau=0.5)
    assert selected == [0], selected
    print("test_greedy_diversity_select_single_candidate: PASS")


def test_full_pipeline_correctness_then_diversity():
    # End-to-end: 5 candidates, pseudo_label=42.
    answers = ["42", "42", "42", "7", "42"]
    pseudo_label = "42"
    embeddings = np.array([
        [1.0, 0.0],    # 0
        [0.0, 1.0],    # 1 (90deg from 0)
        [1.0, 0.0],    # 2 (duplicate of 0)
        [-1.0, -1.0],  # 3 ("7" -- must never reach diversity selection)
        [-1.0, 0.0],   # 4 (180deg from 0)
    ])
    correct = correctness_filter(answers, pseudo_label)
    assert correct == [0, 1, 2, 4], correct
    selected = greedy_diversity_select(correct, embeddings, tau=0.5)
    assert selected == [0, 1, 4], selected
    assert 3 not in selected
    print("test_full_pipeline_correctness_then_diversity: PASS", selected)


if __name__ == "__main__":
    test_correctness_filter_drops_disagreeing_answer()
    test_rbf_similarity_matches_hand_computed()
    test_cosine_similarity_matches_hand_computed()
    test_cosine_similarity_zero_vector_does_not_crash()
    test_greedy_diversity_select_defaults_to_cosine_metric()
    test_greedy_diversity_select_cosine_hand_computed_forward_order()
    test_greedy_diversity_select_cosine_is_order_dependent()
    test_greedy_diversity_select_rbf_metric_still_available()
    test_greedy_diversity_select_unknown_metric_raises()
    test_greedy_diversity_select_single_candidate()
    test_full_pipeline_correctness_then_diversity()
    print("\nAll cmad/selection.py local tests PASSED")
