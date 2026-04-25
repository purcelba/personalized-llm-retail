"""Unit tests for metric computation in evaluate.py."""
from src.evaluate import hit_rate_at_k, ndcg_at_k


def test_hit_rate_hit():
    assert hit_rate_at_k(["A", "B", "C"], {"A"}, k=3) == 1


def test_hit_rate_miss():
    assert hit_rate_at_k(["A", "B", "C"], {"D"}, k=3) == 0


def test_hit_rate_respects_k():
    assert hit_rate_at_k(["A", "B", "C"], {"C"}, k=2) == 0


def test_ndcg_perfect():
    assert ndcg_at_k(["A"], {"A"}, k=1) == 1.0


def test_ndcg_miss():
    assert ndcg_at_k(["A", "B"], {"C"}, k=2) == 0.0
