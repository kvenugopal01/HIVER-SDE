import pytest
from evaluation.metrics import compute_classification_metrics, compute_escalation_metrics

def test_compute_classification_metrics():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    res = compute_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.75
    assert "macro_f1" in res
    assert "per_intent" in res

def test_compute_escalation_metrics():
    expected = ["AUTO_HANDLE", "ESCALATE", "ESCALATE"]
    actual = ["AUTO_HANDLE", "AUTO_HANDLE", "ESCALATE"]
    res = compute_escalation_metrics(expected, actual)
    assert res["auto_handle_pct"] == 66.67
    assert res["escalation_recall"] == 0.5
