import pytest
from src.escalation import EscalationPolicyEngine

def test_escalation_sensitive_keyword():
    engine = EscalationPolicyEngine()
    res = engine.evaluate(
        customer_message="My account was hacked and there is an unauthorized charge",
        intent="ACCOUNT_APPLE_ID_ICLOUD",
        intent_confidence=0.95,
        retrieval_score=0.80,
        retrieved_evidence=[]
    )
    assert res["decision"] == "ESCALATE"
    assert "Sensitive keyword" in res["reason"] or "Intent category" in res["reason"]

def test_escalation_mandatory_intent():
    engine = EscalationPolicyEngine()
    res = engine.evaluate(
        customer_message="My iPhone screen shattered",
        intent="HARDWARE_PHYSICAL_DAMAGE",
        intent_confidence=0.90,
        retrieval_score=0.70,
        retrieved_evidence=[]
    )
    assert res["decision"] == "ESCALATE"
    assert "HARDWARE_PHYSICAL_DAMAGE" in res["reason"]

def test_escalation_low_confidence():
    engine = EscalationPolicyEngine(intent_threshold=0.40)
    res = engine.evaluate(
        customer_message="Some random text issue",
        intent="OS_UPDATE_BUG",
        intent_confidence=0.20, # < 0.40
        retrieval_score=0.50,
        retrieved_evidence=[]
    )
    assert res["decision"] == "ESCALATE"
    assert "Low intent" in res["reason"]

def test_auto_handle_pass():
    engine = EscalationPolicyEngine()
    res = engine.evaluate(
        customer_message="My Wi-Fi drops on iOS 11",
        intent="CONNECTIVITY_WIFI_BT",
        intent_confidence=0.90,
        retrieval_score=0.50,
        retrieved_evidence=[{"has_dm_escalation": False}]
    )
    assert res["decision"] == "AUTO_HANDLE"
