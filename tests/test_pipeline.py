import pytest
from src.pipeline import AppleSupportAgentPipeline

def test_pipeline_output_schema():
    pipeline = AppleSupportAgentPipeline()
    res = pipeline.process_message("My keyboard is lagging on iOS 11")
    
    assert "customer_message" in res
    assert "intent" in res
    assert "intent_confidence" in res
    assert "retrieved_evidence" in res
    assert "draft_reply" in res
    assert "decision" in res
    assert "reason" in res
    
    assert res["decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(res["intent_confidence"], float)
    assert isinstance(res["retrieved_evidence"], list)
