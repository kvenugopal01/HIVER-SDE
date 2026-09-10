import pytest
import pandas as pd
from src.retriever import HistoricalRetriever

def test_retriever_fit_and_retrieve():
    train_data = pd.DataFrame([
        {
            "conversation_id": "conv_1",
            "cust_text": "My iPhone battery drops from 80% to 10% on iOS 11",
            "brand_text": "@user Let's help get your battery life back on point. Try these steps.",
            "has_dm_escalation": False
        },
        {
            "conversation_id": "conv_2",
            "cust_text": "Wi-Fi disconnecting repeatedly on iPad",
            "brand_text": "@user Please try resetting network settings.",
            "has_dm_escalation": False
        }
    ])
    
    retriever = HistoricalRetriever(top_k=2).fit(train_data)
    res = retriever.retrieve("iPhone battery dying fast after update")
    
    assert "top_similarity_score" in res
    assert "evidence" in res
    assert len(res["evidence"]) == 2
    assert res["evidence"][0]["conversation_id"] == "conv_1"
