import os
import pickle
import json
import pandas as pd
from typing import Dict, Any

from src.intent_classifier import CalibratedHybridClassifier
from src.retriever import HistoricalRetriever
from src.escalation import EscalationPolicyEngine
from src.generator import ResponseGenerator

class AppleSupportAgentPipeline:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.intent_clf = None
        self.retriever = None
        self.escalation_engine = EscalationPolicyEngine()
        self.generator = ResponseGenerator()
        self._load_artifacts()
        
    def _load_artifacts(self):
        intent_path = os.path.join(self.processed_dir, "intent_proposed.pkl")
        retriever_path = os.path.join(self.processed_dir, "retriever.pkl")
        
        if not os.path.exists(intent_path) or not os.path.exists(retriever_path):
            print("Artifacts missing. Training and building indices now...")
            from src.intent_classifier import train_and_save_intent_models
            from src.retriever import build_and_save_retriever
            train_and_save_intent_models(output_dir=self.processed_dir)
            build_and_save_retriever(output_path=retriever_path)
            
        with open(intent_path, "rb") as f:
            self.intent_clf = pickle.load(f)
        with open(retriever_path, "rb") as f:
            self.retriever = pickle.load(f)
            
    def process_message(self, message: str) -> Dict[str, Any]:
        # 1. Classify Intent
        intent_res = self.intent_clf.predict_single(message)
        intent = intent_res["intent"]
        confidence = intent_res["intent_confidence"]
        
        # 2. Retrieve Historical Resolution Evidence
        retrieval_res = self.retriever.retrieve(message)
        top_similarity = retrieval_res["top_similarity_score"]
        evidence = retrieval_res["evidence"]
        
        # 3. Evaluate Escalation Policy
        esc_res = self.escalation_engine.evaluate(
            customer_message=message,
            intent=intent,
            intent_confidence=confidence,
            retrieval_score=top_similarity,
            retrieved_evidence=evidence
        )
        decision = esc_res["decision"]
        reason = esc_res["reason"]
        
        # 4. Draft Grounded Response
        draft_reply = self.generator.generate_reply(
            customer_message=message,
            intent=intent,
            retrieved_evidence=evidence,
            decision=decision,
            reason=reason
        )
        
        return {
            "customer_message": message,
            "intent": intent,
            "intent_confidence": confidence,
            "retrieved_evidence": evidence,
            "draft_reply": draft_reply,
            "decision": decision,
            "reason": reason
        }

if __name__ == "__main__":
    pipeline = AppleSupportAgentPipeline()
    res = pipeline.process_message("My phone froze on the Apple logo after updating to iOS 11.")
    print(json.dumps(res, indent=2))
