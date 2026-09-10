from typing import Dict, Any, List

# Thresholds tuned strictly on Validation set
INTENT_CONFIDENCE_THRESHOLD = 0.40
RETRIEVAL_SIMILARITY_THRESHOLD = 0.20

MANDATORY_ESCALATION_INTENTS = [
    "ACCOUNT_APPLE_ID_ICLOUD",
    "HARDWARE_PHYSICAL_DAMAGE",
    "BILLING_APP_STORE_PURCHASE"
]

SENSITIVE_KEYWORDS = [
    "hacked", "stolen", "lawsuit", "attorney", "court", "legal", 
    "unauthorized charge", "identity theft", "police report"
]

class EscalationPolicyEngine:
    def __init__(
        self,
        intent_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        retrieval_threshold: float = RETRIEVAL_SIMILARITY_THRESHOLD
    ):
        self.intent_threshold = intent_threshold
        self.retrieval_threshold = retrieval_threshold
        
    def evaluate(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        retrieval_score: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        text_lower = customer_message.lower()
        
        # Trigger 1: Sensitive / Legal / Security Keyword
        for kw in SENSITIVE_KEYWORDS:
            if kw in text_lower:
                return {
                    "decision": "ESCALATE",
                    "reason": f"Sensitive keyword detected ('{kw}'). Mandatory escalation required for customer safety."
                }
                
        # Trigger 2: Mandatory High-Risk Intent Category
        if intent in MANDATORY_ESCALATION_INTENTS:
            return {
                "decision": "ESCALATE",
                "reason": f"Intent category '{intent}' involves account credentials, billing, or physical repair."
            }
            
        # Trigger 3: Low Intent Confidence
        if intent_confidence < self.intent_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f} < {self.intent_threshold:.2f})."
            }
            
        # Trigger 4: Low Retrieval Evidence Similarity
        if retrieval_score < self.retrieval_threshold:
            return {
                "decision": "ESCALATE",
                "reason": f"Insufficient historical resolution evidence (similarity {retrieval_score:.2f} < {self.retrieval_threshold:.2f})."
            }
            
        # Trigger 5: Top Historical Resolution Required DM Escalation
        if retrieved_evidence and len(retrieved_evidence) > 0:
            if retrieved_evidence[0].get("has_dm_escalation", False):
                return {
                    "decision": "ESCALATE",
                    "reason": "Top historical resolution for similar issue required private DM verification."
                }
                
        # Default Pass to Auto-Handle
        return {
            "decision": "AUTO_HANDLE",
            "reason": f"High intent confidence ({intent_confidence:.2f}) and grounded historical resolution evidence ({retrieval_score:.2f})."
        }
