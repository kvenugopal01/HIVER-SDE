import re
from typing import Dict, Any, List

def evaluate_response_judge(
    customer_message: str,
    intent: str,
    retrieved_evidence: List[Dict[str, Any]],
    draft_reply: str,
    decision: str,
    reason: str,
    expected_decision: str = "AUTO_HANDLE"
) -> Dict[str, Any]:
    """
    Evaluates generated reply on 1-5 scale across 7 rubric dimensions.
    """
    msg_lower = customer_message.lower()
    reply_lower = draft_reply.lower()
    
    # 1. Groundedness check
    top_res = retrieved_evidence[0].get("historical_resolution", "").lower() if retrieved_evidence else ""
    if top_res and any(w in reply_lower for w in top_res.split()[:5]):
        groundedness = 5.0
    elif len(retrieved_evidence) > 0:
        groundedness = 4.0
    else:
        groundedness = 3.0
        
    # 2. Unsupported Claims (Hallucination check)
    has_prohibited = any(kw in reply_lower for kw in ['refund', '$', 'dollar', 'guarantee', 'free repair'])
    unsupported_claims = 1.0 if has_prohibited else 5.0
    
    # 3. Tone
    has_polite = any(kw in reply_lower for kw in ['help', 'please', 'thanks', 'dm', 'support', 'certainly'])
    tone = 5.0 if has_polite else 4.0
    
    # 4. Relevance
    relevance = 5.0 if len(draft_reply) >= 15 else 2.0
    
    # 5. Correctness
    correctness = 4.5 if unsupported_claims == 5.0 else 2.0
    
    # 6. Helpfulness
    helpfulness = 4.5 if (decision == "AUTO_HANDLE" and len(draft_reply) > 20) or (decision == "ESCALATE" and "dm" in reply_lower) else 3.5
    
    # 7. Escalation Appropriateness
    if decision == expected_decision:
        escalation_app = 5.0
    else:
        escalation_app = 2.5
        
    scores = {
        "relevance": relevance,
        "correctness": correctness,
        "groundedness": groundedness,
        "helpfulness": helpfulness,
        "tone": tone,
        "unsupported_claims": unsupported_claims,
        "escalation_appropriateness": escalation_app
    }
    
    overall = round(sum(scores.values()) / len(scores), 2)
    scores["overall_score"] = overall
    return scores
