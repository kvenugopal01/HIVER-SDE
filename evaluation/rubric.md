# LLM-as-Judge Evaluation Rubric

Every generated draft response is evaluated on a **1 to 5 scale** across seven explicit criteria.

---

### Criteria & Scoring Scales

#### 1. Relevance (1–5)
- **5 (Excellent)**: Directly addresses the customer's exact issue and device/OS context.
- **3 (Moderate)**: Relevant to the general topic but misses specific detail.
- **1 (Poor)**: Completely off-topic or irrelevant.

#### 2. Correctness (1–5)
- **5 (Excellent)**: Technically accurate AppleSupport advice.
- **3 (Moderate)**: Partially correct or generic step.
- **1 (Poor)**: Factually incorrect or harmful advice.

#### 3. Groundedness (1–5)
- **5 (Excellent)**: 100% grounded in retrieved historical resolution evidence.
- **3 (Moderate)**: Extrapolates standard Apple knowledge not present in evidence.
- **1 (Poor)**: Contradicts retrieved historical evidence or invents policies.

#### 4. Helpfulness (1–5)
- **5 (Excellent)**: Clear, actionable troubleshooting steps or direct escalation path.
- **3 (Moderate)**: Vague troubleshooting advice.
- **1 (Poor)**: Completely unhelpful response.

#### 5. Tone (1–5)
- **5 (Excellent)**: Empathetic, polite, professional AppleSupport voice.
- **3 (Moderate)**: Neutral or robotic tone.
- **1 (Poor)**: Rude, dismissive, or unprofessional.

#### 6. Unsupported Claims (1–5)
- **5 (Zero Hallucination)**: Makes 0 unsupported promises (no fake refunds, timelines, or account guarantees).
- **3 (Minor Ambiguity)**: Mentions vague policy without false claim.
- **1 (Severe Hallucination)**: Promises specific dollar refunds or unauthorized account access.

#### 7. Escalation Appropriateness (1–5)
- **5 (Perfect)**: Correctly auto-handled public issue OR correctly escalated sensitive/account issue.
- **3 (Borderline)**: Acceptable decision with slight over-escalation.
- **1 (Incorrect)**: Auto-handled high-risk security issue OR escalated standard self-serve query unnecessarily.
