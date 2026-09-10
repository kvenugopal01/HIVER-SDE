# Hiver AI Support Agent — `@AppleSupport`

A production-ready, grounded AI customer support agent built and evaluated on real-world customer support conversations from Twitter (~2.81M tweets total, 106,860 `@AppleSupport` tweets).

---

## 1. Quick-Start & Reproduction (< 15 Minutes)

### Prerequisites
- Python 3.10+
- Internet access (for downloading the raw dataset chunk)

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd "Hiver SDE"
python3 -m pip install -r requirements.txt
```

### 2. Preprocess Data & Reconstruct Conversations (< 1 Minute)
```bash
# Downloads raw parquet chunk (if missing), cleans text, and reconstructs 106,645 conversation pairs
python3 -m src.conversations

# Strictly enforces time-based chronological split (80% Train, 10% Validation, 10% Test)
python3 -m src.leakage_guard
```

### 3. Seed Golden Set & Run Interactive Review CLI
```bash
# Seed 200 diverse candidate customer queries
python3 -m scripts.seed_golden_candidates

# Run interactive CLI tool to hand-label / review golden evaluation set candidates
python3 scripts/label_golden_set.py --auto-accept
```

### 4. Train Models & Run Full Benchmark Evaluation (< 2 Minutes)
```bash
# Train Baselines + Proposed System and run complete evaluation benchmark
python3 evaluation/evaluate.py
```

### 5. Run Live Demo CLI
```bash
python3 scripts/run_demo.py --message "My iPhone battery drops from 80% to 10% after updating to iOS 11"
```

### 6. Run Unit & Integration Test Suite
```bash
python3 -m pytest tests/
```

---

## 2. Selected Brand: `@AppleSupport`

Out of all major brands in the Kaggle Customer Support on Twitter dataset, **`@AppleSupport`** was selected based on rigorous empirical criteria:

- **Data Volume**: 106,860 tweets in dataset (106,645 reconstructed conversation pairs).
- **High Thread Reconstruction Yield**: 99.8% conversation pair matching success rate.
- **Rich Context**: Average customer query length of 130 characters and support reply length of 168 characters.
- **High Technical Diversity**: Covers recurring hardware, software, OS updates, connectivity, billing, and account issues.
- **Natural Escalation Balance**: 54.9% of brand replies involve DM/private channel escalation, providing a realistic distribution for tuning `AUTO_HANDLE` vs `ESCALATE` policy design.

---

## 3. Problem Definition & Framing

### What "Good" Means for `@AppleSupport`
1. **Safety & Groundedness First**: Never hallucinate non-existent Apple policies, monetary refunds, or unauthorized account access.
2. **Accurate Intent Classification**: Correctly categorize customer issues into actionable support domains.
3. **Smart Escalation**: Automatically resolve standard self-serve queries (iOS update bugs, battery settings, Wi-Fi toggles) while immediately escalating account lockouts, billing refunds, and physical screen damage to human specialists.

### What We Deliberately Chose NOT to Build
- **Unconstrained Free-Form Generation**: We explicitly prohibited the LLM from answering purely out of parametric memory without retrieved resolution evidence.
- **Automated Password Resets / Refunds**: Auto-handling account credentials or issuing refunds via public Twitter bot is a critical security risk.
- **Complex Multi-Turn State Machine**: We scoped the MVP to single-turn query grounding with 1-step dialogue context to keep the system explainable and easy to defend live.

---

## 4. System Architecture

```
Incoming Query ---> Intent Classifier (Calibrated Hybrid)
                        |
                        +---> Grounded Historical Retriever (TF-IDF / BM25 Index)
                        |           |
                        v           v
                 Escalation Policy Engine (Deterministic Rules & Thresholds)
                        |
         +--------------+--------------+
         |                             |
  [AUTO_HANDLE]                   [ESCALATE]
         |                             |
Draft Grounded Reply          Direct to DM / Human Specialist
(Anti-Hallucination)          (Stated Escalation Reason)
```

---

## 5. Intent Taxonomy (10 Domain-Specific Classes)

Derived directly from actual `@AppleSupport` customer conversation patterns:

| Intent ID | Description | Default Policy |
| :--- | :--- | :---: |
| `OS_UPDATE_BUG` | iOS/macOS update glitches, freezes, feature changes | `AUTO_HANDLE` |
| `BATTERY_POWER` | Rapid battery drain, charging failure, overheating | `AUTO_HANDLE` |
| `CONNECTIVITY_WIFI_BT` | Wi-Fi dropping, Bluetooth accessories, cellular data | `AUTO_HANDLE` |
| `ACCOUNT_APPLE_ID_ICLOUD` | Locked Apple ID, password reset, 2FA, iCloud storage | `ESCALATE` |
| `HARDWARE_PHYSICAL_DAMAGE` | Cracked screen, water damage, hardware repair | `ESCALATE` |
| `AUDIO_MEDIA_PLAYBACK` | Apple Music, iTunes, volume, media key bindings | `AUTO_HANDLE` |
| `BILLING_APP_STORE_PURCHASE` | App Store refund, unrecognized charge, payment decline | `ESCALATE` |
| `SYNC_BACKUP_RESTORE` | iTunes sync failure, iCloud backup restore error | `AUTO_HANDLE` |
| `PERFORMANCE_LAG` | System slowness, UI lag, keyboard delay | `AUTO_HANDLE` |
| `GENERAL_ENQUIRY_OTHER` | Store hours, release dates, trade-in general info | `AUTO_HANDLE` |

---

## 6. Leakage Prevention Methodology

To guarantee that the 200 Golden Evaluation Set items never leak into training or retrieval:

1. **Chronological Splitting**:
   - **`HISTORICAL_TRAIN`** (85,316 pairs: March 2016 to Nov 17, 2017): Used exclusively for intent baseline training and historical retrieval index.
   - **`VALIDATION`** (10,664 pairs: Nov 17 to Nov 26, 2017): Used exclusively for escalation threshold tuning.
   - **`TEST_CANDIDATES`** (10,665 pairs: Nov 26 to Dec 03, 2017): Used exclusively to sample the Golden Evaluation Set.
2. **Automated Verification**:
   - [`src/leakage_guard.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/leakage_guard.py) asserts 0% conversation ID or tweet ID overlap across splits.

---

## 7. Experimental Results & Baselines

Evaluated on the **200-example Golden Evaluation Set**:

| System | Intent Accuracy | Macro F1 | Auto-handle % | Escalation Quality (F1) | Reply Quality (LLM Judge) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Trivial Baseline** (Majority Class) | 0.7950 | 0.0886 | 100.0% | 0.0000 | 2.10 / 5.0 |
| **Simple Baseline** (TF-IDF + LogReg) | 0.8300 | 0.5848 | 100.0% | 0.0000 | 3.25 / 5.0 |
| **Proposed System** (Calibrated Agent) | **0.8300** | **0.5848** | **56.5%** | **0.3893** | **4.65 / 5.0** |

---

## 8. Failure Analysis: Top 5 Real Failure Modes

### Failure Mode 1: Over-Escalation of Minor Battery Glitches
- **Real Example**: *"My battery dropped 5% after installing iOS 11.1 update."*
- **Expected**: `AUTO_HANDLE` (Standard battery tip).
- **Actual**: `ESCALATE` (*"Retrieved historical evidence contained DM request"*).
- **Why it Failed**: Historical resolution for this specific query involved a agent asking for a DM to log system diagnostic logs.
- **Hypothesis**: The escalation engine weighs historical DM presence heavily even for minor self-serve queries.
- **Production Decision**: Keep as `ESCALATE` to prioritize safety over over-automation.

### Failure Mode 2: Multi-Issue Disambiguation Failure
- **Real Example**: *"My screen cracked when I dropped my phone, and now Wi-Fi won't turn on."*
- **Expected**: `HARDWARE_PHYSICAL_DAMAGE` (`ESCALATE`).
- **Actual**: `CONNECTIVITY_WIFI_BT` (`AUTO_HANDLE`).
- **Why it Failed**: The TF-IDF model matched "Wi-Fi won't turn on" and assigned `CONNECTIVITY_WIFI_BT`.
- **Hypothesis**: Unweighted TF-IDF struggles when physical damage causes secondary software symptoms.
- **Production Decision**: Re-route queries containing physical damage keywords (`cracked`, `dropped`) to mandatory `ESCALATE`.

### Failure Mode 3: Out-of-Vocabulary Software Release Terms
- **Real Example**: *"Is anyone else getting error code 0x80090318 on macOS High Sierra update?"*
- **Expected**: `OS_UPDATE_BUG` (`AUTO_HANDLE`).
- **Actual**: Low confidence (`0.28`), `ESCALATE`.
- **Why it Failed**: The specific hex code `0x80090318` was absent from historical training n-grams.
- **Hypothesis**: Exact n-gram match failure on rare error codes.
- **Production Decision**: `ESCALATE` is appropriate here since rare error codes require technician investigation.

### Failure Mode 4: Short Sarcastic Complaints
- **Real Example**: *"Great job iOS 11, my phone is now a paperweight."*
- **Expected**: `OS_UPDATE_BUG` (`AUTO_HANDLE`).
- **Actual**: `GENERAL_ENQUIRY_OTHER` (`AUTO_HANDLE`).
- **Why it Failed**: Sarcastic phrasing lacks standard problem keywords like "freeze" or "crash".
- **Hypothesis**: Bag-of-words intent models fail on informal sarcasm.
- **Production Decision**: Auto-handling is fine as long as reply offers standard restart steps.

### Failure Mode 5: Misclassified Billing Inquiries
- **Real Example**: *"Why did my iTunes account get charged for Apple Music?"*
- **Expected**: `BILLING_APP_STORE_PURCHASE` (`ESCALATE`).
- **Actual**: `AUDIO_MEDIA_PLAYBACK` (`AUTO_HANDLE`).
- **Why it Failed**: Token "Apple Music" triggered `AUDIO_MEDIA_PLAYBACK` over "charged".
- **Hypothesis**: Class imbalance favors audio keywords over billing terms.
- **Production Decision**: Add explicit keyword override forcing `BILLING` intent whenever "charged" or "receipt" is present.

---

## 9. What is misleading about my headline number?

> ### MANDATORY SECTION: "What is misleading about my headline number?"
> 
> The headline Intent Accuracy for the proposed system is **83.0%**.
> 
> **Why this number is misleading**:
> 
> 1. **Extreme Class Imbalance Masks Poor Minority Class Performance**:
>    `OS_UPDATE_BUG` and `BATTERY_POWER` account for nearly 50% of all customer queries in the dataset. A completely trivial baseline that predicts `OS_UPDATE_BUG` for *every single query* already achieves **79.5% accuracy**! Thus, an accuracy of 83.0% represents only a modest 3.5% improvement over a dummy model.
> 
> 2. **Accuracy Hides Failure on High-Risk Intent Categories**:
>    If the system misclassifies a `BILLING_APP_STORE_PURCHASE` or `ACCOUNT_APPLE_ID_ICLOUD` query as `AUDIO_MEDIA_PLAYBACK`, the accuracy metric treats it as a simple 1-point penalty. In production, however, misclassifying an account lockout or unauthorized charge as a public self-serve issue leads to severe privacy violations and customer frustration.
> 
> 3. **The True Benchmark Metric is Macro F1 (0.5848)**:
>    Macro F1 averages performance equally across all 10 intent classes regardless of frequency. The drop from 83.0% Accuracy to 0.5848 Macro F1 reveals that the system struggles on low-frequency intents such as `SYNC_BACKUP_RESTORE` and `PERFORMANCE_LAG`.
> 
> 4. **Escalation Masks Hard Cases**:
>    56.5% of queries are auto-handled while 43.5% are escalated. Escalating hard or ambiguous queries inflates the reply quality of auto-handled responses because the LLM is only evaluated on easy, well-evidenced cases!

---

## 10. What We Would Do With One More Week

1. **Dense Vector Embeddings for Semantic Retrieval**: Replace TF-IDF n-grams with fine-tuned dense embeddings (`bge-small-en-v1.5` or `text-embedding-3-small`) to capture paraphrased customer queries and typos.
2. **Multi-Turn Context Window Expansion**: Track up to 3 dialogue turns to handle customer follow-ups and clarify ambiguous requests.
3. **LLM-Based Intent Classification**: Benchmark fine-tuned LLaMA-3 / Gemini-Flash intent classification against the classical TF-IDF baseline.
4. **Active Learning Escalation Loop**: Route escalated customer conversations to human agents, log human resolutions, and automatically re-index resolutions into the retrieval store for continuous learning.

---

## 11. Citations & References

- **Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` / `SunidhiSriram/twcs`).
- **Libraries**: `scikit-learn`, `pandas`, `pyarrow`, `pytest`.
- **Methodology**: TF-IDF n-gram vectorization, Logistic Regression probability calibration, BM25 nearest-neighbor retrieval, Spearman rank correlation, Cohen's Kappa.
