# Golden Evaluation Set Methodology

## Overview

The Golden Evaluation Set (`data/golden_eval.csv`) consists of 200 hand-reviewed customer support queries originating from real customer interactions with `@AppleSupport` on Twitter.

This dataset serves as the benchmark for evaluating intent classification, response retrieval, groundedness, and escalation policy decisions.

---

## 1. Sampling Strategy

To ensure comprehensive coverage of production edge cases, candidate conversations were sampled strictly from the **`TEST_CANDIDATES`** split (latest 10% chronologically: Nov 26, 2017 to Dec 3, 2017).

The 200 examples were sampled across five key strata:

| Stratum | Target % | Rationale / Objective |
| :--- | :--- | :--- |
| **Common Intents** (`OS_UPDATE_BUG`, `BATTERY_POWER`) | 45% | High-volume recurring customer issues |
| **Secondary Technical Intents** (`CONNECTIVITY`, `AUDIO`, `SYNC`) | 25% | Moderate-volume hardware/software features |
| **High-Risk Account & Billing Intents** (`ACCOUNT`, `BILLING`, `HARDWARE`) | 20% | High-risk inquiries requiring mandatory human escalation |
| **Ambiguous & Multi-Issue Queries** | 5% | Messages mentioning multiple problems (e.g. battery + Wi-Fi) |
| **Short & Noisy Queries** | 5% | Ultra-concise tweets, typos, or informal slang |

---

## 2. Labeling Procedure & Guidelines

Each candidate example underwent review using `scripts/label_golden_set.py`.

The reviewer assigned:
1. **`true_intent`**: Primary intent from the 10-class taxonomy.
2. **`expected_decision`**: `AUTO_HANDLE` vs `ESCALATE`.
3. **`expected_escalation_reason`**: Explicit human-readable reason for escalation.

### Decision Policy Guidelines:
- **`AUTO_HANDLE`**: Query describes standard technical issue (iOS update bug, battery drain, Wi-Fi disconnect) where public troubleshooting steps exist in AppleSupport knowledge base.
- **`ESCALATE`**: Query involves locked Apple ID/password reset, unauthorized App Store charges/refunds, physical screen damage/hardware repair, or explicitly requires private DM/account verification.

---

## 3. Data Leakage Prevention

- **Strict Boundary**: All 200 golden examples originate strictly from dates after Nov 26, 2017.
- **Zero Train Overlap**: Automated checks in `src/leakage_guard.py` verify that 0% of golden set conversation IDs or tweet IDs exist in `HISTORICAL_TRAIN` or `VALIDATION` splits.
- **Index Isolation**: The historical retrieval engine (`src/retriever.py`) and TF-IDF classifiers are fitted **exclusively** on `HISTORICAL_TRAIN`.
