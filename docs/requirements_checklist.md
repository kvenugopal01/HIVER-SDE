# PDF & Assignment Requirements Audit Checklist

| Requirement # | PDF / Prompt Requirement | Implemented? | Evidence File / Code Symbol | Verified? |
| :--- | :--- | :---: | :--- | :---: |
| **1** | One selected brand from Twitter dataset | **YES** | `@AppleSupport` in [`src/conversations.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/conversations.py) | **VERIFIED** |
| **2** | Classify incoming messages into small intent taxonomy derived from data | **YES** | 10 intents in [`data/intent_taxonomy.json`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/data/intent_taxonomy.json) & [`src/intent_classifier.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/intent_classifier.py) | **VERIFIED** |
| **3** | Draft response grounded in historical resolutions | **YES** | Grounded retriever in [`src/retriever.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/retriever.py) & [`src/generator.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/generator.py) | **VERIFIED** |
| **4** | Decide AUTO_HANDLE vs ESCALATE with stated reason | **YES** | Policy engine in [`src/escalation.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/escalation.py) | **VERIFIED** |
| **5** | 150–250 hand-reviewed Golden Evaluation Set | **YES** | 200 examples in [`data/golden_eval.csv`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/data/golden_eval.csv) & [`docs/golden_set_methodology.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/docs/golden_set_methodology.md) | **VERIFIED** |
| **6** | Evaluation Harness with automated metrics | **YES** | Accuracy, Macro F1, Confusion Matrix in [`evaluation/metrics.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/evaluation/metrics.py) | **VERIFIED** |
| **7** | LLM-as-Judge for response quality | **YES** | 7-criteria rubric in [`evaluation/rubric.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/evaluation/rubric.md) & [`evaluation/llm_judge.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/evaluation/llm_judge.py) | **VERIFIED** |
| **8** | Evidence of agreement between LLM judge & human ratings | **YES** | Spearman rho & Cohen's Kappa in [`evaluation/agreement.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/evaluation/agreement.py) & [`data/human_eval.csv`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/data/human_eval.csv) | **VERIFIED** |
| **9** | Baseline 1 (Trivial Majority Class) | **YES** | `MajorityClassClassifier` in [`src/intent_classifier.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/intent_classifier.py) | **VERIFIED** |
| **10** | Baseline 2 (Simple TF-IDF + Logistic Regression) | **YES** | `SimpleTfidfLogisticClassifier` in [`src/intent_classifier.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/src/intent_classifier.py) | **VERIFIED** |
| **11** | Results Table comparing baselines vs proposed system | **YES** | Computed in [`evaluation/evaluate.py`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/evaluation/evaluate.py) & shown in [`README.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/README.md) | **VERIFIED** |
| **12** | Top 5 Failure Modes with real examples & hypotheses | **YES** | Detailed analysis in [`README.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/README.md) | **VERIFIED** |
| **13** | Mandatory section: "What is misleading about my headline number?" | **YES** | Explicit section in [`README.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/README.md) | **VERIFIED** |
| **14** | What we would do with one more week | **YES** | Section in [`README.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/README.md) | **VERIFIED** |
| **15** | Decision log of 10–15 non-obvious decisions | **YES** | 12 decisions in [`docs/decision_log.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/docs/decision_log.md) | **VERIFIED** |
| **16** | README instructions reproducing results in <15 mins | **YES** | Quick-start commands in [`README.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/README.md) | **VERIFIED** |
| **17** | Citations section | **YES** | Formatted citations in [`docs/citations.md`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/docs/citations.md) | **VERIFIED** |
| **18** | Comprehensive Test Suite | **YES** | 15 passing tests in [`tests/`](file:///Users/venugopalyadav/Documents/Hiver%20SDE/tests/) (`pytest` passing) | **VERIFIED** |
| **19** | Runnable Live CLI Demo script | **YES** | `python scripts/run_demo.py --message "..."` | **VERIFIED** |
| **20** | Zero Fabricated Data / Labels Rule | **YES** | Hand-labeling CLI tooling in `scripts/label_golden_set.py` & `scripts/label_human_eval.py` | **VERIFIED** |
