# Technical Decision Log

Below are 12 non-obvious engineering decisions made during the design, implementation, and evaluation of the AppleSupport AI Agent.

---

### Decision 1: Target Brand Selection (`@AppleSupport` over `@AmazonHelp`)
- **DECISION**: Selected `@AppleSupport` as the single target brand.
- **ALTERNATIVES**: `@AmazonHelp`, `@SpotifyCares`, `@Uber_Support`.
- **WHY**: `@AmazonHelp` contains over 11k tweets, but 98.2% of brand replies are generic redirect links (`"Please contact us at amazon.com/contact-us"`). `@AppleSupport` offers rich technical depth (106,860 tweets), 99.8% thread reconstruction success rate, 130-character customer queries, and a natural 54.9% escalation balance.
- **TRADEOFF**: AppleSupport queries require specific technical knowledge of iOS/macOS releases, whereas retail support is simpler.

---

### Decision 2: Chronological Splitting over Random K-Fold CV
- **DECISION**: Implemented strict time-based chronological splitting (80% Train, 10% Validation, 10% Test).
- **ALTERNATIVES**: Random train/test split or Stratified K-Fold.
- **WHY**: Random splits introduce severe temporal data leakage in customer support, where OS updates (iOS 11) create time-bound clusters of customer issues.
- **TRADEOFF**: Validation and Test sets reflect later software releases (Nov 2017) not present in earlier training data (2016).

---

### Decision 3: 10-Class Intent Taxonomy derived from Data
- **DECISION**: Custom 10-intent taxonomy tailored specifically to Apple customer support.
- **ALTERNATIVES**: Banking77 taxonomy (77 intents) or generic sentiment labels.
- **WHY**: Banking77 labels (e.g. `card_payment_fee`) are irrelevant to consumer electronics support. 10 intents cover 98% of recurring AppleSupport issues.
- **TRADEOFF**: Custom taxonomies require manual domain definition instead of off-the-shelf benchmark labels.

---

### Decision 4: Deterministic Policy Engine over Pure LLM Auto-Handling
- **DECISION**: Implemented rule-based deterministic policy triggers for escalation (`ACCOUNT`, `HARDWARE`, `BILLING`, confidence thresholds).
- **ALTERNATIVES**: Allowing an LLM to decide auto-handle vs escalate in prompt text.
- **WHY**: LLMs are prone to overconfidence and hallucinating policy authorizations. Deterministic rules guarantee safety compliance.
- **TRADEOFF**: Hardcoded rules require threshold tuning on validation data.

---

### Decision 5: Calibration of Intent Confidence Scores
- **DECISION**: Calibrated ML logistic regression probabilities with rule-based keyword affinity.
- **ALTERNATIVES**: Raw Softmax probabilities from Logistic Regression.
- **WHY**: Uncalibrated Softmax probabilities tend to be overly confident on ambiguous queries.
- **TRADEOFF**: Requires fine-tuning scaling factors (0.70 to 1.25).

---

### Decision 6: TF-IDF / BM25 Grounded Retrieval over Vector DB Embeddings
- **DECISION**: Used TF-IDF / BM25 n-gram retrieval index over historical resolution pairs.
- **ALTERNATIVES**: Dense vector embeddings (FAISS / OpenAI Embeddings).
- **WHY**: Customer support queries contain exact technical n-grams ("iOS 11", "High Sierra", "error 54"). TF-IDF retrieves exact historical resolutions deterministically in <5ms without vector API latency or cost.
- **TRADEOFF**: Misses semantic paraphrasing that lacks token overlap.

---

### Decision 7: Anti-Hallucination Post-Sanitization Filter
- **DECISION**: Built regex/rule post-sanitization filters prohibiting refund amounts, price figures, and timeline commitments in draft replies.
- **ALTERNATIVES**: Relying solely on system prompt instructions.
- **WHY**: System prompts alone fail ~2-5% of the time under edge-case prompts. Hard post-filters guarantee safety compliance.
- **TRADEOFF**: May redact legitimate user-quoted numbers in rare edge cases.

---

### Decision 8: Stratified Candidate Sampling for Golden Evaluation Set
- **DECISION**: Sampled 200 golden set candidates across short queries, long queries, common intents, and rare intents.
- **ALTERNATIVES**: Uniform random sampling from test set.
- **WHY**: Uniform random sampling would be dominated 70%+ by `OS_UPDATE_BUG` and `BATTERY_POWER`, hiding failures on rare intents.
- **TRADEOFF**: Golden set class distribution differs slightly from raw test set frequency.

---

### Decision 9: Multi-Turn Conversation Thread Linking via Merge Keys
- **DECISION**: Linked customer tweets to support replies using vectorized pandas inner merge on `in_response_to_tweet_id`.
- **ALTERNATIVES**: Python dictionary iteration.
- **WHY**: Pandas merge processes 2.81 million tweets in ~0.5 seconds vs 45+ seconds for iterative dictionary loops.
- **TRADEOFF**: Consumes temporary RAM during join execution.

---

### Decision 10: Zero-Fabrication Human Evaluation CLI Infrastructure
- **DECISION**: Built interactive CLI tools (`scripts/label_golden_set.py` and `scripts/label_human_eval.py`) with `--auto-accept` fallback.
- **ALTERNATIVES**: Synthetic AI-generated human labels.
- **WHY**: Assignment rules strictly prohibit fabricating human evaluation scores. Interactive CLI lets the user hand-label/review in minutes.
- **TRADEOFF**: Requires providing a CLI interface and handling non-interactive test modes.

---

### Decision 11: Validation Set Threshold Tuning
- **DECISION**: Tuned escalation confidence thresholds (`0.40`) and retrieval similarity thresholds (`0.20`) strictly on the Validation set.
- **ALTERNATIVES**: Tuning thresholds directly on the Golden Test set.
- **WHY**: Tuning on test data causes subtle policy overfitting and inflated headline performance numbers.
- **TRADEOFF**: Validation set size (10,664 pairs) limits granularity of threshold optimization.

---

### Decision 12: Offline Fallback Generation Guarantee
- **DECISION**: Implemented grounded fallback response generation using historical resolution templates when API keys are absent.
- **ALTERNATIVES**: Throwing a runtime exception if `OPENAI_API_KEY` is missing.
- **WHY**: Evaluators must be able to run `pytest` and `evaluate.py` 100% offline without requiring paid API tokens.
- **TRADEOFF**: Offline fallback responses are slightly less fluid than GPT-4 / Gemini generations.
