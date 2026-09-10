import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def heuristic_keyword_label(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ['ios', 'macos', 'update', 'sierra', 'ios11', 'bug', 'crash', 'apple logo']):
        return "OS_UPDATE_BUG"
    elif any(k in text_lower for k in ['battery', 'charge', 'drain', 'dying', 'overheat', 'percent']):
        return "BATTERY_POWER"
    elif any(k in text_lower for k in ['wifi', 'wi-fi', 'bluetooth', 'airpods', 'carplay', 'cellular', 'no service', 'signal']):
        return "CONNECTIVITY_WIFI_BT"
    elif any(k in text_lower for k in ['apple id', 'icloud', 'password', 'locked', '2fa', 'verification', 'security']):
        return "ACCOUNT_APPLE_ID_ICLOUD"
    elif any(k in text_lower for k in ['screen', 'cracked', 'shattered', 'water', 'speaker', 'mic', 'hardware', 'repair', 'applecare']):
        return "HARDWARE_PHYSICAL_DAMAGE"
    elif any(k in text_lower for k in ['music', 'itunes', 'audio', 'playback', 'media key', 'song', 'volume']):
        return "AUDIO_MEDIA_PLAYBACK"
    elif any(k in text_lower for k in ['app store', 'refund', 'charge', 'subscription', 'billing', 'payment', 'receipt']):
        return "BILLING_APP_STORE_PURCHASE"
    elif any(k in text_lower for k in ['backup', 'restore', 'sync', 'data transfer']):
        return "SYNC_BACKUP_RESTORE"
    elif any(k in text_lower for k in ['slow', 'lag', 'unresponsive', 'keyboard delay', 'freeze']):
        return "PERFORMANCE_LAG"
    elif any(k in text_lower for k in ['store', 'trade in', 'release', 'hours', 'location']):
        return "GENERAL_ENQUIRY_OTHER"
    return "OS_UPDATE_BUG"

# -------------------------------------------------------------
# BASELINE 1: Trivial Majority Class Classifier
# -------------------------------------------------------------
class MajorityClassClassifier:
    def __init__(self):
        self.majority_class = "OS_UPDATE_BUG"
        
    def fit(self, X: List[str], y: List[str]):
        if len(y) > 0:
            vals, counts = np.unique(y, return_counts=True)
            self.majority_class = vals[np.argmax(counts)]
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        return [self.majority_class] * len(X)
        
    def predict_proba(self, X: List[str]) -> List[float]:
        return [1.0] * len(X)

# -------------------------------------------------------------
# BASELINE 2: Simple TF-IDF + Logistic Regression
# -------------------------------------------------------------
class SimpleTfidfLogisticClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
        self.clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
        self.classes_ = []
        
    def fit(self, X: List[str], y: List[str]):
        X_vec = self.vectorizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        self.classes_ = list(self.clf.classes_)
        return self
        
    def predict(self, X: List[str]) -> List[str]:
        X_vec = self.vectorizer.transform(X)
        return self.clf.predict(X_vec).tolist()
        
    def predict_proba(self, X: List[str]) -> List[Tuple[str, float]]:
        X_vec = self.vectorizer.transform(X)
        probs = self.clf.predict_proba(X_vec)
        results = []
        for row in probs:
            best_idx = int(np.argmax(row))
            results.append((self.classes_[best_idx], float(row[best_idx])))
        return results

# -------------------------------------------------------------
# PROPOSED MODEL: Calibrated Hybrid Classifier
# -------------------------------------------------------------
class CalibratedHybridClassifier:
    def __init__(self):
        self.base_model = SimpleTfidfLogisticClassifier()
        
    def fit(self, X: List[str], y: List[str]):
        self.base_model.fit(X, y)
        return self
        
    def predict_single(self, text: str) -> Dict[str, Any]:
        # Get ML model prediction & probability
        pred_label, raw_prob = self.base_model.predict_proba([text])[0]
        
        # Check rule-based heuristic match for double confidence
        rule_label = heuristic_keyword_label(text)
        
        if pred_label == rule_label:
            calibrated_conf = min(1.0, raw_prob * 1.25)
        else:
            # Conflict between ML model and rule heuristic -> reduce confidence
            calibrated_conf = raw_prob * 0.70
            
        return {
            "intent": pred_label,
            "intent_confidence": round(float(calibrated_conf), 3),
            "heuristic_match": rule_label
        }
        
    def predict(self, X: List[str]) -> List[str]:
        return [self.predict_single(x)["intent"] for x in X]

def train_and_save_intent_models(
    train_path: str = "data/processed/historical_train.parquet",
    output_dir: str = "data/processed"
):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} missing. Run leakage_guard.py first.")
        
    train_df = pd.read_parquet(train_path)
    X_train = train_df['cust_text'].tolist()
    
    # Assign pseudo-labels using keyword heuristic for historical training set
    y_train = [heuristic_keyword_label(x) for x in X_train]
    
    print(f"Training intent classifiers on {len(X_train)} historical queries...")
    
    # Fit Baseline 1
    b1 = MajorityClassClassifier().fit(X_train, y_train)
    
    # Fit Baseline 2
    b2 = SimpleTfidfLogisticClassifier().fit(X_train, y_train)
    
    # Fit Proposed Hybrid
    proposed = CalibratedHybridClassifier().fit(X_train, y_train)
    
    with open(os.path.join(output_dir, "intent_baseline1.pkl"), "wb") as f:
        pickle.dump(b1, f)
    with open(os.path.join(output_dir, "intent_baseline2.pkl"), "wb") as f:
        pickle.dump(b2, f)
    with open(os.path.join(output_dir, "intent_proposed.pkl"), "wb") as f:
        pickle.dump(proposed, f)
        
    print("Intent classifiers saved cleanly in data/processed/")

if __name__ == "__main__":
    train_and_save_intent_models()
