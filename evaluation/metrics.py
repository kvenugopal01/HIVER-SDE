import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    unique_labels = sorted(list(set(y_true).union(set(y_pred))))
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_true, y_pred, labels=unique_labels, zero_division=0)
    
    per_intent_stats = {}
    for i, label in enumerate(unique_labels):
        per_intent_stats[label] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i])
        }
        
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "per_intent": per_intent_stats,
        "labels": unique_labels,
        "confusion_matrix": cm.tolist()
    }

def compute_escalation_metrics(expected_decisions: List[str], actual_decisions: List[str]) -> Dict[str, Any]:
    total = len(expected_decisions)
    if total == 0:
        return {}
        
    auto_count = sum(1 for d in actual_decisions if d == "AUTO_HANDLE")
    auto_pct = round(auto_count / total * 100, 2)
    
    # Binary conversion: ESCALATE = 1, AUTO_HANDLE = 0
    y_true = [1 if d == "ESCALATE" else 0 for d in expected_decisions]
    y_pred = [1 if d == "ESCALATE" else 0 for d in actual_decisions]
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        "auto_handle_pct": auto_pct,
        "escalation_precision": round(float(prec), 4),
        "escalation_recall": round(float(rec), 4),
        "escalation_f1": round(float(f1), 4)
    }
