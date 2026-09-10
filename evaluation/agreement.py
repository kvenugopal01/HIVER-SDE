import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

def compute_judge_human_agreement(human_eval_path: str = "data/human_eval.csv") -> Dict[str, Any]:
    """
    Calculates Spearman correlation and Cohen's Kappa between human ratings and LLM judge ratings.
    Returns status PENDING if human evaluations have not been completed.
    """
    if not os.path.exists(human_eval_path):
        return {
            "status": "PENDING",
            "message": "THIS STEP REQUIRES YOUR HUMAN INPUT. data/human_eval.csv not found.",
            "spearman_correlation": None,
            "cohens_kappa": None
        }
        
    df = pd.read_csv(human_eval_path)
    
    # Check if human ratings exist
    if 'human_overall_score' not in df.columns or df['human_overall_score'].isnull().all():
        return {
            "status": "PENDING",
            "message": "THIS STEP REQUIRES YOUR HUMAN INPUT. data/human_eval.csv has no human scores.",
            "spearman_correlation": None,
            "cohens_kappa": None
        }
        
    valid_df = df.dropna(subset=['human_overall_score', 'judge_overall_score']).copy()
    if len(valid_df) < 5:
        return {
            "status": "PENDING",
            "message": f"THIS STEP REQUIRES YOUR HUMAN INPUT. Insufficient rated samples ({len(valid_df)} < 5).",
            "spearman_correlation": None,
            "cohens_kappa": None
        }
        
    # Spearman rank correlation
    rho, pval = spearmanr(valid_df['human_overall_score'], valid_df['judge_overall_score'])
    
    # Cohen's Kappa on categorical escalation decision
    human_dec = valid_df['human_expected_decision'].astype(str)
    judge_dec = valid_df['agent_decision'].astype(str)
    kappa = cohen_kappa_score(human_dec, judge_dec)
    
    return {
        "status": "COMPLETED",
        "sample_size": len(valid_df),
        "spearman_correlation": round(float(rho), 4),
        "spearman_p_value": round(float(pval), 4),
        "cohens_kappa": round(float(kappa), 4)
    }

if __name__ == "__main__":
    res = compute_judge_human_agreement()
    print(res)
