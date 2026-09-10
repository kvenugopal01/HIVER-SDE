import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import AppleSupportAgentPipeline
from evaluation.llm_judge import evaluate_response_judge

def create_human_eval_dataset(
    golden_path: str = "data/golden_eval.csv",
    output_path: str = "data/human_eval.csv",
    sample_size: int = 30,
    auto_accept: bool = False
):
    if not os.path.exists(golden_path):
        from scripts.seed_golden_candidates import seed_golden_set
        seed_golden_set(output_path=golden_path)
        
    golden_df = pd.read_csv(golden_path).head(sample_size)
    pipeline = AppleSupportAgentPipeline()
    
    rows = []
    print("\nGenerating agent responses for human evaluation subset...")
    for idx, row in golden_df.iterrows():
        cust_msg = row['customer_message']
        res = pipeline.process_message(cust_msg)
        
        # Compute LLM judge scores
        judge_scores = evaluate_response_judge(
            customer_message=cust_msg,
            intent=res['intent'],
            retrieved_evidence=res['retrieved_evidence'],
            draft_reply=res['draft_reply'],
            decision=res['decision'],
            reason=res['reason'],
            expected_decision=row['expected_decision']
        )
        
        human_score = judge_scores['overall_score'] if auto_accept else None
        
        rows.append({
            "example_id": row['example_id'],
            "customer_message": cust_msg,
            "agent_intent": res['intent'],
            "agent_decision": res['decision'],
            "agent_draft_reply": res['draft_reply'],
            "agent_reason": res['reason'],
            "human_expected_decision": row['expected_decision'],
            "judge_overall_score": judge_scores['overall_score'],
            "human_overall_score": human_score,
            "human_notes": "Auto-accepted" if auto_accept else "Pending human rating"
        })
        
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(eval_df)} human evaluation candidate items in {output_path}")
    print("THIS STEP REQUIRES YOUR HUMAN INPUT.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human Evaluation Rating CLI")
    parser.add_argument("--auto-accept", action="store_true", help="Auto-accept generated judge scores as human scores")
    args = parser.parse_args()
    create_human_eval_dataset(auto_accept=args.auto_accept)
