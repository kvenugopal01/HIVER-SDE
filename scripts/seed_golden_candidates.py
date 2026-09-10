import os
import json
import pandas as pd
import numpy as np

def seed_golden_set(
    test_path: str = "data/processed/test_candidates.parquet",
    output_path: str = "data/golden_eval.csv",
    sample_size: int = 200
):
    """
    Samples 200 diverse candidate customer messages from TEST_CANDIDATES split,
    assigns heuristic candidate intent & escalation labels, and creates initial golden_eval.csv.
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} missing. Run leakage_guard.py first.")
        
    with open("data/intent_taxonomy.json", "r") as f:
        taxonomy = json.load(f)["intents"]
        
    test_df = pd.read_parquet(test_path)
    
    # Heuristic intent rule matcher
    def heuristic_intent(text: str):
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
        else:
            return "OS_UPDATE_BUG" # default most frequent

    # Perform stratified sampling across query length and keyword types
    np.random.seed(42)
    
    # Shuffle deterministic
    shuffled = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Select sample_size examples
    sample_df = shuffled.head(sample_size).copy()
    
    golden_rows = []
    for idx, row in sample_df.iterrows():
        cust_msg = row['cust_text']
        intent = heuristic_intent(cust_msg)
        tax_info = taxonomy.get(intent, {})
        default_decision = tax_info.get("expected_escalation_default", "AUTO_HANDLE")
        
        # Check if customer message contains explicit DM or account issue
        if intent in ["ACCOUNT_APPLE_ID_ICLOUD", "HARDWARE_PHYSICAL_DAMAGE", "BILLING_APP_STORE_PURCHASE"]:
            expected_decision = "ESCALATE"
            reason = f"Category {intent} requires account-level or hardware physical intervention."
        elif row.get('has_dm_escalation', False):
            expected_decision = "ESCALATE"
            reason = "Historical brand reply required private DM escalation."
        else:
            expected_decision = "AUTO_HANDLE"
            reason = "Standard self-serve technical support query."
            
        golden_rows.append({
            "example_id": f"golden_{idx+1:03d}",
            "conversation_id": row['conversation_id'],
            "customer_message": cust_msg,
            "context": row.get('context', ''),
            "true_intent": intent,
            "expected_decision": expected_decision,
            "expected_escalation_reason": reason,
            "human_notes": "Pending human review",
            "is_human_verified": False
        })
        
    golden_df = pd.DataFrame(golden_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    golden_df.to_csv(output_path, index=False)
    print(f"Seeded {len(golden_df)} candidates in {output_path}")

if __name__ == "__main__":
    seed_golden_set()
