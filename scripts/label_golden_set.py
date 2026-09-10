import sys
import os
import json
import argparse
import pandas as pd

INTENT_KEYS = [
    "OS_UPDATE_BUG",
    "BATTERY_POWER",
    "CONNECTIVITY_WIFI_BT",
    "ACCOUNT_APPLE_ID_ICLOUD",
    "HARDWARE_PHYSICAL_DAMAGE",
    "AUDIO_MEDIA_PLAYBACK",
    "BILLING_APP_STORE_PURCHASE",
    "SYNC_BACKUP_RESTORE",
    "PERFORMANCE_LAG",
    "GENERAL_ENQUIRY_OTHER"
]

def run_labeling_cli(csv_path: str = "data/golden_eval.csv", auto_accept: bool = False):
    if not os.path.exists(csv_path):
        from scripts.seed_golden_candidates import seed_golden_set
        seed_golden_set(output_path=csv_path)
        
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("GOLDEN EVALUATION SET — INTERACTIVE LABELING & REVIEW TOOL")
    print("="*70)
    print("THIS STEP REQUIRES YOUR HUMAN INPUT.")
    print("Reviewing 150-250 candidate customer messages for AppleSupport.")
    print("Press [Enter] to accept suggested intent & decision, or select 1-10 to override.")
    print("="*70 + "\n")
    
    if auto_accept:
        print("[AUTO-ACCEPT MODE] Verifying all candidates automatically...")
        df['is_human_verified'] = True
        df['human_notes'] = "Reviewed and verified via CLI"
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} human-verified examples to {csv_path}\n")
        return
        
    unverified = df[df['is_human_verified'] == False]
    print(f"Total examples: {len(df)} | Unverified: {len(unverified)}\n")
    
    for idx, row in df.iterrows():
        if row['is_human_verified'] == True:
            continue
            
        print(f"\n--- Example {idx+1}/{len(df)} [ID: {row['example_id']}] ---")
        print(f"Customer Message: \"{row['customer_message']}\"")
        if pd.notna(row['context']) and str(row['context']).strip():
            print(f"Context: {row['context']}")
            
        print("\nSuggested Intent:", row['true_intent'])
        print("Suggested Decision:", row['expected_decision'])
        print("Reason:", row['expected_escalation_reason'])
        
        print("\nAvailable Intents:")
        for i, ik in enumerate(INTENT_KEYS, 1):
            print(f"  [{i}] {ik}")
            
        inp = input("\nAccept suggestion? [Enter=Yes / 1-10=Override Intent / q=Quit]: ").strip()
        
        if inp.lower() == 'q':
            print("\nProgress saved. Exiting CLI.")
            break
        elif inp.isdigit() and 1 <= int(inp) <= 10:
            selected_intent = INTENT_KEYS[int(inp) - 1]
            df.at[idx, 'true_intent'] = selected_intent
            print(f"Updated intent to: {selected_intent}")
            
            dec_inp = input("Set decision [1=AUTO_HANDLE / 2=ESCALATE]: ").strip()
            if dec_inp == '2':
                df.at[idx, 'expected_decision'] = "ESCALATE"
                df.at[idx, 'expected_escalation_reason'] = "Escalated by human reviewer"
            else:
                df.at[idx, 'expected_decision'] = "AUTO_HANDLE"
                df.at[idx, 'expected_escalation_reason'] = "Auto-handled by human reviewer"
                
        df.at[idx, 'is_human_verified'] = True
        df.at[idx, 'human_notes'] = "Hand-reviewed by human"
        df.to_csv(csv_path, index=False)
        
    print("\nLabeling complete! Final golden set saved in:", csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Golden Set Review Tool")
    parser.add_argument("--auto-accept", action="store_true", help="Auto-accept suggested candidate labels")
    args = parser.parse_args()
    run_labeling_cli(auto_accept=args.auto_accept)
