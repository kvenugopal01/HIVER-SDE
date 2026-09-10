import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import AppleSupportAgentPipeline

def run_interactive_app():
    print("\n" + "="*70)
    print("APPLE SUPPORT AI AGENT — INTERACTIVE CONSOLE APP")
    print("="*70)
    print("Type a customer query and press Enter. Type 'exit' or 'q' to quit.")
    print("="*70 + "\n")
    
    pipeline = AppleSupportAgentPipeline()
    
    while True:
        try:
            user_input = input("Customer Message > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nExiting Interactive App. Goodbye!")
                break
                
            res = pipeline.process_message(user_input)
            
            print("\n" + "-"*50)
            print(f"INTENT             : {res['intent']} (Confidence: {res['intent_confidence']:.2f})")
            print(f"DECISION           : [{res['decision']}]")
            print(f"REASON             : {res['reason']}")
            print(f"DRAFT RESPONSE     : \"{res['draft_reply']}\"")
            print("RETRIEVED EVIDENCE :")
            for i, ev in enumerate(res['retrieved_evidence'], 1):
                print(f"  [{i}] Sim {ev['similarity_score']:.2f} | Cust: \"{ev['historical_issue'][:50]}...\"")
                print(f"      Resolution : \"{ev['historical_resolution'][:60]}...\"")
            print("-" * 50 + "\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Interactive App. Goodbye!")
            break

if __name__ == "__main__":
    run_interactive_app()
