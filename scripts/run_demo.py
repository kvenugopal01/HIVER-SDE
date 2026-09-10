import os
import sys
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import AppleSupportAgentPipeline

def run_demo(message: str):
    pipeline = AppleSupportAgentPipeline()
    res = pipeline.process_message(message)
    
    print("\n" + "="*70)
    print("APPLE SUPPORT AI AGENT — LIVE DEMO")
    print("="*70)
    print(f"\nCustomer message:\n  \"{res['customer_message']}\"\n")
    print(f"Intent:\n  {res['intent']} (Confidence: {res['intent_confidence']:.2f})\n")
    
    print("Retrieved historical evidence:")
    for i, ev in enumerate(res['retrieved_evidence'], 1):
        print(f"  [{i}] (Sim: {ev['similarity_score']:.2f}) Cust: \"{ev['historical_issue'][:60]}...\"")
        print(f"      Brand Reply: \"{ev['historical_resolution'][:70]}...\"")
        
    print(f"\nDraft response:\n  \"{res['draft_reply']}\"\n")
    print(f"Decision:\n  [{res['decision']}]\n")
    print(f"Reason:\n  {res['reason']}\n")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Demo for AppleSupport AI Agent")
    parser.add_argument("--message", type=str, default="My iPhone battery drops from 80% to 10% after installing iOS 11", help="Customer query text")
    args = parser.parse_args()
    run_demo(args.message)
