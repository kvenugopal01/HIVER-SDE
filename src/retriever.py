import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HistoricalRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
        self.train_pairs: List[Dict[str, Any]] = []
        self.index_matrix = None
        
    def fit(self, train_df: pd.DataFrame):
        print(f"Building Historical Retrieval Index over {len(train_df)} resolution pairs...")
        self.train_pairs = train_df[['conversation_id', 'cust_text', 'brand_text', 'has_dm_escalation']].to_dict('records')
        cust_texts = [p['cust_text'] for p in self.train_pairs]
        self.index_matrix = self.vectorizer.fit_transform(cust_texts)
        print("Historical Retrieval Index built successfully.")
        return self
        
    def retrieve(self, query: str) -> Dict[str, Any]:
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.index_matrix)[0]
        
        # Get top-k indices sorted descending
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        evidence = []
        max_sim = float(similarities[top_indices[0]]) if len(top_indices) > 0 else 0.0
        
        for idx in top_indices:
            pair = self.train_pairs[idx]
            sim_score = float(similarities[idx])
            evidence.append({
                "conversation_id": pair['conversation_id'],
                "historical_issue": pair['cust_text'],
                "historical_resolution": pair['brand_text'],
                "similarity_score": round(sim_score, 4),
                "has_dm_escalation": pair['has_dm_escalation']
            })
            
        return {
            "top_similarity_score": round(max_sim, 4),
            "evidence": evidence
        }

def build_and_save_retriever(
    train_path: str = "data/processed/historical_train.parquet",
    output_path: str = "data/processed/retriever.pkl"
):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} missing. Run leakage_guard.py first.")
        
    train_df = pd.read_parquet(train_path)
    retriever = HistoricalRetriever(top_k=3).fit(train_df)
    
    with open(output_path, "wb") as f:
        pickle.dump(retriever, f)
        
    print(f"Retriever index saved cleanly to {output_path}")

if __name__ == "__main__":
    build_and_save_retriever()
