import os
import re
import pandas as pd
from typing import List, Dict, Any

from src.data_preprocessing import load_raw_data, preprocess_tweets

DM_KEYWORDS = [
    'dm us', 'direct message', 'send us a dm', 'private message', 
    'reach out in dm', 'link in bio', 'send us a message', 'click the link'
]

def is_dm_escalation(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in DM_KEYWORDS)

def build_apple_conversations(df: pd.DataFrame, target_brand: str = "AppleSupport") -> pd.DataFrame:
    """
    Reconstructs customer-support pairs for target_brand using vectorized pandas operations.
    """
    # Separate brand responses and inbound customer tweets
    brand_df = df[df['author_id'] == target_brand].copy()
    cust_df = df[df['inbound'] == True].copy()
    
    # Ensure join keys are numeric int64
    brand_df['in_response_to_tweet_id'] = pd.to_numeric(brand_df['in_response_to_tweet_id'], errors='coerce').astype('Int64')
    cust_df['tweet_id'] = pd.to_numeric(cust_df['tweet_id'], errors='coerce').astype('Int64')
    
    # Merge brand replies with customer tweets on in_response_to_tweet_id == tweet_id
    merged = pd.merge(
        brand_df,
        cust_df,
        left_on='in_response_to_tweet_id',
        right_on='tweet_id',
        suffixes=('_brand', '_cust')
    )
    
    # Clean text columns
    merged['cust_text'] = merged['text_clean_cust']
    merged['brand_text'] = merged['text_clean_brand']
    
    # Filter minimum length
    merged = merged[merged['cust_text'].str.len() >= 10].copy()
    
    # Identify DM escalation
    merged['has_dm_escalation'] = merged['brand_text'].apply(is_dm_escalation)
    
    # Format outcome dataframe
    result = pd.DataFrame({
        'conversation_id': 'conv_' + target_brand.lower() + '_' + merged['tweet_id_cust'].astype(str),
        'cust_tweet_id': merged['tweet_id_cust'],
        'cust_author_id': merged['author_id_cust'].astype(str),
        'cust_text': merged['cust_text'],
        'brand_tweet_id': merged['tweet_id_brand'],
        'brand_text': merged['brand_text'],
        'created_at': merged['created_at_dt_cust'],
        'context': '',
        'has_dm_escalation': merged['has_dm_escalation']
    })
    
    # Sort chronologically by customer timestamp
    result = result.sort_values(by='created_at').reset_index(drop=True)
    return result

def process_and_save_conversations(
    raw_dir: str = "data/raw", 
    output_path: str = "data/processed/apple_conversations.parquet"
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    raw_df = load_raw_data(raw_dir)
    clean_df = preprocess_tweets(raw_df)
    conv_df = build_apple_conversations(clean_df, target_brand="AppleSupport")
    conv_df.to_parquet(output_path, index=False)
    print(f"Successfully saved {len(conv_df)} AppleSupport conversation pairs to {output_path}")
    return conv_df

if __name__ == "__main__":
    process_and_save_conversations()
