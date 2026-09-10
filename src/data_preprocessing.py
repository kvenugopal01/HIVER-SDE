import html
import re
import os
import glob
import pandas as pd
from datetime import datetime

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Decode HTML entities
    text = html.unescape(text)
    # Replace multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_twitter_date(date_str: str) -> datetime:
    """
    Parses Twitter date string format: 'Tue Oct 31 22:10:47 +0000 2017'
    """
    try:
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        # Fallback for ISO format or pandas timestamp
        return pd.to_datetime(date_str, utc=True)

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Loads raw parquet or CSV files from data_dir.
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        full_df = pd.concat(dfs, ignore_index=True)
    else:
        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if csv_files:
            dfs = [pd.read_csv(f) for f in csv_files]
            full_df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(f"No parquet or CSV files found in {data_dir}")
    
    return full_df

def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw tweets, parses timestamps, removes exact duplicates.
    """
    df = df.copy()
    
    # Ensure tweet_id is int
    df['tweet_id'] = pd.to_numeric(df['tweet_id'], errors='coerce').astype('Int64')
    
    # Clean text
    df['text_clean'] = df['text'].apply(clean_text)
    
    # Filter empty texts
    df = df[df['text_clean'] != ""].copy()
    
    # Parse created_at to UTC datetime
    df['created_at_dt'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce', utc=True)
    
    # Drop rows where timestamp parsing failed
    df = df.dropna(subset=['created_at_dt']).copy()
    
    # Deduplicate on tweet_id
    df = df.drop_duplicates(subset=['tweet_id']).copy()
    
    return df

if __name__ == "__main__":
    print("Loading raw data...")
    raw_df = load_raw_data("data/raw")
    print(f"Raw data loaded: {len(raw_df)} rows")
    
    processed_df = preprocess_tweets(raw_df)
    print(f"Preprocessed data: {len(processed_df)} rows")
