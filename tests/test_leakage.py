import pytest
import pandas as pd
from src.leakage_guard import make_chronological_split

def test_make_chronological_split():
    # Create sample chronological dataframe
    dates = pd.date_range("2017-01-01", periods=100, freq="D", tz="UTC")
    data = []
    for i in range(100):
        data.append({
            "conversation_id": f"conv_{i}",
            "cust_tweet_id": i,
            "cust_text": f"Sample message {i}",
            "created_at": dates[i]
        })
    df = pd.DataFrame(data)
    
    train_df, val_df, test_df = make_chronological_split(df, train_ratio=0.8, val_ratio=0.1)
    
    assert len(train_df) == 80
    assert len(val_df) == 10
    assert len(test_df) == 10
    
    # Assert zero ID overlap
    train_ids = set(train_df['conversation_id'])
    val_ids = set(val_df['conversation_id'])
    test_ids = set(test_df['conversation_id'])
    
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    
    # Assert strict timestamp monotonicity
    assert train_df['created_at'].max() <= val_df['created_at'].min()
    assert val_df['created_at'].max() <= test_df['created_at'].min()
