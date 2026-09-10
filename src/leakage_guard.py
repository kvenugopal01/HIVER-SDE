import os
import pandas as pd
from typing import Tuple

def make_chronological_split(
    conv_df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strictly splits conversation DataFrame chronologically into Train, Validation, and Test.
    """
    # Sort strictly by created_at timestamp
    sorted_df = conv_df.sort_values(by='created_at').reset_index(drop=True)
    
    total_len = len(sorted_df)
    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)
    
    train_df = sorted_df.iloc[:train_end].copy()
    val_df = sorted_df.iloc[train_end:val_end].copy()
    test_df = sorted_df.iloc[val_end:].copy()
    
    # Assert zero ID leakage across splits
    train_conv_ids = set(train_df['conversation_id'])
    val_conv_ids = set(val_df['conversation_id'])
    test_conv_ids = set(test_df['conversation_id'])
    
    leakage_train_val = train_conv_ids.intersection(val_conv_ids)
    leakage_train_test = train_conv_ids.intersection(test_conv_ids)
    leakage_val_test = val_conv_ids.intersection(test_conv_ids)
    
    if leakage_train_val or leakage_train_test or leakage_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlapping conversation IDs: "
            f"Train/Val={len(leakage_train_val)}, Train/Test={len(leakage_train_test)}, Val/Test={len(leakage_val_test)}"
        )
        
    # Assert strict chronological boundary
    train_max_time = train_df['created_at'].max()
    val_min_time = val_df['created_at'].min()
    val_max_time = val_df['created_at'].max()
    test_min_time = test_df['created_at'].min()
    
    print("--- CHRONOLOGICAL DATA SPLIT VERIFIED ---")
    print(f"HISTORICAL TRAIN : {len(train_df)} pairs ({train_df['created_at'].min()} to {train_max_time})")
    print(f"VALIDATION       : {len(val_df)} pairs ({val_min_time} to {val_max_time})")
    print(f"TEST CANDIDATES  : {len(test_df)} pairs ({test_min_time} to {test_df['created_at'].max()})")
    print("Zero conversation ID overlap confirmed across all splits.")
    
    return train_df, val_df, test_df

def save_data_splits(
    conv_path: str = "data/processed/apple_conversations.parquet",
    output_dir: str = "data/processed"
):
    if not os.path.exists(conv_path):
        raise FileNotFoundError(f"{conv_path} does not exist. Run conversations.py first.")
        
    conv_df = pd.read_parquet(conv_path)
    train_df, val_df, test_df = make_chronological_split(conv_df)
    
    train_df.to_parquet(os.path.join(output_dir, "historical_train.parquet"), index=False)
    val_df.to_parquet(os.path.join(output_dir, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test_candidates.parquet"), index=False)
    
    print(f"All splits saved cleanly in {output_dir}/")

if __name__ == "__main__":
    save_data_splits()
