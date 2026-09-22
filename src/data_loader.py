import os
import glob
import pandas as pd
from src.health_engine import enrich_with_health_metrics_fast

def load_data():
    """
    Load all parquet chunks (data_part_0.parquet, data_part_1.parquet, etc.)
    and enrich with health metrics.
    """
    # Possible file locations
    parquet_files = sorted(glob.glob("data/data_part_*.parquet") + glob.glob("data_part_*.parquet"))
    
    if parquet_files:
        print(f"Loading {len(parquet_files)} parquet file(s): {parquet_files}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully loaded {len(df):,} total rows.")
        return enrich_with_health_metrics_fast(df)
    
    # Fallback if no parquet files are found
    print("Parquet files not found. Generating sample data for testing...")
    from src.data_loader_mock import generate_mock_accounts
    return generate_mock_accounts(5000)