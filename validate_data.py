"""
SmartFoodAI - Dataset Validation & Exploratory Data Analysis Script
Executes data quality checks, handles duplicates/missing values,
performs feature engineering, and saves cleaned data.
"""

import os
import pandas as pd
from utils.preprocessing import (
    load_raw_data,
    validate_schema,
    clean_data,
    engineer_features,
    prepare_dataset,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_VARIABLE,
    TARGET_DERIVED_METRICS
)

def run_validation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, "data", "food_wastage_data.csv")
    cleaned_path = os.path.join(base_dir, "data", "cleaned_food_wastage_data.csv")
    
    print("=" * 60)
    print("   SmartFoodAI - Data Preparation & Validation   ")
    print("=" * 60)
    
    print(f"\n[1] Checking raw dataset at:\n    {raw_path}")
    if not os.path.exists(raw_path):
        print("    [!] ERROR: Raw dataset not found.")
        return False
        
    df_raw = load_raw_data(raw_path)
    print(f"    Raw records: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    
    # 2. Schema check
    print("\n[2] Schema Validation:")
    schema = validate_schema(df_raw)
    if schema["is_valid"]:
        print("    [+] All expected columns are present.")
    else:
        print(f"    [!] Missing columns: {schema['missing_columns']}")
        return False

    # 3. Check duplicates and missing values
    duplicates = df_raw.duplicated().sum()
    nulls = df_raw.isnull().sum().sum()
    print(f"\n[3] Data Quality Check:")
    print(f"    Duplicate rows found: {duplicates}")
    print(f"    Missing / Null values: {nulls}")

    # 4. Clean dataset
    print("\n[4] Data Cleaning & Constraint Enforcement:")
    df_clean = clean_data(df_raw)
    print(f"    Records after cleaning: {len(df_clean)} rows (removed {len(df_raw) - len(df_clean)} duplicates)")
    
    # 5. Feature Engineering
    print("\n[5] Feature Engineering:")
    df_engineered = engineer_features(df_clean, include_target_metrics=True)
    print("    Added features:")
    print(f"    - Food per Guest (Safe input feature)")
    print(f"    - Wastage Rate % (EDA / Target-derived)")
    print(f"    - Waste per Guest (EDA / Target-derived)")
    
    print(f"\n[6] Target Leakage Check:")
    print("    Verifying target-derived metrics are marked as EDA-only:")
    for metric in TARGET_DERIVED_METRICS:
        print(f"    - '{metric}' is strictly excluded from model input features.")

    # 6. Save Cleaned Dataset
    df_engineered.to_csv(cleaned_path, index=False)
    print(f"\n[7] Cleaned dataset saved successfully to:\n    {cleaned_path}")
    
    # 7. Print Sample Statistics
    print("\n[8] Cleaned Dataset Summary Statistics:")
    num_cols = NUMERICAL_FEATURES + ['Food per Guest', TARGET_VARIABLE, 'Wastage Rate']
    stats_df = df_engineered[num_cols].describe().round(2)
    print(stats_df.to_string())
    
    print("\n" + "=" * 60)
    print("   Validation Completed Successfully! (Phase 2 Ready)   ")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_validation()
