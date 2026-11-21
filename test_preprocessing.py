"""
SmartFoodAI - Preprocessing & Feature Pipeline Test Script (Phase 3)
Verifies ColumnTransformer, One-Hot Encoding, StandardScaler, and 80:20 split.
"""

import os
import pandas as pd
from utils.preprocessing import (
    load_raw_data,
    clean_data,
    engineer_features,
    build_preprocessor,
    get_feature_names,
    prepare_train_test_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_VARIABLE,
    TARGET_DERIVED_METRICS
)

def test_pipeline():
    print("=" * 60)
    print("   SmartFoodAI - Preprocessing & Pipeline Test (Phase 3)   ")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_path = os.path.join(base_dir, "data", "cleaned_food_wastage_data.csv")

    if not os.path.exists(cleaned_path):
        print(f"[!] Cleaned dataset not found at {cleaned_path}. Creating it...")
        raw_path = os.path.join(base_dir, "data", "food_wastage_data.csv")
        df_raw = load_raw_data(raw_path)
        df_clean = clean_data(df_raw)
        df_clean = engineer_features(df_clean, include_target_metrics=True)
        df_clean.to_csv(cleaned_path, index=False)
    else:
        df_clean = pd.read_csv(cleaned_path)

    print(f"\n[1] Loaded cleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    # 2. Train-test split (80:20)
    print("\n[2] Splitting into Train (80%) and Test (20%) sets:")
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_clean, test_size=0.2, random_state=42)
    print(f"    X_train shape: {X_train.shape} (80%)")
    print(f"    X_test shape:  {X_test.shape} (20%)")
    print(f"    y_train shape: {y_train.shape}")
    print(f"    y_test shape:  {y_test.shape}")

    # 3. Target Leakage Assertion
    print("\n[3] Target Leakage Audit:")
    for metric in TARGET_DERIVED_METRICS:
        if metric in X_train.columns:
            raise ValueError(f"CRITICAL: Target leakage found! '{metric}' is inside training features.")
    print("    [+] PASS: No target-derived metrics found in training features.")

    # 4. Fit Preprocessor
    print("\n[4] Fitting ColumnTransformer Pipeline:")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)
    print(f"    Original input features:    {X_train.shape[1]}")
    print(f"    Transformed feature count:  {X_train_trans.shape[1]}")
    print(f"    Total feature names mapped: {len(feature_names)}")

    print("\n[5] Transformed Features Sample:")
    for i, name in enumerate(feature_names[:10]):
        print(f"    Feature {i+1:02d}: {name}")
    if len(feature_names) > 10:
        print(f"    ... and {len(feature_names) - 10} more features.")

    # 5. Sanity Checks
    assert not pd.isna(X_train_trans).any(), "NaN values found in transformed train set!"
    assert not pd.isna(X_test_trans).any(), "NaN values found in transformed test set!"
    assert X_train_trans.shape[1] == len(feature_names), "Feature names count mismatch!"

    print("\n" + "=" * 60)
    print("   Phase 3 Verification PASSED! Preprocessing Ready for ML   ")
    print("=" * 60)
    return True

if __name__ == "__main__":
    test_pipeline()
