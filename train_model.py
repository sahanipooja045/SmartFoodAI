"""
SmartFoodAI - Model Training & Evaluation Script
Trains a Random Forest Regressor using Scikit-Learn Pipeline and ColumnTransformer.
Evaluates model using MAE, RMSE, and R², and serializes artifacts using Joblib.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import (
    load_raw_data,
    clean_data,
    engineer_features,
    build_preprocessor,
    get_feature_names,
    prepare_train_test_data,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_VARIABLE,
    TARGET_DERIVED_METRICS
)

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "cleaned_food_wastage_data.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("      SmartFoodAI - Machine Learning Model Training (Phase 4)     ")
    print("=" * 65)

    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"[!] Cleaned dataset not found. Generating from raw data...")
        raw_path = os.path.join(base_dir, "data", "food_wastage_data.csv")
        df_raw = load_raw_data(raw_path)
        df_clean = clean_data(df_raw)
        df_clean = engineer_features(df_clean, include_target_metrics=True)
        df_clean.to_csv(data_path, index=False)
    else:
        df_clean = pd.read_csv(data_path)

    print(f"\n[1] Loaded Dataset: {df_clean.shape[0]} samples, {df_clean.shape[1]} columns")

    # 2. Extract unique categorical options for UI dropdowns and validation
    categorical_options = {}
    for col in CATEGORICAL_FEATURES:
        categorical_options[col] = sorted(df_clean[col].dropna().unique().tolist())

    numerical_ranges = {
        'Number of Guests': {
            'min': int(df_clean['Number of Guests'].min()),
            'max': int(df_clean['Number of Guests'].max()),
            'default': int(df_clean['Number of Guests'].median())
        },
        'Quantity of Food': {
            'min': int(df_clean['Quantity of Food'].min()),
            'max': int(df_clean['Quantity of Food'].max()),
            'default': int(df_clean['Quantity of Food'].median())
        }
    }

    # 3. 80:20 Train-Test Split (Zero Target Leakage)
    print("\n[2] Splitting Data (80% Train, 20% Test, random_state=42)...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        df_clean, test_size=0.2, random_state=42
    )
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples:  {len(X_test)}")

    # 4. Construct Preprocessing Pipeline
    print("\n[3] Fitting Scikit-Learn Preprocessing Pipeline...")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    transformed_feature_names = get_feature_names(preprocessor)
    print(f"    Post-encoding feature space: {len(transformed_feature_names)} features")

    # 5. Initialize & Train Random Forest Regressor
    print("\n[4] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_model)
    ])

    full_pipeline.fit(X_train, y_train)
    print("    [+] Model training complete.")

    # 6. Evaluation on Train & Test sets
    print("\n[5] Model Evaluation Metrics:")
    y_train_pred = full_pipeline.predict(X_train)
    y_test_pred = full_pipeline.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Baseline comparison (Mean predictor)
    mean_baseline_pred = np.full_like(y_test, fill_value=y_train.mean())
    baseline_mae = mean_absolute_error(y_test, mean_baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_baseline_pred))

    print(f"\n    {'Metric':<25} | {'Training Set':<15} | {'Test Set (Unseen)':<18} | {'Baseline (Mean)':<15}")
    print("    " + "-" * 78)
    print(f"    {'MAE (Mean Absolute Error)':<25} | {train_mae:<15.3f} | {test_mae:<18.3f} | {baseline_mae:<15.3f}")
    print(f"    {'RMSE (Root Mean Sq Error)':<25} | {train_rmse:<15.3f} | {test_rmse:<18.3f} | {baseline_rmse:<15.3f}")
    print(f"    {'R² Score (Variance Explained)':<25} | {train_r2:<15.3f} | {test_r2:<18.3f} | {'0.000':<15}")

    # 7. Extract Feature Importances
    fitted_rf = full_pipeline.named_steps['regressor']
    importances = fitted_rf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': transformed_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n[6] Top 5 Most Influential Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"    {idx+1}. {row['Feature']:<30} : {row['Importance']*100:.2f}%")

    # 8. Save Serialized Artifacts with Joblib
    print("\n[7] Serializing Model Artifacts to models/...")
    
    # Save standalone fitted preprocessor & standalone model (useful for SHAP)
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    model_path = os.path.join(models_dir, "random_forest_model.joblib")
    pipeline_path = os.path.join(models_dir, "food_waste_pipeline.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.joblib")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(fitted_rf, model_path)
    joblib.dump(full_pipeline, pipeline_path)

    # Save representative background sample transformed for SHAP TreeExplainer
    X_train_trans = preprocessor.transform(X_train)
    shap_background_sample = X_train_trans[:100]

    metadata = {
        'model_name': 'Random Forest Regressor',
        'metrics': {
            'train_mae': round(float(train_mae), 3),
            'train_rmse': round(float(train_rmse), 3),
            'train_r2': round(float(train_r2), 3),
            'test_mae': round(float(test_mae), 3),
            'test_rmse': round(float(test_rmse), 3),
            'test_r2': round(float(test_r2), 3),
            'baseline_mae': round(float(baseline_mae), 3),
            'baseline_rmse': round(float(baseline_rmse), 3),
        },
        'feature_names': transformed_feature_names,
        'categorical_features': CATEGORICAL_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_options': categorical_options,
        'numerical_ranges': numerical_ranges,
        'top_importances': importance_df.head(10).to_dict(orient='records'),
        'target_variable': TARGET_VARIABLE,
        'shap_background_sample': shap_background_sample
    }
    joblib.dump(metadata, metadata_path)

    print(f"    [+] Saved: {pipeline_path}")
    print(f"    [+] Saved: {model_path}")
    print(f"    [+] Saved: {preprocessor_path}")
    print(f"    [+] Saved: {metadata_path}")

    print("\n" + "=" * 65)
    print("   Model Training & Serialization Completed Successfully!   ")
    print("=" * 65)
    return True

if __name__ == "__main__":
    train_and_evaluate()
