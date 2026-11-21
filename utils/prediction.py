"""
SmartFoodAI - Prediction & Risk Assessment Engine (Phase 5)
Handles model loading, input validation, food waste inference,
wastage percentage calculations, and configurable risk assessment.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Configurable Project-Defined Risk Thresholds (%)
# Note: These are heuristic risk categories defined for the SmartFoodAI project,
# not official statutory or government standards.
DEFAULT_RISK_THRESHOLDS = {
    "low_max": 6.0,     # Below 6.0% -> Low Risk
    "medium_max": 10.0  # 6.0% to 10.0% -> Medium Risk, > 10.0% -> High Risk
}

RISK_METADATA = {
    "Low Risk": {
        "color": "#10B981",  # Emerald Green
        "badge": "LOW",
        "description": "Safe operational range. Minimal surplus expected."
    },
    "Medium Risk": {
        "color": "#F59E0B",  # Amber
        "badge": "MEDIUM",
        "description": "Moderate surplus projected. Recommend monitoring portion size and preparation buffers."
    },
    "High Risk": {
        "color": "#EF4444",  # Red
        "badge": "HIGH",
        "description": "Substantial surplus projected. High likelihood of unnecessary food waste. Adjustment recommended."
    }
}


def get_model_paths() -> Tuple[str, str]:
    """Returns absolute paths for pipeline and metadata artifacts."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pipeline_path = os.path.join(base_dir, "models", "food_waste_pipeline.joblib")
    metadata_path = os.path.join(base_dir, "models", "model_metadata.joblib")
    return pipeline_path, metadata_path


def load_model_artifacts():
    """
    Loads serialized pipeline and metadata.
    Raises FileNotFoundError with clear instructions if models do not exist.
    """
    pipeline_path, metadata_path = get_model_paths()
    if not os.path.exists(pipeline_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Trained model artifacts not found. Please run 'python train_model.py' first.\n"
            f"Missing path: {pipeline_path}"
        )
    pipeline = joblib.load(pipeline_path)
    metadata = joblib.load(metadata_path)
    return pipeline, metadata


def validate_input_data(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates restaurant inputs:
    - Number of Guests > 0
    - Quantity of Food > 0
    - Checks required categorical keys
    - Automatically derives 'Food per Guest'
    """
    required_keys = [
        'Type of Food',
        'Number of Guests',
        'Event Type',
        'Quantity of Food',
        'Storage Conditions',
        'Purchase History',
        'Seasonality',
        'Preparation Method',
        'Geographical Location',
        'Pricing'
    ]

    for key in required_keys:
        if key not in input_data:
            raise ValueError(f"Missing required input parameter: '{key}'")

    try:
        guests = int(input_data['Number of Guests'])
        quantity = float(input_data['Quantity of Food'])
    except (ValueError, TypeError):
        raise ValueError("Guests and Quantity must be valid positive numbers.")

    if guests <= 0:
        raise ValueError(f"Number of Guests must be greater than 0 (received: {guests}).")
    if quantity <= 0:
        raise ValueError(f"Quantity of Food must be greater than 0 (received: {quantity}).")

    # Safe engineered feature
    food_per_guest = round(quantity / guests, 3)

    record = {
        'Number of Guests': guests,
        'Quantity of Food': quantity,
        'Food per Guest': food_per_guest,
        'Type of Food': str(input_data['Type of Food']),
        'Event Type': str(input_data['Event Type']),
        'Storage Conditions': str(input_data['Storage Conditions']),
        'Purchase History': str(input_data['Purchase History']),
        'Seasonality': str(input_data['Seasonality']),
        'Preparation Method': str(input_data['Preparation Method']),
        'Geographical Location': str(input_data['Geographical Location']),
        'Pricing': str(input_data['Pricing'])
    }

    return pd.DataFrame([record])


def classify_waste_risk(
    wastage_percentage: float,
    thresholds: Dict[str, float] = None
) -> Dict[str, str]:
    """
    Classifies food waste risk level based on predicted wastage rate (percentage).
    Returns risk category, badge color, and description.
    """
    if thresholds is None:
        thresholds = DEFAULT_RISK_THRESHOLDS

    if wastage_percentage < thresholds["low_max"]:
        category = "Low Risk"
    elif wastage_percentage <= thresholds["medium_max"]:
        category = "Medium Risk"
    else:
        category = "High Risk"

    info = RISK_METADATA[category]
    return {
        "level": category,
        "color": info["color"],
        "badge": info["badge"],
        "description": info["description"],
        "threshold_used": f"< {thresholds['low_max']}% Low, {thresholds['low_max']}-{thresholds['medium_max']}% Medium, > {thresholds['medium_max']}% High"
    }


def predict_food_waste(
    input_data: Dict[str, Any],
    pipeline=None,
    thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Complete inference pipeline:
    1. Validates inputs & constructs single-row DataFrame.
    2. Runs trained pipeline to predict wastage amount.
    3. Enforces physical boundary constraints: 0 <= waste <= quantity.
    4. Calculates wastage percentage.
    5. Evaluates project-defined risk assessment.
    """
    if pipeline is None:
        pipeline, _ = load_model_artifacts()

    input_df = validate_input_data(input_data)
    quantity = float(input_df['Quantity of Food'].iloc[0])
    guests = int(input_df['Number of Guests'].iloc[0])

    # ML Inference
    raw_prediction = float(pipeline.predict(input_df)[0])

    # Physical logical constraints
    predicted_waste = round(float(np.clip(raw_prediction, 0.0, quantity)), 2)

    # Wastage percentage calculation
    wastage_percentage = round((predicted_waste / quantity) * 100.0, 2)

    # Waste per guest
    waste_per_guest = round(predicted_waste / guests, 3)

    # Risk level classification
    risk_info = classify_waste_risk(wastage_percentage, thresholds)

    return {
        "predicted_waste": predicted_waste,
        "quantity_of_food": quantity,
        "number_of_guests": guests,
        "food_per_guest": round(quantity / guests, 3),
        "waste_per_guest": waste_per_guest,
        "wastage_percentage": wastage_percentage,
        "risk_level": risk_info["level"],
        "risk_color": risk_info["color"],
        "risk_badge": risk_info["badge"],
        "risk_description": risk_info["description"],
        "risk_thresholds": risk_info["threshold_used"],
        "input_features": input_data
    }
