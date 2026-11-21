"""
SmartFoodAI - Smart Recommendation Engine (Phase 6)
Generates candidate quantities (85%-100% of current quantity in 5-unit steps),
evaluates predicted waste across candidates using the ML model, and identifies
the waste-minimizing preparation recommendation with transparent heuristic disclaimers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from utils.prediction import (
    load_model_artifacts,
    predict_food_waste,
    classify_waste_risk,
    validate_input_data
)

HEURISTIC_DISCLAIMER = (
    "Note: This recommendation is a waste-minimization heuristic derived from model associations, "
    "not guaranteed customer demand optimization. Because individual customer appetite, table seconds, "
    "and shortage costs are not recorded in the dataset, kitchen management should exercise judgment "
    "before reducing quantities."
)


def generate_candidate_quantities(
    current_quantity: float,
    min_ratio: float = 0.85,
    max_ratio: float = 1.0,
    step: float = 5.0
) -> List[float]:
    """
    Generates candidate food quantities between min_ratio (default 85%)
    and max_ratio (default 100%) of the current preparation quantity.
    Uses 5-unit steps where applicable and ensures strictly positive values.
    """
    if current_quantity <= 0:
        raise ValueError("Current quantity must be greater than 0.")

    min_qty = current_quantity * min_ratio
    max_qty = current_quantity * max_ratio

    # If the span allows 5-unit steps
    if (max_qty - min_qty) >= step:
        # Snap lower bound to next multiple of step for clean restaurant batches
        start_qty = np.ceil(min_qty / step) * step
        candidates = list(np.arange(start_qty, max_qty, step))
    else:
        # For small quantities, generate 5 evenly spaced candidate points
        candidates = list(np.linspace(min_qty, max_qty, 5))

    # Always ensure current_quantity is evaluated as the baseline upper boundary
    if max_qty not in candidates:
        candidates.append(max_qty)

    # Filter out non-positive values, round, remove duplicates and sort
    clean_candidates = sorted(list(set([round(float(q), 1) for q in candidates if q > 0])))
    return clean_candidates


def recommend_optimal_quantity(
    input_data: Dict[str, Any],
    pipeline=None,
    min_ratio: float = 0.85,
    step: float = 5.0
) -> Dict[str, Any]:
    """
    Recommends optimal food quantity:
    1. Evaluates current plan (baseline).
    2. Tests candidate quantities in the [85%, 100%] range.
    3. Finds candidate that minimizes predicted food waste.
    4. Computes potential waste reduction, waste rates, and risk shift.
    5. Returns candidate comparison curve for interactive visualization.
    """
    if pipeline is None:
        pipeline, _ = load_model_artifacts()

    # Validate inputs
    _ = validate_input_data(input_data)
    current_quantity = float(input_data['Quantity of Food'])
    guests = int(input_data['Number of Guests'])

    # 1. Baseline Evaluation
    baseline_result = predict_food_waste(input_data, pipeline=pipeline)
    current_waste = baseline_result['predicted_waste']
    current_waste_rate = baseline_result['wastage_percentage']
    current_risk = baseline_result['risk_level']

    # 2. Candidate Generation
    candidates = generate_candidate_quantities(
        current_quantity=current_quantity,
        min_ratio=min_ratio,
        max_ratio=1.0,
        step=step
    )

    # 3. Candidate Evaluation
    candidate_records = []
    best_candidate_qty = current_quantity
    min_candidate_waste = current_waste

    for cand_qty in candidates:
        cand_input = input_data.copy()
        cand_input['Quantity of Food'] = cand_qty

        # Evaluate candidate with ML model
        cand_result = predict_food_waste(cand_input, pipeline=pipeline)
        cand_waste = cand_result['predicted_waste']
        cand_rate = cand_result['wastage_percentage']
        cand_risk = cand_result['risk_level']

        candidate_records.append({
            'candidate_quantity': cand_qty,
            'predicted_waste': cand_waste,
            'wastage_rate': cand_rate,
            'food_per_guest': round(cand_qty / guests, 3),
            'risk_level': cand_risk,
            'is_current': (cand_qty == current_quantity)
        })

        # Track candidate with lowest predicted waste
        # If waste is tied, prefer higher quantity to minimize customer shortage risk
        if cand_waste < min_candidate_waste:
            min_candidate_waste = cand_waste
            best_candidate_qty = cand_qty

    # 4. Compute Reduction Metrics
    optimal_result = predict_food_waste(
        {**input_data, 'Quantity of Food': best_candidate_qty},
        pipeline=pipeline
    )
    rec_waste = optimal_result['predicted_waste']
    rec_waste_rate = optimal_result['wastage_percentage']
    rec_risk = optimal_result['risk_level']

    potential_waste_reduction = round(max(0.0, current_waste - rec_waste), 2)
    quantity_reduction = round(current_quantity - best_candidate_qty, 2)
    
    if current_waste > 0:
        waste_reduction_pct = round((potential_waste_reduction / current_waste) * 100.0, 1)
    else:
        waste_reduction_pct = 0.0

    return {
        "current_quantity": current_quantity,
        "recommended_quantity": best_candidate_qty,
        "quantity_reduction": quantity_reduction,
        "current_predicted_waste": current_waste,
        "recommended_predicted_waste": rec_waste,
        "potential_waste_reduction": potential_waste_reduction,
        "waste_reduction_percentage": waste_reduction_pct,
        "current_waste_rate": current_waste_rate,
        "recommended_waste_rate": rec_waste_rate,
        "current_risk": current_risk,
        "recommended_risk": rec_risk,
        "current_food_per_guest": round(current_quantity / guests, 3),
        "recommended_food_per_guest": round(best_candidate_qty / guests, 3),
        "candidates": candidate_records,
        "disclaimer": HEURISTIC_DISCLAIMER
    }
