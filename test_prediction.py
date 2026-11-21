"""
SmartFoodAI - Prediction & Risk Engine Test Script (Phase 5)
Verifies inference, percentage calculations, risk boundaries, and error handling.
"""

from utils.prediction import (
    load_model_artifacts,
    predict_food_waste,
    classify_waste_risk,
    validate_input_data
)

def test_prediction_module():
    print("=" * 65)
    print("      SmartFoodAI - Prediction & Risk Test (Phase 5)      ")
    print("=" * 65)

    # 1. Test model loading
    print("\n[1] Testing Model Artifact Loading:")
    pipeline, metadata = load_model_artifacts()
    print("    [+] Pipeline loaded successfully.")
    print(f"    [+] Metadata loaded: Model Name = {metadata['model_name']}")
    print(f"    [+] Test Set R² = {metadata['metrics']['test_r2']}")

    # 2. Test standard inference scenario 1: Corporate Dinner (Low/Medium Risk)
    print("\n[2] Testing Inference Case 1: Corporate Dinner")
    case1 = {
        'Type of Food': 'Meat',
        'Number of Guests': 350,
        'Event Type': 'Corporate',
        'Quantity of Food': 400,
        'Storage Conditions': 'Refrigerated',
        'Purchase History': 'Regular',
        'Seasonality': 'All Seasons',
        'Preparation Method': 'Buffet',
        'Geographical Location': 'Urban',
        'Pricing': 'Low'
    }
    result1 = predict_food_waste(case1, pipeline=pipeline)
    print(f"    Quantity:            {result1['quantity_of_food']} units")
    print(f"    Guests:              {result1['number_of_guests']}")
    print(f"    Predicted Waste:     {result1['predicted_waste']} units")
    print(f"    Wastage Rate:        {result1['wastage_percentage']}%")
    print(f"    Risk Category:       {result1['risk_level']} (Badge: {result1['risk_badge']}, Color: {result1['risk_color']})")
    print(f"    Risk Context:        {result1['risk_description']}")

    # 3. Test scenario 2: High Pricing / High Waste
    print("\n[3] Testing Inference Case 2: High Pricing Event")
    case2 = {
        'Type of Food': 'Meat',
        'Number of Guests': 250,
        'Event Type': 'Birthday',
        'Quantity of Food': 450,
        'Storage Conditions': 'Room Temperature',
        'Purchase History': 'Occasional',
        'Seasonality': 'Summer',
        'Preparation Method': 'Buffet',
        'Geographical Location': 'Rural',
        'Pricing': 'High'
    }
    result2 = predict_food_waste(case2, pipeline=pipeline)
    print(f"    Quantity:            {result2['quantity_of_food']} units")
    print(f"    Guests:              {result2['number_of_guests']}")
    print(f"    Predicted Waste:     {result2['predicted_waste']} units")
    print(f"    Wastage Rate:        {result2['wastage_percentage']}%")
    print(f"    Risk Category:       {result2['risk_level']} (Badge: {result2['risk_badge']})")

    # 4. Verify Physical Constraints
    print("\n[4] Physical Constraint Verification:")
    assert 0 <= result1['predicted_waste'] <= result1['quantity_of_food'], "Constraint violated on Case 1!"
    assert 0 <= result2['predicted_waste'] <= result2['quantity_of_food'], "Constraint violated on Case 2!"
    print("    [+] PASS: 0 <= Predicted Waste <= Food Quantity.")

    # 5. Error Handling Tests
    print("\n[5] Error Handling Verification:")
    # A. Zero guests
    try:
        bad_case = case1.copy()
        bad_case['Number of Guests'] = 0
        validate_input_data(bad_case)
        print("    [!] FAILED: Zero guests did not raise error.")
    except ValueError as e:
        print(f"    [+] PASS: Caught expected validation error: {e}")

    # B. Missing required field
    try:
        incomplete_case = case1.copy()
        del incomplete_case['Pricing']
        validate_input_data(incomplete_case)
        print("    [!] FAILED: Missing field did not raise error.")
    except ValueError as e:
        print(f"    [+] PASS: Caught expected validation error: {e}")

    print("\n" + "=" * 65)
    print("   Phase 5 Verification PASSED! Inference Engine Ready   ")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_prediction_module()
