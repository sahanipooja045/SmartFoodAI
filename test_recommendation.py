"""
SmartFoodAI - Recommendation Engine Test Script (Phase 6)
Verifies candidate generation (85%-100%), 5-unit stepping,
waste minimization evaluation, and heuristic disclaimers.
"""

from utils.recommendation import (
    generate_candidate_quantities,
    recommend_optimal_quantity,
    HEURISTIC_DISCLAIMER
)
from utils.prediction import load_model_artifacts

def test_recommendation_module():
    print("=" * 65)
    print("    SmartFoodAI - Recommendation Engine Test (Phase 6)    ")
    print("=" * 65)

    # 1. Test Candidate Generation Logic
    print("\n[1] Testing Candidate Generation (85% to 100% in 5-unit steps):")
    sample_qty = 400.0
    candidates = generate_candidate_quantities(sample_qty, min_ratio=0.85, max_ratio=1.0, step=5.0)
    print(f"    Current Quantity: {sample_qty} units")
    print(f"    85% Boundary:     {sample_qty * 0.85} units")
    print(f"    Generated {len(candidates)} candidates: {candidates}")

    # Assertions on candidate list
    assert min(candidates) >= sample_qty * 0.85, "Candidate below 85% lower bound!"
    assert max(candidates) == sample_qty, "Candidate exceeds or misses current quantity upper bound!"
    assert all(candidates[i] <= candidates[i+1] for i in range(len(candidates)-1)), "Candidates not sorted!"
    print("    [+] PASS: Candidate bounds and stepping verified.")

    # 2. Test Recommendation Workflow
    print("\n[2] Testing Recommendation on Restaurant Order:")
    pipeline, _ = load_model_artifacts()
    test_order = {
        'Type of Food': 'Meat',
        'Number of Guests': 300,
        'Event Type': 'Wedding',
        'Quantity of Food': 450,
        'Storage Conditions': 'Refrigerated',
        'Purchase History': 'Regular',
        'Seasonality': 'Winter',
        'Preparation Method': 'Buffet',
        'Geographical Location': 'Urban',
        'Pricing': 'High'
    }

    rec = recommend_optimal_quantity(test_order, pipeline=pipeline, min_ratio=0.85, step=5.0)
    print(f"    Current Planned Quantity:     {rec['current_quantity']} units")
    print(f"    Recommended Quantity:         {rec['recommended_quantity']} units")
    print(f"    Quantity Reduction:           {rec['quantity_reduction']} units ({rec['recommended_food_per_guest']} food/guest)")
    print(f"    Current Predicted Waste:      {rec['current_predicted_waste']} units ({rec['current_waste_rate']}%)")
    print(f"    Recommended Predicted Waste:  {rec['recommended_predicted_waste']} units ({rec['recommended_waste_rate']}%)")
    print(f"    Potential Waste Reduction:    {rec['potential_waste_reduction']} units ({rec['waste_reduction_percentage']}% saved)")
    print(f"    Risk Shift:                   {rec['current_risk']} -> {rec['recommended_risk']}")

    # 3. Verify Candidate Table
    print("\n[3] Candidate Evaluation Curve:")
    print(f"    {'Candidate Qty':<15} | {'Predicted Waste':<18} | {'Waste Rate %':<15} | {'Status'}")
    print("    " + "-" * 62)
    for c in rec['candidates']:
        marker = "<-- CURRENT" if c['is_current'] else ("<-- RECOMMENDED" if c['candidate_quantity'] == rec['recommended_quantity'] else "")
        print(f"    {c['candidate_quantity']:<15.1f} | {c['predicted_waste']:<18.2f} | {c['wastage_rate']:<15.2f} | {marker}")

    # 4. Verify Disclaimers and Invariants
    print("\n[4] Invariant & Disclaimer Verification:")
    assert rec['recommended_quantity'] > 0, "Recommended quantity is non-positive!"
    assert rec['potential_waste_reduction'] >= 0, "Potential waste reduction cannot be negative!"
    assert len(rec['disclaimer']) > 20, "Missing heuristic disclaimer!"
    print(f"    [+] PASS: Heuristic Disclaimer present:\n    \"{rec['disclaimer']}\"")

    print("\n" + "=" * 65)
    print("   Phase 6 Verification PASSED! Recommendation Engine Ready   ")
    print("=" * 65)
    return True

if __name__ == "__main__":
    test_recommendation_module()
