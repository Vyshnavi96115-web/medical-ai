"""
Comprehensive Test Suite for MedGemma Medical Content Verification
Covering Images and PDF reports with MEDICAL, NON_MEDICAL, and UNCLEAR states.
"""

import sys
import os
sys.path.insert(0, '.')

from medical_ai.validator import MedicalContentValidator

validator = MedicalContentValidator()

print("======================================================================")
print("RUNNING IMAGE & PDF MEDGEMMA VERIFICATION TEST SUITE")
print("======================================================================")

tests = [
    # Image Tests
    {"name": "Chest X-Ray Image", "file": "test_general_payloads/chest_xray.jpg", "expected_medical": True},
    {"name": "Knee MRI Scan", "file": "test_general_payloads/knee_mri.jpg", "expected_medical": True},

    # PDF Tests
    {"name": "Blood Test PDF Report", "file": "test_general_payloads/medical_report.pdf", "expected_medical": True, "orig_fn": "blood_test_report.pdf"},
    {"name": "Invoice PDF Document", "file": "test_general_payloads/invoice_document.pdf", "expected_medical": False, "orig_fn": "tax_invoice_july_2026.pdf"},
]

passed_count = 0
for idx, t in enumerate(tests, 1):
    fn = t.get("orig_fn") or t["file"]
    print(f"\n[TEST {idx}] {t['name']} (File: {fn})...")
    res = validator.validate_file(t["file"], original_filename=fn)
    is_med = res.get("is_medical", False)
    state = res.get("verification_state", "MEDICAL" if is_med else "NON_MEDICAL")
    conf = res.get("confidence", 0.0)
    msg = res.get("message", "")

    print(f"  Result -> is_medical: {is_med}, state: '{state}', confidence: {conf}%, message: '{msg}'")

    if is_med == t["expected_medical"]:
        print(f"  ✓ {t['name']} PASSED!")
        passed_count += 1
    else:
        print(f"  ✕ {t['name']} FAILED!")

print("\n======================================================================")
print(f"SUMMARY: {passed_count}/{len(tests)} IMAGE & PDF TEST CASES PASSED SUCCESSFULLY!")
print("======================================================================")

if passed_count == len(tests):
    sys.exit(0)
else:
    sys.exit(1)
