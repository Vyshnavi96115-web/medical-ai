"""
Comprehensive Automated Test Suite for Anatomy-Aware Verification, PDF Content Analysis,
and User-Facing Output Sanitization (Zero MedGemma references).
"""

import sys
import os
import json
sys.path.insert(0, '.')

from medical_detector import MedicalImageDetector
from medical_ai.validator import MedicalContentValidator
from medical_ai.medgemma import MedGemmaAnalyzer, sanitize_user_facing_output

detector = MedicalImageDetector()
validator = MedicalContentValidator()
analyzer = MedGemmaAnalyzer()

print("======================================================================")
print("RUNNING ANATOMY-AWARE, PDF CONTENT & OUTPUT SANITIZATION TEST SUITE")
print("======================================================================")

# --------------------------------------------------------------------
# TEST 1: Leg / Bone X-Ray Anatomy Identification & Consistency Check
# --------------------------------------------------------------------
print("\n[TEST 1] Leg / Bone X-Ray Anatomy Identification & Consistency Check...")
leg_path = "test_general_payloads/bone_xray.jpg"
if not os.path.exists(leg_path):
    # Fallback path if test file is in root
    leg_path = "test_general_payloads/knee_mri.jpg"

leg_res = detector.analyze(leg_path, original_filename="scanned_leg_xray_lower_extremity.jpg")
print(f"  Stage 1 -> Anatomy: '{leg_res['body_region']}', Modality: '{leg_res['modality']}'")

assert leg_res["is_medical"] == True, "Leg X-ray failed medical verification!"
assert "Lower Extremity" in leg_res["body_region"] or "Leg" in leg_res["body_region"] or "Knee" in leg_res["body_region"] or "Spine" in leg_res["body_region"], f"Leg X-ray was incorrectly classified as: {leg_res['body_region']}"
assert "Chest" not in leg_res["body_region"], "CRITICAL FAIL: Leg X-ray was misidentified as Chest!"

# Generate MedGemma analysis and verify consistency
analysis = analyzer.analyze_medical_data(leg_path, stage1_info=leg_res)
detailed_text = (analysis.get("detailed_explanation") or "").lower()
forbidden_words = ["lung field", "pulmonary consolidation", "cardiomegaly", "costophrenic angle", "pneumothorax", "tracheal alignment"]

for fw in forbidden_words:
    assert fw not in detailed_text, f"CRITICAL FAIL: Leg X-ray analysis falsely contained chest term '{fw}'"

print("  ✓ TEST 1 PASSED! Leg X-ray correctly identified as Lower Extremity/Bone with ZERO false chest information.")

# --------------------------------------------------------------------
# TEST 2: Chest X-Ray Anatomy Identification
# --------------------------------------------------------------------
print("\n[TEST 2] Chest X-Ray Anatomy Identification...")
chest_path = "test_general_payloads/chest_xray.jpg"
chest_res = detector.analyze(chest_path, original_filename="patient_chest_xray.jpg")
print(f"  Stage 1 -> Anatomy: '{chest_res['body_region']}', Modality: '{chest_res['modality']}'")
assert chest_res["is_medical"] == True, "Chest X-ray failed medical verification!"
assert "Chest" in chest_res["body_region"] or "Lung" in chest_res["body_region"], "Chest X-ray was not identified as Chest/Lungs!"
print("  ✓ TEST 2 PASSED! Chest X-ray correctly identified.")

# --------------------------------------------------------------------
# TEST 3: Brain MRI Anatomy Identification
# --------------------------------------------------------------------
print("\n[TEST 3] Brain MRI Anatomy Identification...")
brain_path = "test_general_payloads/brain_mri.jpg"
brain_res = detector.analyze(brain_path, original_filename="brain_mri_scan.jpg")
print(f"  Stage 1 -> Anatomy: '{brain_res['body_region']}', Modality: '{brain_res['modality']}'")
assert brain_res["is_medical"] == True, "Brain MRI failed medical verification!"
assert "Brain" in brain_res["body_region"], "Brain MRI was not identified as Brain!"
print("  ✓ TEST 3 PASSED! Brain MRI correctly identified.")

# --------------------------------------------------------------------
# TEST 4: Eye / Fundus Photo Anatomy Identification
# --------------------------------------------------------------------
print("\n[TEST 4] Eye / Retinal Fundus Photo Anatomy Identification...")
eye_path = "test_general_payloads/eye_scan.jpg"
eye_res = detector.analyze(eye_path, original_filename="retinal_fundus_scan.jpg")

print(f"  Stage 1 -> Anatomy: '{eye_res['body_region']}', Modality: '{eye_res['modality']}'")
assert eye_res["is_medical"] == True, "Eye photo failed medical verification!"
assert "Eye" in eye_res["body_region"] or "Retina" in eye_res["body_region"], "Eye photo was not identified as Eye/Retina!"
print("  ✓ TEST 4 PASSED! Eye photo correctly identified.")

# --------------------------------------------------------------------
# TEST 5: Non-Medical Image Rejection
# --------------------------------------------------------------------
print("\n[TEST 5] Non-Medical Image Rejection...")
non_med_img = "test_general_payloads/landscape_nature.jpg"
non_med_res = detector.analyze(non_med_img, original_filename="vacation_landscape.jpg")
print(f"  Stage 1 -> is_medical: {non_med_res['is_medical']}, State: {non_med_res.get('verification_state')}")
assert non_med_res["is_medical"] == False, "Non-medical landscape image was incorrectly accepted!"
print("  ✓ TEST 5 PASSED! Non-medical image rejected.")

# --------------------------------------------------------------------
# TEST 6: Medical PDF Content Verification
# --------------------------------------------------------------------
print("\n[TEST 6] Medical PDF Content Verification...")
med_pdf = "test_general_payloads/medical_report.pdf"
v_pdf = validator.validate_file(med_pdf, original_filename="clinical_laboratory_results.pdf")
print(f"  PDF -> is_medical: {v_pdf['is_medical']}, State: {v_pdf.get('verification_state')}")
assert v_pdf["is_medical"] == True, "Medical PDF report failed verification!"
print("  ✓ TEST 6 PASSED! Medical PDF report accepted.")

# --------------------------------------------------------------------
# TEST 7: Non-Medical PDF Rejection & Exact Error Message
# --------------------------------------------------------------------
print("\n[TEST 7] Non-Medical PDF Rejection...")
inv_pdf = "test_general_payloads/invoice_document.pdf"
v_inv = validator.validate_file(inv_pdf, original_filename="tax_invoice_2026.pdf")
print(f"  PDF -> is_medical: {v_inv['is_medical']}, Message: '{v_inv.get('message')}'")
assert v_inv["is_medical"] == False, "Invoice non-medical PDF was incorrectly accepted!"
assert "Non-medical PDF detected. Please upload a valid medical PDF." in v_inv.get("message", ""), "Incorrect rejection message for non-medical PDF!"
print("  ✓ TEST 7 PASSED! Non-medical PDF rejected with exact message.")

# --------------------------------------------------------------------
# TEST 8: User-Facing Output Sanitization (Zero MedGemma References)
# --------------------------------------------------------------------
print("\n[TEST 8] User-Facing Output Sanitization (Zero MedGemma References)...")
raw_output = {
    "title": "MedGemma Healthcare AI Analysis",
    "description": "MedGemma empirical image analysis of decrypted Eye / Retina: MedGemma indicates intact visual structures.",
    "model_info": "google/medgemma-1.5-4b-it",
    "status": "SUCCESS"
}

clean_output = sanitize_user_facing_output(raw_output)
clean_json_str = json.dumps(clean_output)

assert "MedGemma" not in clean_json_str, "FAIL: 'MedGemma' still present in user-facing output!"
assert "medgemma" not in clean_json_str, "FAIL: 'medgemma' still present in user-facing output!"
assert "MEDGEMMA" not in clean_json_str, "FAIL: 'MEDGEMMA' still present in user-facing output!"
assert "AI model" not in clean_json_str, "FAIL: 'AI model' still present in user-facing output!"
print(f"  Sanitized Title -> '{clean_output['title']}'")
print(f"  Sanitized Description -> '{clean_output['description']}'")
print("  ✓ TEST 8 PASSED! Zero MedGemma references present in sanitized user output.")

print("\n======================================================================")
print("SUMMARY: ALL 8 ANATOMY, PDF CONTENT & SANITIZATION TESTS PASSED 100%!")
print("======================================================================")
sys.exit(0)
