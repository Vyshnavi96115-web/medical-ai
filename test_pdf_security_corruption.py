"""
Comprehensive Test Suite for PDF Medical Verification & Eve Interception Sentence Corruption
"""

import sys
import os
sys.path.insert(0, '.')

from medical_ai.validator import MedicalContentValidator
from app import create_corrupted_pdf

validator = MedicalContentValidator()

print("======================================================================")
print("RUNNING PDF MEDICAL VERIFICATION & EVE SENTENCE CORRUPTION TEST SUITE")
print("======================================================================")

# 1. Verification Tests
med_pdf = "test_general_payloads/medical_report.pdf"
inv_pdf = "test_general_payloads/invoice_document.pdf"

res_med = validator.validate_file(med_pdf, original_filename="clinical_lab_report.pdf")
print(f"\n[TEST 1] Medical PDF -> is_medical: {res_med['is_medical']}, state: {res_med.get('verification_state')}")
assert res_med["is_medical"] == True, "Medical PDF failed verification!"

res_inv = validator.validate_file(inv_pdf, original_filename="tax_invoice_july.pdf")
print(f"\n[TEST 2] Invoice Non-Medical PDF -> is_medical: {res_inv['is_medical']}, state: {res_inv.get('verification_state')}")
assert res_inv["is_medical"] == False, "Non-medical PDF was incorrectly accepted!"

# 2. Eve Sentence Corruption Test
corrupted_pdf_fn = create_corrupted_pdf(med_pdf, qber=45.0)
corrupted_pdf_path = os.path.join("uploads", corrupted_pdf_fn)
print(f"\n[TEST 3] Eve ON -> Corrupted PDF generated at: {corrupted_pdf_path}")
assert os.path.exists(corrupted_pdf_path), "Corrupted PDF file was not created!"
assert os.path.getsize(corrupted_pdf_path) > 100, "Corrupted PDF file is empty!"

print("\n======================================================================")
print("SUMMARY: ALL PDF MEDICAL VERIFICATION & EVE CORRUPTION TESTS PASSED!")
print("======================================================================")
sys.exit(0)
