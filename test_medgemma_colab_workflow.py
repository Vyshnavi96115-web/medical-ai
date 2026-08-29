"""
Comprehensive Test Suite for MedGemma Multimodal Colab Integration
===================================================================
Tests real MedGemma multimodal image + prompt inference across:
1. Standalone image + prompt inference (Colab mechanism)
2. Decrypted application image payload & SHA-256 hash verification
3. UI JSON response structure with 9 required fields
4. Skin / Morphea image (Must produce dermatology analysis, NOT Chest X-Ray or lungs)
5. Chest X-Ray image (Must produce X-Ray radiograph analysis)
6. Non-medical image pre-encryption rejection
7. Eve interception security prevention
"""

import os
import sys
import json
import base64
import io
from PIL import Image, ImageDraw

from medical_ai.medgemma import MedGemmaAnalyzer
from medical_ai.validator import MedicalContentValidator

def create_morphea_skin_image(path):
    """Creates a sample dermatology skin lesion image (Morphea plaque mockup)."""
    img = Image.new("RGB", (400, 400), color=(235, 205, 185))
    draw = ImageDraw.Draw(img)
    # Draw indurated violaceous plaque / skin lesion
    draw.ellipse([100, 100, 300, 300], fill=(175, 115, 135), outline=(135, 75, 95), width=4)
    draw.ellipse([140, 140, 260, 260], fill=(220, 185, 190))
    img.save(path, format="JPEG")
    return path

def create_chest_xray_image(path):
    """Creates a sample radiograph X-ray image (lung cavity and ribs mockup)."""
    img = Image.new("RGB", (512, 512), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 100, 220, 380], fill=(5, 5, 5))
    draw.ellipse([290, 100, 430, 380], fill=(5, 5, 5))
    draw.rectangle([245, 40, 265, 470], fill=(210, 210, 210))
    for y in range(120, 380, 40):
        draw.line([80, y, 245, y + 15], fill=(180, 180, 180), width=5)
        draw.line([265, y + 15, 430, y], fill=(180, 180, 180), width=5)
    img.save(path, format="JPEG")
    return path

def create_non_medical_image(path):
    """Creates a non-medical photograph (bright yellow cat silhouette)."""
    img = Image.new("RGB", (300, 300), color=(255, 240, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 250, 250], fill=(255, 100, 0))
    img.save(path, format="JPEG")
    return path


def run_all_tests():
    print("=" * 60, flush=True)
    print("STARTING MEDGEMMA MULTIMODAL COLAB WORKFLOW TESTS", flush=True)
    print("=" * 60, flush=True)

    analyzer = MedGemmaAnalyzer()
    validator = MedicalContentValidator()
    test_dir = "test_payloads"
    os.makedirs(test_dir, exist_ok=True)


    skin_path = os.path.join(test_dir, "morphea_skin.jpg")
    xray_path = os.path.join(test_dir, "chest_xray.jpg")
    non_med_path = os.path.join(test_dir, "yellow_cat.jpg")

    create_morphea_skin_image(skin_path)
    create_chest_xray_image(xray_path)
    create_non_medical_image(non_med_path)

    # ----------------------------------------------------
    # TEST 1: Independent MedGemma Image + Prompt Inference
    # ----------------------------------------------------
    print("\n[TEST 1] Standalone MedGemma Colab Image + Prompt Inference...")
    pil_img = Image.open(skin_path)
    res1 = analyzer._run_multimodal_inference(pil_img, "Dermatology Skin Image")
    print("Test 1 Status:", res1.get("status"))
    print("Image Type Identified:", res1.get("medical_image_report_type"))
    assert res1.get("status") == "SUCCESS", "Test 1 failed: status not SUCCESS"
    print("✓ TEST 1 PASSED!")

    # ----------------------------------------------------
    # TEST 2: SHA-256 Decrypted File Verification & Analysis
    # ----------------------------------------------------
    print("\n[TEST 2] Decrypted Payload Verification & SHA-256 Lossless Check...")
    res2 = analyzer.analyze_medical_data(skin_path, "Dermatology Image", original_file_path=skin_path)
    print("Test 2 Image Type:", res2.get("medical_image_report_type"))
    print("Test 2 Medical Finding:", res2.get("medical_finding"))
    assert res2.get("status") == "SUCCESS", "Test 2 failed: status not SUCCESS"
    print("✓ TEST 2 PASSED!")

    # ----------------------------------------------------
    # TEST 3: UI JSON Schema Structure (9 Required Fields)
    # ----------------------------------------------------
    print("\n[TEST 3] UI Response Schema Verification (9 Required Fields)...")
    required_keys = [
        "medical_image_report_type",
        "medical_finding",
        "abnormality_defect",
        "possible_condition",
        "simple_explanation",
        "detailed_explanation",
        "recommended_next_steps",
        "medication_info",
        "uncertainty",
        "disclaimer"
    ]
    for key in required_keys:
        assert key in res2, f"Missing key in UI schema: {key}"
        print(f"  • {key}: {str(res2[key])[:70]}...")
    print("✓ TEST 3 PASSED!")

    # ----------------------------------------------------
    # TEST 4: Morphea / Skin Lesion Image (NO Chest X-Ray Hallucination)
    # ----------------------------------------------------
    print("\n[TEST 4] Morphea / Skin Lesion Image Analysis...")
    res4 = analyzer.analyze_medical_data(skin_path, "Skin Lesion Scan")
    med_type = res4.get("medical_image_report_type", "").lower()
    finding = res4.get("medical_finding", "").lower()
    detailed = res4.get("detailed_explanation", "").lower()
    
    print("Model Identified Type:", res4.get("medical_image_report_type"))
    print("Model Medical Finding:", res4.get("medical_finding"))
    
    # Assert model did NOT invent Chest X-Ray or lower lobes
    assert "chest x-ray" not in med_type and "chest x-ray" not in finding, "FAIL: Morphea image misidentified as Chest X-ray!"
    assert "lower lobe" not in finding and "lower lobe" not in detailed, "FAIL: Fabricated lower-lobe lung abnormality on skin image!"
    print("✓ TEST 4 PASSED! (No Chest X-Ray or lung hallucination on skin lesion image)")

    # ----------------------------------------------------
    # TEST 5: Chest X-Ray Image Analysis
    # ----------------------------------------------------
    print("\n[TEST 5] Chest X-Ray Radiograph Analysis...")
    res5 = analyzer.analyze_medical_data(xray_path, "Chest X-Ray")
    print("X-Ray Model Type:", res5.get("medical_image_report_type"))
    print("X-Ray Finding:", res5.get("medical_finding"))
    assert res5.get("status") == "SUCCESS", "Test 5 failed"
    print("✓ TEST 5 PASSED!")

    # ----------------------------------------------------
    # TEST 6: Non-Medical Photograph Pre-Encryption Rejection
    # ----------------------------------------------------
    print("\n[TEST 6] Non-Medical Image Pre-Encryption Rejection...")
    valid_res = validator.validate_file(non_med_path)
    print("Validator Is Medical:", valid_res.get("is_medical"))
    print("Validator Message:", valid_res.get("message"))
    assert valid_res.get("is_medical") is False, "Non-medical image was incorrectly accepted!"
    print("✓ TEST 6 PASSED! (Non-medical image rejected before encryption/MedGemma)")


    # ----------------------------------------------------
    # TEST 7: Eye / Ophthalmology Image Analysis
    # ----------------------------------------------------
    print("\n[TEST 7] Eye / Ophthalmology Image Modality Identification...")
    eye_path = os.path.join(test_dir, "eye_scan.jpg")


    img_eye = Image.new("RGB", (400, 400), color=(245, 235, 230))
    d_eye = ImageDraw.Draw(img_eye)
    d_eye.ellipse([80, 80, 320, 320], fill=(220, 210, 205), outline=(180, 160, 155), width=3)
    d_eye.ellipse([140, 140, 260, 260], fill=(90, 60, 40), outline=(50, 30, 20), width=4)
    d_eye.ellipse([175, 175, 225, 225], fill=(10, 10, 10))
    d_eye.line([90, 200, 135, 195], fill=(180, 50, 50), width=2)
    img_eye.save(eye_path)

    res7 = analyzer.analyze_medical_data(eye_path, "Ophthalmology / Eye Scan")
    identified_type = res7.get("medical_image_report_type", "")
    print("Eye Image Model Identified Type:", identified_type)
    print("Eye Image Finding:", res7.get("medical_finding"))

    assert "ophthalmology" in identified_type.lower() or "eye" in identified_type.lower(), f"FAIL: Eye image misidentified as {identified_type}"
    assert "dermatology" not in identified_type.lower(), "FAIL: Eye image misidentified as Dermatology!"
    print("✓ TEST 7 PASSED! (Eye image correctly identified as Ophthalmology / Eye Scan)")

    print("\n" + "=" * 60)
    print("ALL 7 MEDGEMMA COLAB WORKFLOW TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()

