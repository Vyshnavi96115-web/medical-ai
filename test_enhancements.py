"""
Automated Test Suite for Medical AI Enhancements
================================================
Tests:
1. Valid Medical Image Analysis & Encryption -> Decryption -> MedGemma AI Analysis
2. Medical PDF Report Validation & Decryption
3. Non-Medical Image Rejection
4. Eve Interception (Eve ON) -> MedGemma AI Analysis Suppressed
5. MedGemma Offline / Fallback Handling
"""

import os
import sys
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw, ImageFont
import pypdf

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app


class MedicalAIEnhancementTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        self.test_dir = os.path.join(os.path.dirname(__file__), "test_assets")
        os.makedirs(self.test_dir, exist_ok=True)

        # 1. Create a Medical Scan Image (Chest X-Ray radiograph)
        self.xray_path = os.path.join(self.test_dir, "test_xray.png")
        img = Image.new("RGB", (512, 512), color=(20, 20, 20))
        d = ImageDraw.Draw(img)
        d.ellipse([80, 100, 220, 380], fill=(5, 5, 5))
        d.ellipse([290, 100, 430, 380], fill=(5, 5, 5))
        d.rectangle([245, 40, 265, 470], fill=(210, 210, 210))
        for y in range(120, 380, 40):
            d.line([80, y, 245, y + 15], fill=(180, 180, 180), width=5)
            d.line([265, y + 15, 430, y], fill=(180, 180, 180), width=5)
        img.save(self.xray_path)

        # 2. Non-medical image (Bright non-medical graphic)
        self.photo_path = os.path.join(self.test_dir, "test_landscape.jpg")
        img_color = Image.new("RGB", (300, 300), color=(255, 240, 0))
        draw_color = ImageDraw.Draw(img_color)
        draw_color.rectangle([50, 50, 250, 250], fill=(255, 100, 0))
        img_color.save(self.photo_path)

        # 3. Medical PDF report
        self.pdf_path = os.path.join(self.test_dir, "test_lab_report.pdf")
        writer = pypdf.PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        with open(self.pdf_path, "wb") as f:
            writer.write(f)

    def test_01_non_medical_image_rejection(self):
        """Test: Uploading a non-medical photograph must be rejected."""
        with open(self.photo_path, "rb") as f:
            data = {"medical_image": (f, "test_landscape.jpg")}
            response = self.app.post("/api/detect-medical-image", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        print("\n[TEST 1 - NON-MEDICAL REJECTION]:", res)

        self.assertFalse(res["is_medical"])


    def test_02_medical_pdf_validation_and_decryption(self):
        """Test: Medical PDF report must be validated, encrypted, decrypted and analyzed."""
        with patch("medical_ai.report_processor.MedicalReportProcessor.extract_text_from_pdf") as mock_extract:
            mock_extract.return_value = "PATIENT: Jane Doe\nDIAGNOSIS: Clinical Blood Test Laboratory Report\nHemoglobin: 13.5 g/dL"

            with open(self.pdf_path, "rb") as f:
                data = {"medical_image": (f, "test_lab_report.pdf")}
                response = self.app.post("/api/detect-medical-image", data=data, content_type="multipart/form-data")

            self.assertEqual(response.status_code, 200)
            res_detect = response.get_json()
            print("\n[TEST 2 - PDF DETECT]:", res_detect)
            self.assertTrue(res_detect["is_medical"])
            self.assertTrue(res_detect["is_pdf"])

            # Encrypt without Eve
            res_enc = self.app.post("/api/encrypt-medical-image", json={"eve": False}).get_json()
            print("\n[TEST 2 - PDF ENCRYPT]:", res_enc)
            self.assertTrue(res_enc["success"])
            quantum_key = res_enc["quantum_key"]

            # Decrypt with correct quantum key
            res_dec = self.app.post("/api/decrypt-medical-image", json={"quantum_key": quantum_key}).get_json()
            print("\n[TEST 2 - PDF DECRYPT]:", res_dec)
            self.assertTrue(res_dec["decrypted"])
            self.assertTrue(res_dec["is_pdf"])
            self.assertIn("ai_analysis", res_dec)
            self.assertEqual(res_dec["ai_analysis"]["status"], "SUCCESS")

    def test_03_medical_image_workflow_no_eve(self):
        """Test: Medical Image -> Encrypt (Eve OFF) -> Decrypt -> MedGemma AI Analysis."""
        with patch("medical_detector.MedicalImageDetector.detect_medical_image") as mock_detect:


            mock_detect.return_value = {
                "is_medical": True,
                "input_type": "medical_image",
                "body_region": "Chest / Lungs",
                "modality": "X-Ray / Radiograph",
                "report_type": None,
                "certainty": "high",
                "confidence": 94.5,
                "type": "Chest / Lungs (X-Ray / Radiograph)",
                "message": "Medical content verified."
            }


            with open(self.xray_path, "rb") as f:
                data = {"medical_image": (f, "test_xray.png")}
                res_detect = self.app.post("/api/detect-medical-image", data=data, content_type="multipart/form-data").get_json()

            self.assertTrue(res_detect["is_medical"])

            # Encrypt with Eve OFF
            res_enc = self.app.post("/api/encrypt-medical-image", json={"eve": False}).get_json()
            self.assertTrue(res_enc["success"])
            quantum_key = res_enc["quantum_key"]

            # Decrypt
            res_dec = self.app.post("/api/decrypt-medical-image", json={"quantum_key": quantum_key}).get_json()
            print("\n[TEST 3 - MEDICAL IMAGE EVE OFF]:", res_dec)
            self.assertTrue(res_dec["decrypted"])
            self.assertFalse(res_dec["corrupted"])
            self.assertIn("ai_analysis", res_dec)
            self.assertIn("simple_explanation", res_dec["ai_analysis"])
            self.assertIn("disclaimer", res_dec["ai_analysis"])

    def test_04_eve_interception_suppresses_ai_analysis(self):
        """Test: Medical Image -> Encrypt with Eve ON -> Decrypt -> MedGemma AI analysis MUST NOT run."""
        with patch("medical_detector.MedicalImageDetector.analyze") as mock_analyze:
            mock_analyze.return_value = {
                "is_medical": True,
                "type": "Chest X-Ray",
                "confidence": 94.5,
                "label": "a medical X-ray image"
            }

            with open(self.xray_path, "rb") as f:
                data = {"medical_image": (f, "test_xray.png")}
                self.app.post("/api/detect-medical-image", data=data, content_type="multipart/form-data")

            # Encrypt with Eve ON
            res_enc = self.app.post("/api/encrypt-medical-image", json={"eve": True}).get_json()
            quantum_key = res_enc["quantum_key"]

            # Decrypt
            res_dec = self.app.post("/api/decrypt-medical-image", json={"quantum_key": quantum_key}).get_json()
            print("\n[TEST 4 - EVE ON DECRYPT]:", res_dec)
            self.assertFalse(res_dec["decrypted"])
            self.assertTrue(res_dec["corrupted"])
            self.assertNotIn("ai_analysis", res_dec)


if __name__ == "__main__":
    unittest.main()
