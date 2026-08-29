"""
Medical Content Validator (Stage 1 Pipeline Integration)
=========================================================
Validation layer that verifies uploaded files (Images & PDFs)
and extracts Stage 1 identification metadata (input_type, body_region, modality, report_type, certainty).
"""

import os
from PIL import Image
from medical_detector import MedicalImageDetector
from .report_processor import MedicalReportProcessor


class MedicalContentValidator:
    """Medical validation layer for images and PDF documents."""

    def __init__(self):
        self.image_detector = MedicalImageDetector()
        self.report_processor = MedicalReportProcessor()

    def validate_file(self, file_path):
        """
        Main validation entry point for uploaded files (Images & PDFs).
        Returns full Stage 1 classification metadata.
        """
        if not os.path.exists(file_path):
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "None",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Invalid File",
                "confidence": 0.0,
                "message": "File not found.",
                "is_pdf": False,
                "extracted_text": ""
            }

        # Check if PDF
        if self.report_processor.is_pdf(file_path):
            return self._validate_pdf(file_path)

        # Image Validation via Vision Classifier
        return self._validate_image(file_path)

    def _validate_pdf(self, pdf_path):
        """Validate PDF document for medical report content and extract Stage 1 metadata."""
        text = self.report_processor.extract_text_from_pdf(pdf_path)
        is_medical_text = self.report_processor.contains_medical_content(text)

        text_lower = text.lower() if text else ""
        body_region = "Systemic / Clinical Document"
        if "brain" in text_lower or "head" in text_lower:
            body_region = "Brain"
        elif "chest" in text_lower or "lung" in text_lower:
            body_region = "Chest / Lungs"
        elif "heart" in text_lower or "cardiac" in text_lower or "ecg" in text_lower:
            body_region = "Heart / Cardiac"
        elif "skin" in text_lower or "derma" in text_lower:
            body_region = "Skin / Dermatology"
        elif "eye" in text_lower or "ophthalm" in text_lower:
            body_region = "Eye / Ophthalmology"
        elif "kidney" in text_lower or "renal" in text_lower:
            body_region = "Kidney / Renal"
        elif "liver" in text_lower or "hepat" in text_lower:
            body_region = "Liver / Hepatobiliary"

        report_type = "Clinical Laboratory Report"
        if "blood" in text_lower or "cbc" in text_lower or "hemoglobin" in text_lower:
            report_type = "Blood Test Laboratory Report"
        elif "x-ray" in text_lower or "mri" in text_lower or "ct" in text_lower or "radiology" in text_lower:
            report_type = "Radiology Diagnostic Report"

        if is_medical_text:
            return {
                "is_medical": True,
                "input_type": "medical_report",
                "body_region": body_region,
                "modality": "Medical Document",
                "report_type": report_type,
                "certainty": "high",
                "medical_type": f"Medical Report ({report_type})",
                "confidence": 95.0,
                "message": "Medical report document verified successfully.",
                "is_pdf": True,
                "extracted_text": text
            }

        # If PDF has no text layer, render preview image and run vision classification
        temp_img_path = pdf_path + "_preview.png"
        try:
            self.report_processor.pdf_to_preview_image(pdf_path, temp_img_path)
            img_result = self.image_detector.analyze(temp_img_path)
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

            if img_result["is_medical"]:
                return {
                    "is_medical": True,
                    "input_type": "medical_report",
                    "body_region": img_result["body_region"],
                    "modality": "Medical Document",
                    "report_type": "Scanned Medical Report",
                    "certainty": img_result["certainty"],
                    "medical_type": f"Scanned Medical Document ({img_result['body_region']})",
                    "confidence": img_result["confidence"],
                    "message": "Medical report document detected successfully.",
                    "is_pdf": True,
                    "extracted_text": text
                }
        except Exception as err:
            print(f"[VALIDATOR] PDF vision validation error: {err}")
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        return {
            "is_medical": False,
            "input_type": "non_medical",
            "body_region": "Not applicable",
            "modality": "Non-medical Document",
            "report_type": None,
            "certainty": "high",
            "medical_type": "Non-Medical Document",
            "confidence": 0.0,
            "message": "This image does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.",
            "is_pdf": True,
            "extracted_text": text
        }

    def _validate_image(self, image_path):
        """Validate image file via Stage 1 vision classifier."""
        try:
            result = self.image_detector.analyze(image_path)
            result["is_pdf"] = False
            result["extracted_text"] = ""
            result["medical_type"] = result["type"]
            return result
        except Exception as error:
            print(f"[VALIDATOR] Image validation error: {error}")
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Error",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Error",
                "confidence": 0.0,
                "message": "This image does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.",
                "is_pdf": False,
                "extracted_text": ""
            }
