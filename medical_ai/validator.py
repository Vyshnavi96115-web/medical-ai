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

    def validate_file(self, file_path, original_filename=None):
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
            return self._validate_pdf(file_path, original_filename=original_filename)

        # Image Validation via Vision Classifier
        return self._validate_image(file_path, original_filename=original_filename)

    def _validate_pdf(self, pdf_path, original_filename=None):
        """Validate PDF document for medical report content and extract Stage 1 metadata."""
        fn_target = original_filename or pdf_path
        fn_lower = fn_target.lower()

        # 1. Filename Non-Medical Indicator Check
        non_med_kws = ["invoice", "resume", "cv", "tax", "bill", "receipt", "statement", "contract", "homework", "assignment", "ticket", "passport", "manual", "novel", "ebook", "attack", "titan", "anime", "wallpaper", "cat", "dog", "car", "portrait", "selfie"]
        med_kws = ["eye", "retina", "fundus", "xray", "x-ray", "mri", "ct", "scan", "lesion", "skin", "ultrasound", "ecg", "report", "lab", "blood", "cbc", "patient", "doctor", "clinic", "hospital", "medical", "pathology", "radiology"]

        is_filename_non_med = any(kw in fn_lower for kw in non_med_kws) and not any(kw in fn_lower for kw in med_kws)
        if is_filename_non_med:
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Document",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Document",
                "confidence": 0.0,
                "message": "This PDF document does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.",
                "is_pdf": True,
                "extracted_text": ""
            }

        # 2. Extract Text & Check Clinical Terminology
        text = self.report_processor.extract_text_from_pdf(pdf_path)
        text_lower = text.lower() if text else ""

        non_med_text = ["invoice", "total due", "amount due", "tax invoice", "curriculum vitae", "bank statement", "account number", "balance", "software engineer", "purchase order", "payment receipt"]
        med_terms = ["patient", "diagnosis", "laboratory", "radiology", "blood", "cbc", "hemoglobin", "wbc", "platelet", "glucose", "cholesterol", "physician", "hospital", "clinic", "impression", "findings", "pathology", "specimen", "vital", "prescription", "ultrasound", "x-ray", "mri", "ct scan", "ecg", "retina", "ophthalmology"]

        matched_med = sum(1 for kw in med_terms if kw in text_lower)
        matched_non_med = sum(1 for kw in non_med_text if kw in text_lower)

        if matched_non_med > 0 and matched_med < 2:
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Document",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Document",
                "confidence": 0.0,
                "message": "This PDF document does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.",
                "is_pdf": True,
                "extracted_text": text
            }

        # 3. Vision Preview Classifier fallback if text is sparse or PDF contains embedded scan
        temp_img_path = pdf_path + "_preview.png"
        img_result = None
        try:
            self.report_processor.pdf_to_preview_image(pdf_path, temp_img_path)
            img_result = self.image_detector.analyze(temp_img_path, original_filename=original_filename)
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
        except Exception as err:
            print(f"[VALIDATOR] PDF vision validation notice: {err}")
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        has_medical_indicators = (matched_med >= 2) or any(kw in fn_lower for kw in med_kws) or (img_result and img_result.get("is_medical"))

        if not has_medical_indicators:
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Document",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Document",
                "confidence": 0.0,
                "message": "This PDF document does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report.",
                "is_pdf": True,
                "extracted_text": text
            }

        # 4. Identify specific PDF report type & organ:
        if "eye" in text_lower or "retina" in text_lower or "fundus" in text_lower or "ophthalm" in text_lower or "eye" in fn_lower:
            reg, mod, rep = "Eye / Retina", "Ophthalmic Diagnostic PDF Report", "Retinal & Ophthalmic Report"
        elif "brain" in text_lower or "head" in text_lower or "mri" in text_lower or "ct" in text_lower or "brain" in fn_lower:
            reg, mod, rep = "Brain", "Neuroimaging PDF Report", "Brain MRI/CT Diagnostic Report"
        elif "chest" in text_lower or "lung" in text_lower or "x-ray" in text_lower or "radiology" in text_lower or "chest" in fn_lower:
            reg, mod, rep = "Chest / Lungs", "Radiology PDF Report", "Chest Radiology Report"
        elif "skin" in text_lower or "derma" in text_lower or "lesion" in text_lower or "skin" in fn_lower:
            reg, mod, rep = "Skin / Dermatology", "Dermatology PDF Report", "Skin Lesion Report"
        elif "cbc" in text_lower or "blood" in text_lower or "hemoglobin" in text_lower or "blood" in fn_lower:
            reg, mod, rep = "Blood & Hematology", "Clinical Laboratory PDF", "Blood Test Laboratory Report"
        elif "pathology" in text_lower or "biopsy" in text_lower or "tissue" in text_lower:
            reg, mod, rep = "Histopathology / Tissue", "Pathology PDF Report", "Histopathology Tissue Report"
        elif "heart" in text_lower or "cardiac" in text_lower or "ecg" in text_lower:
            reg, mod, rep = "Heart / Cardiac", "Cardiology PDF Report", "ECG Diagnostic Report"
        else:
            reg, mod, rep = "Systemic / Clinical Document", "Medical PDF Document", "Clinical Laboratory Report"

        return {
            "is_medical": True,
            "input_type": "medical_report",
            "body_region": reg,
            "modality": mod,
            "report_type": rep,
            "medical_type": f"{reg} ({rep})",
            "certainty": "high",
            "confidence": 95.0,
            "message": f"Medical PDF report ({rep}) verified successfully.",
            "is_pdf": True,
            "extracted_text": text
        }


    def _validate_image(self, image_path, original_filename=None):
        """Validate image file via Stage 1 vision classifier."""
        try:
            result = self.image_detector.analyze(image_path, original_filename=original_filename)
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
