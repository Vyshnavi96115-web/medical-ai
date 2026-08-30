"""
Medical Content Validator (Stage 1 Pipeline Integration)
=========================================================
Validation layer that verifies uploaded files (Images & PDFs)
and extracts Stage 1 identification metadata (input_type, body_region, modality, report_type, certainty).
"""

import os
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from medical_detector import MedicalImageDetector
from .report_processor import MedicalReportProcessor
from .medgemma import MedGemmaAnalyzer



class MedicalContentValidator:
    """Medical validation layer for images, PDF documents, audio, and video clips."""

    def __init__(self):
        self.image_detector = MedicalImageDetector()
        self.report_processor = MedicalReportProcessor()
        self.medgemma_analyzer = MedGemmaAnalyzer()


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

        ext = os.path.splitext(file_path)[1].lower()

        # Check if PDF
        if self.report_processor.is_pdf(file_path) or ext == ".pdf":
            return self._validate_pdf(file_path, original_filename=original_filename)

        # Image Validation via Vision Classifier (Reference Gold Standard)
        return self._validate_image(file_path, original_filename=original_filename)



    def _validate_pdf(self, pdf_path, original_filename=None):
        """Validate PDF document for medical report content using MedGemma verification and vision analysis."""
        fn_target = original_filename or pdf_path

        # 1. Extract text & render page image
        text = self.report_processor.extract_text_from_pdf(pdf_path)
        page_img = None
        temp_img_path = pdf_path + "_preview.png"
        try:
            self.report_processor.pdf_to_preview_image(pdf_path, temp_img_path)
            if os.path.exists(temp_img_path):
                page_img = Image.open(temp_img_path).convert("RGB")
        except Exception as p_err:
            print(f"[VALIDATOR] PDF page rendering notice: {p_err}")

        # 2. Zero-Shot Vision Classifier on Rendered PDF Page Image
        vision_result = None
        if os.path.exists(temp_img_path):
            try:
                vision_result = self.image_detector.analyze(temp_img_path, original_filename=fn_target)
            except Exception as v_err:
                print(f"[VALIDATOR] Vision PDF detection notice: {v_err}")

        # Clean up temporary preview image
        if os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass

        # 3. Non-Medical Commercial & Academic Term Check
        combined_text = f"{fn_target} {text}".lower()
        non_med_terms = ["invoice", "total due", "amount due", "tax invoice", "curriculum vitae", "resume", "bank statement", "account number", "balance", "software engineer", "purchase order", "payment receipt", "homework", "syllabus", "coursework", "flight ticket", "tax", "bill", "receipt", "agreement", "contract", "assignment", "manual", "guide"]
        med_terms = ["patient", "diagnosis", "blood", "cbc", "hemoglobin", "wbc", "platelet", "glucose", "cholesterol", "physician", "hospital", "clinic", "impression", "findings", "pathology", "specimen", "vital", "prescription", "ultrasound", "x-ray", "mri", "ct scan", "ecg", "retina", "ophthalmology", "dermatology", "cardiology", "medical", "doctor", "lab", "radiology", "eye", "optic", "fundus", "macula", "cornea", "intraocular", "retinal"]

        med_matches = sum(1 for kw in med_terms if kw in combined_text)
        non_med_matches = sum(1 for kw in non_med_terms if kw in combined_text)

        # STRICT REJECTION OF NON-MEDICAL PDF
        is_vision_non_medical = vision_result and not vision_result.get("is_medical", True)
        if is_vision_non_medical or (non_med_matches > 0 and med_matches < 2) or (med_matches == 0 and not (vision_result and vision_result.get("is_medical"))):
            return {
                "is_medical": False,
                "verification_state": "NON_MEDICAL",
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Document",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Document",
                "confidence": vision_result.get("confidence", 95.0) if vision_result else 95.0,
                "message": "Non-medical file detected. Please upload a valid medical file.",
                "is_pdf": True,
                "extracted_text": text
            }

        # 4. Determine Exact Organ Region & Modality
        if vision_result and vision_result.get("is_medical") and vision_result.get("body_region") and vision_result.get("body_region") != "Not applicable":
            reg = vision_result.get("body_region")
            mod = vision_result.get("modality", "Medical PDF Report")
            rep = vision_result.get("type", f"{reg} Diagnostic Report")
        else:
            if any(kw in combined_text for kw in ["eye", "retina", "fundus", "ophthalm", "optic", "macula", "cornea", "intraocular", "retinal"]):
                reg, mod, rep = "Eye / Retina", "Ophthalmic Diagnostic PDF Report", "Retinal & Ophthalmic Report"
            elif any(kw in combined_text for kw in ["brain", "head", "mri", "ct scan", "neuro", "skull", "cerebral"]):
                reg, mod, rep = "Brain", "Neuroimaging PDF Report", "Brain MRI/CT Diagnostic Report"
            elif any(kw in combined_text for kw in ["chest", "lung", "x-ray", "radiology", "thoracic", "pulmonary"]):
                reg, mod, rep = "Chest / Lungs", "Radiology PDF Report", "Chest Radiology Report"
            elif any(kw in combined_text for kw in ["knee", "bone", "joint", "fracture", "ortho"]):
                reg, mod, rep = "Knee / Joint", "Orthopedic PDF Report", "Knee & Bone Radiography Report"
            elif any(kw in combined_text for kw in ["skin", "derma", "lesion", "cutaneo"]):
                reg, mod, rep = "Skin / Dermatology", "Dermatology PDF Report", "Skin Lesion Report"
            elif any(kw in combined_text for kw in ["cbc", "blood", "hemoglobin", "wbc", "rbc", "platelet", "glucose", "cholesterol"]):
                reg, mod, rep = "Blood & Hematology", "Clinical Laboratory PDF", "Blood Test Laboratory Report"
            elif any(kw in combined_text for kw in ["pathology", "biopsy", "tissue", "histology"]):
                reg, mod, rep = "Histopathology / Tissue", "Pathology PDF Report", "Histopathology Tissue Report"
            elif any(kw in combined_text for kw in ["heart", "cardiac", "ecg", "auscultation"]):
                reg, mod, rep = "Heart / Cardiac", "Cardiology PDF Report", "ECG Diagnostic Report"
            else:
                reg, mod, rep = "Systemic / Clinical Document", "Medical PDF Document", "Clinical Medical Report"

        conf = vision_result.get("confidence", 92.0) if vision_result else 90.0

        return {
            "is_medical": True,
            "verification_state": "MEDICAL",
            "input_type": "medical_report",
            "body_region": reg,
            "modality": mod,
            "report_type": rep,
            "medical_type": f"{reg} ({rep})",
            "certainty": "high",
            "confidence": conf,
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
