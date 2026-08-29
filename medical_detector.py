"""
Medical Image & Document Detector (Stage 1 General Identification Pipeline)
========================================================================
Performs zero-shot visual classification to dynamically identify:
- Is input medical vs non-medical?
- Input Type: medical_image vs medical_report vs non_medical
- Anatomical Region: Brain, Chest/Lungs, Heart, Skin, Eye, Kidney, Liver, Bone, Knee, Spine, Abdomen, Histopathology, etc., or Unknown
- Imaging Modality: X-Ray, CT Scan, MRI, Ultrasound, Clinical Photograph, Dermoscopy, Fundus Photograph, Microscopy, ECG, Medical Document, etc.
- Certainty Score: high, medium, low, unknown
"""

import os
from PIL import Image
from transformers import pipeline


class MedicalImageDetector:

    def __init__(self):
        self._classifier = None

    @property
    def classifier(self):
        if self._classifier is None:
            print("Loading general medical image detection model...")
            self._classifier = pipeline(
                "zero-shot-image-classification",
                model="openai/clip-vit-base-patch32"
            )
            print("General medical image detection model ready.")
        return self._classifier

    def detect_medical_image(self, image_path):
        return self.analyze(image_path)


    def analyze(self, image_path):

        """
        Executes Stage 1 General Input Identification.
        Does NOT rely on fixed organ defaults or hardcoded assumptions.
        """
        image = Image.open(image_path).convert("RGB")

        # ----------------------------------------------------
        # 1. SCREENING: Medical vs Non-Medical
        # ----------------------------------------------------
        screening_medical = [
            "an X-ray, CT or MRI scan",
            "a diagnostic radiology image or ultrasound",
            "a clinical photograph of skin, eye or body part",
            "a fundus photograph or ophthalmology scan",
            "a medical document or report",
            "a histopathology tissue slide"
        ]
        screening_non_medical = [
            "a normal photograph of a pet or animal",
            "a cartoon graphic or illustration",
            "a landscape or nature photograph",
            "a selfie or portrait photograph",
            "an everyday non-medical object"
        ]

        screening_results = self.classifier(
            image,
            candidate_labels=screening_medical + screening_non_medical
        )
        scores = {res["label"]: res["score"] for res in screening_results}
        max_med = max(scores.get(lbl, 0.0) for lbl in screening_medical)
        max_non_med = max(scores.get(lbl, 0.0) for lbl in screening_non_medical)

        is_medical = (max_med >= 0.08 or max_med >= max_non_med * 0.60)
        if max_non_med > 0.60 and max_non_med > max_med * 2.5:
            is_medical = False







        if not is_medical:
            return {
                "is_medical": False,
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical photograph",
                "report_type": None,
                "certainty": "high" if max_non_med > 0.4 else "medium",
                "confidence": round(max_non_med * 100, 2),
                "type": "Non-Medical Image",
                "message": "This image does not appear to be a medical image or medical report. Please upload a valid medical scan or medical report."
            }

        # ----------------------------------------------------
        # 2. INPUT TYPE: Document vs Image
        # ----------------------------------------------------
        doc_labels = [
            "a printed medical report text document",
            "a laboratory result sheet",
            "a hospital clinical report document",
            "a medical diagnostic scan image"
        ]
        doc_results = self.classifier(image, candidate_labels=doc_labels)
        best_doc_label = doc_results[0]["label"]
        best_doc_score = doc_results[0]["score"]

        is_document = "document" in best_doc_label or "sheet" in best_doc_label or "text" in best_doc_label
        input_type = "medical_report" if is_document else "medical_image"

        # ----------------------------------------------------
        # 3. MODALITY IDENTIFICATION (DYNAMIC)
        # ----------------------------------------------------
        modality_labels = [
            "an X-ray radiograph",
            "a CT scan",
            "an MRI scan",
            "an ultrasound sonogram",
            "an echocardiogram",
            "a clinical photograph",
            "a dermoscopy image",
            "a fundus photograph of an eye",
            "a microscopy histopathology slide",
            "an ECG trace graph",
            "a medical report document"
        ]
        mod_results = self.classifier(image, candidate_labels=modality_labels)
        top_mod_label = mod_results[0]["label"]
        top_mod_score = mod_results[0]["score"]

        modality_map = {
            "X-ray": "X-Ray / Radiograph",
            "CT": "CT Scan",
            "MRI": "MRI Scan",
            "ultrasound": "Ultrasound / Sonogram",
            "echocardiogram": "Echocardiogram",
            "clinical photograph": "Clinical Photograph",
            "dermoscopy": "Dermoscopy",
            "fundus": "Fundus Photograph / Ophthalmoscopy",
            "microscopy": "Microscopy / Histopathology",
            "ECG": "ECG Trace",
            "report document": "Medical Document"
        }
        modality = "Medical Diagnostic Image"
        for key, val in modality_map.items():
            if key.lower() in top_mod_label.lower():
                modality = val
                break

        # ----------------------------------------------------
        # 4. ANATOMICAL BODY REGION IDENTIFICATION (DYNAMIC)
        # ----------------------------------------------------
        region_labels = [
            "brain or head",
            "chest or lungs",
            "heart or cardiac",
            "skin or lesion",
            "eye or ocular",
            "kidney or renal",
            "liver or hepatobiliary",
            "bone or skeleton",
            "knee or joint",
            "spine or vertebra",
            "abdomen or stomach",
            "breast or mammogram",
            "histopathology tissue slide",
            "blood report document"
        ]
        region_results = self.classifier(image, candidate_labels=region_labels)
        top_reg_label = region_results[0]["label"]
        top_reg_score = region_results[0]["score"]

        region_map = {
            "brain": "Brain",
            "chest": "Chest / Lungs",
            "heart": "Heart / Cardiac",
            "skin": "Skin / Dermatology",
            "eye": "Eye / Ophthalmology",
            "kidney": "Kidney / Renal",
            "liver": "Liver / Hepatobiliary",
            "bone": "Bone / Skeletal",
            "knee": "Knee / Joint",
            "spine": "Spine",
            "abdomen": "Abdomen",
            "breast": "Breast",
            "tissue": "Histopathology Tissue",
            "blood": "Systemic / Blood"
        }

        body_region = "Unknown / Unable to determine"
        if top_reg_score >= 0.12:
            for key, val in region_map.items():
                if key in top_reg_label.lower():
                    body_region = val
                    break

        if input_type == "medical_report" and body_region == "Unknown / Unable to determine":
            body_region = "Systemic / Clinical Document"

        # ----------------------------------------------------
        # 5. CERTAINTY CALCULATION
        # ----------------------------------------------------
        if top_mod_score >= 0.30 and (top_reg_score >= 0.20 or is_document):
            certainty = "high"
        elif top_mod_score >= 0.15 or top_reg_score >= 0.12:
            certainty = "medium"
        elif body_region == "Unknown / Unable to determine":
            certainty = "unknown"
        else:
            certainty = "low"

        report_type = None
        if input_type == "medical_report":
            if "blood" in top_reg_label.lower() or "laboratory" in best_doc_label.lower():
                report_type = "Laboratory Report"
            else:
                report_type = "Medical Diagnostic Report"


        display_type = f"{body_region} ({modality})" if body_region != "Unknown / Unable to determine" else modality

        return {
            "is_medical": True,
            "input_type": input_type,
            "body_region": body_region,
            "modality": modality,
            "report_type": report_type,
            "certainty": certainty,
            "confidence": round(max_med * 100, 2),
            "type": display_type,
            "message": f"Medical content verified ({display_type})."
        }