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

        # Check if Audio
        if ext in {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"}:
            return self._validate_audio(file_path, original_filename=original_filename)

        # Check if Video
        if ext in {".mp4", ".avi", ".mov", ".webm", ".mkv"}:
            return self._validate_video(file_path, original_filename=original_filename)

        # Check if PDF
        if self.report_processor.is_pdf(file_path):
            return self._validate_pdf(file_path, original_filename=original_filename)

        # Image Validation via Vision Classifier
        return self._validate_image(file_path, original_filename=original_filename)

    def _validate_audio(self, audio_path, original_filename=None):
        """Validate medical auscultation/phonocardiogram audio files using MedGemma verification."""
        fn_target = original_filename or audio_path
        fn_lower = fn_target.lower()

        # Transcript extraction / Filename context
        transcript = f"Audio Recording File: {os.path.basename(fn_target)}"
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            if audio_path.lower().endswith(".wav"):
                with sr.AudioFile(audio_path) as source:
                    audio_data = r.record(source, duration=30)
                    transcript += "\nTranscribed Audio Content:\n" + r.recognize_google(audio_data)
        except Exception:
            pass

        # MedGemma Verification
        ver = self.medgemma_analyzer.verify_medical_content_with_medgemma(None, media_type="audio", additional_text=transcript)
        state = ver.get("state", "MEDICAL")
        conf = float(ver.get("confidence", 90.0))
        reason = ver.get("reason", "")

        if state == "NON_MEDICAL" or conf < 35.0:
            return {
                "is_medical": False,
                "verification_state": "NON_MEDICAL",
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Audio",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Audio",
                "confidence": conf,
                "message": "Non-medical file detected. Please upload a valid medical file.",
                "is_pdf": False,
                "extracted_text": ""
            }

        if state == "UNCLEAR" or conf < 70.0:
            return {
                "is_medical": False,
                "verification_state": "UNCLEAR",
                "input_type": "unclear",
                "body_region": "Not applicable",
                "modality": "Unclear Audio Payload",
                "report_type": None,
                "certainty": "low",
                "medical_type": "Unclear Audio",
                "confidence": conf,
                "message": "Unable to verify this file as a medical file. Please upload a clearer medical file.",
                "is_pdf": False,
                "extracted_text": ""
            }

        return {
            "is_medical": True,
            "verification_state": "MEDICAL",
            "input_type": "medical_audio",
            "body_region": "Heart / Lungs / Phonocardiogram",
            "modality": "Stethoscope / Auscultation Audio",
            "report_type": None,
            "certainty": "high",
            "medical_type": "Medical Audio (Stethoscope / Auscultation Audio)",
            "confidence": conf,
            "message": "Medical stethoscope / auscultation audio verified successfully.",
            "is_pdf": False,
            "extracted_text": transcript
        }

    def _validate_video(self, video_path, original_filename=None):
        """Validate medical ultrasound/endoscopy video clips using MedGemma verification."""
        fn_target = original_filename or video_path
        fn_lower = fn_target.lower()

        # Filename Non-Medical Indicator Check
        non_med_video_kws = ["movie", "film", "anime", "cartoon", "trailer", "gameplay", "tiktok", "vlog", "vimeo", "youtube", "cinema", "show", "travel", "vacation", "trip", "song", "music", "dance", "cat", "dog", "car", "sports", "game", "wedding", "party", "funny", "family", "personal", "nature"]
        med_video_kws = ["ultrasound", "echo", "echocardiogram", "endoscopy", "laparoscopy", "doppler", "cardiac", "motion", "scan", "video", "clip", "recording", "surgery", "clinic", "hospital", "medical", "patient", "procedure", "colonoscopy", "angioplasty", "fluoroscopy"]

        is_non_med_fn = any(kw in fn_lower for kw in non_med_video_kws) and not any(kw in fn_lower for kw in med_video_kws)
        if is_non_med_fn:
            return {
                "is_medical": False,
                "verification_state": "NON_MEDICAL",
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Video",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Video",
                "confidence": 95.0,
                "message": "Non-medical file detected. Please upload a valid medical file.",
                "is_pdf": False,
                "extracted_text": ""
            }

        grid_img = None
        if cv2 is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    indices = [int(total_frames * i / 6) for i in range(6)]
                    extracted_frames = []
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_f = Image.fromarray(rgb).resize((250, 250))
                            extracted_frames.append(pil_f)
                    cap.release()

                    if len(extracted_frames) == 6:
                        grid_img = Image.new("RGB", (750, 500), color=(0, 0, 0))
                        grid_img.paste(extracted_frames[0], (0, 0))
                        grid_img.paste(extracted_frames[1], (250, 0))
                        grid_img.paste(extracted_frames[2], (500, 0))
                        grid_img.paste(extracted_frames[3], (0, 250))
                        grid_img.paste(extracted_frames[4], (250, 250))
                        grid_img.paste(extracted_frames[5], (500, 250))
            except Exception as vid_err:
                print(f"[VALIDATOR] Video frame extraction notice: {vid_err}")

        # MedGemma Verification
        ver = self.medgemma_analyzer.verify_medical_content_with_medgemma(grid_img, media_type="video", additional_text=f"Video Filename: {os.path.basename(fn_target)}")
        state = ver.get("state", "MEDICAL")
        conf = float(ver.get("confidence", 90.0))

        if state == "NON_MEDICAL" or conf < 35.0:
            return {
                "is_medical": False,
                "verification_state": "NON_MEDICAL",
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Video",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Video",
                "confidence": conf,
                "message": "Non-medical file detected. Please upload a valid medical file.",
                "is_pdf": False,
                "extracted_text": ""
            }

        if state == "UNCLEAR" or conf < 70.0:
            return {
                "is_medical": False,
                "verification_state": "UNCLEAR",
                "input_type": "unclear",
                "body_region": "Not applicable",
                "modality": "Unclear Video Payload",
                "report_type": None,
                "certainty": "low",
                "medical_type": "Unclear Video",
                "confidence": conf,
                "message": "Unable to verify this file as a medical file. Please upload a clearer medical file.",
                "is_pdf": False,
                "extracted_text": ""
            }

        return {
            "is_medical": True,
            "verification_state": "MEDICAL",
            "input_type": "medical_video",
            "body_region": "Cardiac / Endoscopy / Ultrasound",
            "modality": "Ultrasound / Endoscopy Video",
            "report_type": None,
            "certainty": "high",
            "medical_type": "Medical Video (Ultrasound / Endoscopy Clip)",
            "confidence": conf,
            "message": "Medical ultrasound / endoscopy video clip verified successfully.",
            "is_pdf": False,
            "extracted_text": ""
        }


    def _validate_pdf(self, pdf_path, original_filename=None):
        """Validate PDF document for medical report content using MedGemma verification."""
        fn_target = original_filename or pdf_path

        # 1. Extract text & render page image
        text = self.report_processor.extract_text_from_pdf(pdf_path)
        page_img = None
        temp_img_path = pdf_path + "_preview.png"
        try:
            self.report_processor.pdf_to_preview_image(pdf_path, temp_img_path)
            if os.path.exists(temp_img_path):
                page_img = Image.open(temp_img_path).convert("RGB")
                os.remove(temp_img_path)
        except Exception:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        # 2. MedGemma Verification
        ver = self.medgemma_analyzer.verify_medical_content_with_medgemma(page_img, media_type="pdf", additional_text=f"Document Filename: {fn_target}\n\n{text}")

        state = ver.get("state", "MEDICAL")
        conf = float(ver.get("confidence", 95.0))

        if state == "NON_MEDICAL" or conf < 35.0:
            return {
                "is_medical": False,
                "verification_state": "NON_MEDICAL",
                "input_type": "non_medical",
                "body_region": "Not applicable",
                "modality": "Non-medical Document",
                "report_type": None,
                "certainty": "high",
                "medical_type": "Non-Medical Document",
                "confidence": conf,
                "message": "Non-medical file detected. Please upload a valid medical file.",
                "is_pdf": True,
                "extracted_text": text
            }

        if state == "UNCLEAR" or conf < 70.0:
            return {
                "is_medical": False,
                "verification_state": "UNCLEAR",
                "input_type": "unclear",
                "body_region": "Not applicable",
                "modality": "Unclear Document Payload",
                "report_type": None,
                "certainty": "low",
                "medical_type": "Unclear PDF Document",
                "confidence": conf,
                "message": "Unable to verify this file as a medical file. Please upload a clearer medical file.",
                "is_pdf": True,
                "extracted_text": text
            }

        text_lower = (text or "").lower()
        if "eye" in text_lower or "retina" in text_lower or "fundus" in text_lower or "ophthalm" in text_lower or "eye" in fn_target.lower():
            reg, mod, rep = "Eye / Retina", "Ophthalmic Diagnostic PDF Report", "Retinal & Ophthalmic Report"
        elif "brain" in text_lower or "head" in text_lower or "mri" in text_lower or "ct" in text_lower or "brain" in fn_target.lower():
            reg, mod, rep = "Brain", "Neuroimaging PDF Report", "Brain MRI/CT Diagnostic Report"
        elif "chest" in text_lower or "lung" in text_lower or "x-ray" in text_lower or "radiology" in text_lower or "chest" in fn_target.lower():
            reg, mod, rep = "Chest / Lungs", "Radiology PDF Report", "Chest Radiology Report"
        elif "skin" in text_lower or "derma" in text_lower or "lesion" in text_lower or "skin" in fn_target.lower():
            reg, mod, rep = "Skin / Dermatology", "Dermatology PDF Report", "Skin Lesion Report"
        elif "cbc" in text_lower or "blood" in text_lower or "hemoglobin" in text_lower or "blood" in fn_target.lower():
            reg, mod, rep = "Blood & Hematology", "Clinical Laboratory PDF", "Blood Test Laboratory Report"
        elif "pathology" in text_lower or "biopsy" in text_lower or "tissue" in text_lower:
            reg, mod, rep = "Histopathology / Tissue", "Pathology PDF Report", "Histopathology Tissue Report"
        elif "heart" in text_lower or "cardiac" in text_lower or "ecg" in text_lower:
            reg, mod, rep = "Heart / Cardiac", "Cardiology PDF Report", "ECG Diagnostic Report"
        else:
            reg, mod, rep = "Systemic / Clinical Document", "Medical PDF Document", "Clinical Laboratory Report"

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
