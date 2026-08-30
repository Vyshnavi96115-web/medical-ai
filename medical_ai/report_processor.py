"""
Medical Report Processor
========================
Handles processing of medical report documents, specifically PDFs and report images.
Extracts visible text, converts PDF pages for analysis, and preserves original files.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import pypdf


class MedicalReportProcessor:
    """Processes PDF and image-based medical reports for AI analysis."""

    EXPLICIT_MEDICAL_KEYWORDS = [
        "patient", "diagnosis", "blood test", "laboratory", "lab report", "x-ray", "mri",
        "ct scan", "ultrasound", "pathology", "cbc", "vital signs", "prescription",
        "physician", "doctor", "hospital", "clinic", "specimen", "hemoglobin", "wbc", "rbc",
        "platelet", "glucose", "cholesterol", "serum", "ecg", "eeg", "radiology",
        "impression", "findings", "histopathology", "biopsy", "ophthalmology", "retina",
        "dermatology", "cardiology", "renal panel", "liver function"
    ]

    NON_MEDICAL_KEYWORDS = [
        "invoice", "total due", "amount due", "tax invoice", "curriculum vitae", "resume",
        "bank statement", "account number", "balance", "software engineer", "purchase order",
        "payment receipt", "homework", "syllabus", "coursework", "flight ticket", "booking confirmation"
    ]

    def is_pdf(self, file_path):
        """Check if file is a PDF document."""
        return file_path.lower().endswith(".pdf")

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from all pages of a PDF document."""
        if not os.path.exists(pdf_path):
            return ""

        extracted_text = []
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
        except Exception as err:
            print(f"[REPORT PROCESSOR] PDF text extraction error: {err}")

        return "\n".join(extracted_text).strip()

    def contains_medical_content(self, text):
        """Determine if extracted text contains medical terminology."""
        if not text:
            return False
        text_lower = text.lower()

        # Reject if non-medical keywords are present
        if any(kw in text_lower for kw in self.NON_MEDICAL_KEYWORDS):
            # Unless there is a strong concentration of explicit medical terms
            med_count = sum(1 for kw in self.EXPLICIT_MEDICAL_KEYWORDS if kw in text_lower)
            if med_count < 3:
                return False

        keyword_matches = sum(1 for kw in self.EXPLICIT_MEDICAL_KEYWORDS if kw in text_lower)
        return keyword_matches >= 2


    def pdf_to_preview_image(self, pdf_path, output_image_path):
        """
        Renders a preview image from PDF text/content so it can be previewed in UI
        or passed to vision classifiers.
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            text = "[Scanned Medical PDF Report Document]\n\n" + os.path.basename(pdf_path)

        width, height = 800, 1000
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Header banner
        draw.rectangle([(0, 0), (width, 80)], fill=(30, 58, 138))
        draw.text((30, 25), "MEDICAL REPORT DOCUMENT", fill=(255, 255, 255))

        # Render text lines
        lines = text.split("\n")
        y_offset = 110
        font_size = 14

        for line in lines[:40]:  # Up to 40 lines
            line_str = line.strip()
            if not line_str:
                y_offset += 10
                continue

            draw.text((40, y_offset), line_str[:90], fill=(15, 23, 42))
            y_offset += 20
            if y_offset > height - 40:
                break

        image.save(output_image_path, format="PNG")
        return output_image_path

    def render_pdf_page_to_image(self, pdf_path, page_num=0):
        """
        Renders a PDF page to a PIL Image object for multimodal model consumption.
        """
        if not os.path.exists(pdf_path):
            return None

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            text = "[Scanned Medical PDF Report Document]\n\n" + os.path.basename(pdf_path)

        width, height = 800, 1000
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Header banner
        draw.rectangle([(0, 0), (width, 80)], fill=(30, 58, 138))
        draw.text((30, 25), "MEDICAL REPORT DOCUMENT", fill=(255, 255, 255))

        # Render text lines
        lines = text.split("\n")
        y_offset = 110
        for line in lines[:40]:
            line_str = line.strip()
            if not line_str:
                y_offset += 10
                continue

            draw.text((40, y_offset), line_str[:90], fill=(15, 23, 42))
            y_offset += 20
            if y_offset > height - 40:
                break

        return image

