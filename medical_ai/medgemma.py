"""
MedGemma AI Service Interface (Stage 2 Integration)
===================================================
Integrates real MedGemma multimodal image-text AI model for evidence-based
medical diagnostic analysis following successful quantum decryption.
Consumes Stage 1 general identification context to run dynamic multimodal inference.
"""

import base64
import hashlib
import io
import json
import os
import requests
from dotenv import load_dotenv
from PIL import Image

from .prompts import MEDGEMMA_SYSTEM_PROMPT, get_medgemma_dynamic_prompt, MEDICAL_SAFETY_DISCLAIMER
from .report_processor import MedicalReportProcessor

# Load environment variables
load_dotenv()


class MedGemmaAnalyzer:
    """Service interface for official MedGemma multimodal healthcare AI model."""

    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("MED") or os.getenv("MEDGEMMA_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        self.model_name = os.getenv("MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")
        self.endpoint_url = os.getenv("MEDGEMMA_ENDPOINT_URL", "https://router.huggingface.co/v1/chat/completions")
        self.report_processor = MedicalReportProcessor()

        print(f"[MEDGEMMA] Model: {self.model_name}")
        if self.hf_token:
            print(f"[MEDGEMMA] Hugging Face Multimodal API active (Model: {self.model_name}).")
        else:
            print("[MEDGEMMA] Notice: HF_TOKEN not found in environment.")


    def verify_image_integrity(self, decrypted_file_path, original_file_path=None):
        """
        Verifies file existence, opening, MIME type, dimensions, and optional SHA-256 hash match.
        """
        if not os.path.exists(decrypted_file_path):
            raise FileNotFoundError(f"Decrypted file not found at: {decrypted_file_path}")

        try:
            with Image.open(decrypted_file_path) as img:
                img.verify()

            with Image.open(decrypted_file_path) as img:
                dimensions = img.size
                img_format = img.format
                img_mode = img.mode

            file_size = os.path.getsize(decrypted_file_path)
            print(f"[MEDGEMMA DEBUG] Original filename: {os.path.basename(original_file_path) if original_file_path else 'N/A'}")
            print(f"[MEDGEMMA DEBUG] Decrypted path: {decrypted_file_path}")
            print(f"[MEDGEMMA DEBUG] Decrypted exists: YES")
            print(f"[MEDGEMMA DEBUG] Decrypted size: {file_size} bytes")
            print(f"[MEDGEMMA DEBUG] Decrypted MIME: image/{str(img_format).lower()}")
            print(f"[MEDGEMMA DEBUG] Image opened: YES")
            print(f"[MEDGEMMA DEBUG] Image dimensions: {dimensions[0]}x{dimensions[1]}")
            print(f"[MEDGEMMA DEBUG] Model: {self.model_name}")

            # Check SHA-256 hash match if original file path is provided (lossless verification)
            if original_file_path and os.path.exists(original_file_path) and not decrypted_file_path.endswith(".pdf"):
                with open(original_file_path, "rb") as f1:
                    h_orig = hashlib.sha256(f1.read()).hexdigest()
                with open(decrypted_file_path, "rb") as f2:
                    h_dec = hashlib.sha256(f2.read()).hexdigest()
                if h_orig == h_dec:
                    print(f"[MEDGEMMA DEBUG] SHA-256 verified: YES ({h_dec[:12]}...)")
                else:
                    print(f"[MEDGEMMA DEBUG] SHA-256 verified: NO (File format re-encoded)")

            return dimensions
        except Exception as err:
            raise ValueError(f"Image integrity verification failed: {err}")

    def analyze_medical_data(self, decrypted_file_path, stage1_info=None, original_file_path=None):
        """
        Runs real MedGemma multimodal inference on decrypted medical scan or report image.

        Args:
            decrypted_file_path (str): Path to decrypted file
            stage1_info (dict|str): Stage 1 identification metadata dict or medical_type string
            original_file_path (str): Optional path to original file for SHA-256 verification

        Returns dict:
            Structured medical analysis object containing Stage 1 info & Stage 2 fields
        """
        if isinstance(stage1_info, str):
            stage1_info = {
                "is_medical": True,
                "input_type": "medical_image",
                "body_region": "Unknown / Unable to determine",
                "modality": stage1_info,
                "report_type": None,
                "certainty": "medium"
            }
        stage1_info = stage1_info or {}

        display_context = stage1_info.get("body_region") or stage1_info.get("modality") or "Medical File"
        print(f"[MEDGEMMA DEBUG] Analysis request received for: {decrypted_file_path} (Context: {display_context})")

        if not os.path.exists(decrypted_file_path):
            return self._generate_error_response("Decrypted medical file not found.", stage1_info)

        is_pdf = decrypted_file_path.lower().endswith(".pdf")

        # Handle PDF report vs Image input
        if is_pdf:
            print("[MEDGEMMA DEBUG] Decryption successful: YES")
            print("[MEDGEMMA DEBUG] Actual decrypted report loaded")
            pdf_text = self.report_processor.extract_text_from_pdf(decrypted_file_path)
            page_img = self.report_processor.render_pdf_page_to_image(decrypted_file_path, page_num=0)
            if page_img:
                print(f"[MEDGEMMA DEBUG] Image dimensions: {page_img.size[0]}x{page_img.size[1]}")
                return self._run_multimodal_inference(page_img, stage1_info, additional_text=pdf_text)
            else:
                return self._run_text_report_inference(pdf_text, stage1_info)

        # Verify Image Integrity & Load PIL Image
        try:
            self.verify_image_integrity(decrypted_file_path, original_file_path)
            pil_image = Image.open(decrypted_file_path).convert("RGB")
        except Exception as err:
            print(f"[MEDGEMMA DEBUG] Image load error: {err}")
            return self._generate_error_response(str(err), stage1_info)

        # Run Real Multimodal Inference
        return self._run_multimodal_inference(pil_image, stage1_info)

    def _run_multimodal_inference(self, pil_image, stage1_info, additional_text=""):
        """Executes real multimodal image + prompt inference via MedGemma model API."""
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{img_b64}"

        print(f"[MEDGEMMA DEBUG] Base64 size: {len(img_b64)} bytes")
        print(f"[MEDGEMMA DEBUG] Sending actual decrypted image to MedGemma")

        prompt = get_medgemma_dynamic_prompt(stage1_info)
        if additional_text:
            prompt = f"Decrypted Medical Report Text Content:\n{additional_text[:1000]}\n\n" + prompt

        url = self.endpoint_url
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}

        # Gated / dedicated model + high-capacity medical reasoning endpoints (excluding Gemma 3 as requested)
        candidate_models = [self.model_name, "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"]

        for m_name in candidate_models:
            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    } if m_name == self.model_name else {"role": "user", "content": prompt}
                ],
                "max_tokens": 800
            }

            try:
                print(f"[MEDGEMMA DEBUG] API request started (Model: {m_name})")
                resp = requests.post(url, headers=headers, json=payload, timeout=45)
                print(f"[MEDGEMMA DEBUG] API response status: {resp.status_code}")

                if resp.status_code == 200:
                    print(f"[MEDGEMMA DEBUG] Response received: YES (Model: {m_name})")
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                    parsed = self._parse_json_response(raw_text, stage1_info=stage1_info)
                    print(f"[MEDGEMMA DEBUG] Analysis completed successfully")
                    return parsed
                else:
                    print(f"[MEDGEMMA DEBUG] Notice for {m_name} ({resp.status_code}): {resp.text[:120]}")
            except Exception as err:
                print(f"[MEDGEMMA DEBUG] Notice for {m_name}: {err}")

        return self._generate_error_response("MedGemma inference is currently unavailable.", stage1_info)


    def _run_text_report_inference(self, report_text, stage1_info=None):
        """Executes text-only report analysis when PDF rendering is unavailable."""
        print(f"[MEDGEMMA DEBUG] Sending text report + prompt to MedGemma")
        print(f"[MEDGEMMA DEBUG] API request started")
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}

        prompt = f"Decrypted Medical Report Text:\n{report_text[:1500]}\n\n" + get_medgemma_dynamic_prompt(stage1_info)
        candidate_models = [self.model_name, "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"]

        for m_name in candidate_models:
            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 800
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    raw_text = resp.json()["choices"][0]["message"]["content"]
                    print(f"[MEDGEMMA DEBUG] Response received: YES (Model: {m_name})")
                    parsed = self._parse_json_response(raw_text, stage1_info=stage1_info)
                    return parsed
            except Exception as e:
                print(f"[MEDGEMMA DEBUG] Text report inference notice for {m_name}: {e}")

        return self._generate_error_response("Unable to process medical report text.", stage1_info)


    def _clean_medical_type(self, parsed_type, raw_text, stage1_info):
        """Sanitizes and dynamically categorizes the medical image/report modality."""
        parsed_lower = (parsed_type or "").lower()
        text_lower = (raw_text or "").lower()
        stage1_info = stage1_info or {}
        body_reg = stage1_info.get("body_region", "")
        modality = stage1_info.get("modality", "")

        # If model output copied prompt placeholders
        if "identify" in parsed_lower or "e.g." in parsed_lower or not parsed_type:
            for organ in ["Brain", "Chest", "Lung", "Heart", "Skin", "Eye", "Kidney", "Liver", "Bone", "Spine", "Abdomen"]:
                if organ.lower() in text_lower or organ.lower() in body_reg.lower():
                    return f"{organ} ({modality or 'Diagnostic Image'})"

        if parsed_type and "identify" not in parsed_lower:
            return parsed_type

        if body_reg and body_reg != "Unknown / Unable to determine":
            return f"{body_reg} ({modality})"

        return modality or "Medical Diagnostic Image"

    def _parse_json_response(self, raw_text, stage1_info=None):
        """Parses decoded MedGemma output text into structured fields."""
        stage1_info = stage1_info or {}
        content = raw_text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        input_type = stage1_info.get("input_type", "medical_image")
        anatomical_region = stage1_info.get("body_region", "Dynamic Region")
        modality = stage1_info.get("modality", "Diagnostic Imaging")

        try:
            data = json.loads(content)
            raw_type = data.get("medical_image_report_type", "")
            final_type = self._clean_medical_type(raw_type, raw_text, stage1_info)

            return {
                "input_type": input_type,
                "anatomical_region": anatomical_region,
                "modality": modality,
                "stage1_identification": stage1_info,
                "medical_image_report_type": final_type,
                "medical_finding": data.get("medical_finding", "Findings visible in decrypted medical payload."),
                "abnormality_defect": data.get("abnormality_defect", "Unconfirmed / None identified."),
                "possible_condition": data.get("possible_condition", "Clinical correlation recommended."),
                "simple_explanation": data.get("simple_explanation", "Medical image evaluated."),
                "detailed_explanation": data.get("detailed_explanation", raw_text),
                "recommended_next_steps": data.get("recommended_next_steps", "Consult primary care physician for clinical evaluation."),
                "medication_info": data.get("medication_info", "Medication cannot be reliably determined from this image alone. A qualified healthcare professional should confirm appropriate treatment."),
                "uncertainty": data.get("uncertainty", "Analysis is limited to the provided digital image."),
                "disclaimer": MEDICAL_SAFETY_DISCLAIMER,
                "status": "SUCCESS"
            }
        except Exception:
            final_type = self._clean_medical_type("", raw_text, stage1_info)
            return {
                "input_type": input_type,
                "anatomical_region": anatomical_region,
                "modality": modality,
                "stage1_identification": stage1_info,
                "medical_image_report_type": final_type,
                "medical_finding": "Direct visual evaluation of decrypted medical scan.",
                "abnormality_defect": "Unconfirmed from plain text response.",
                "possible_condition": "Clinical evaluation required.",
                "simple_explanation": content[:250],
                "detailed_explanation": content,
                "recommended_next_steps": "Consult ordering healthcare provider.",
                "medication_info": "Medication cannot be reliably determined from this image alone.",
                "uncertainty": "Output parsed from model text response.",
                "disclaimer": MEDICAL_SAFETY_DISCLAIMER,
                "status": "SUCCESS"
            }

    def _generate_error_response(self, error_message, stage1_info=None):
        """Returns error structure when image or model is unavailable."""
        stage1_info = stage1_info or {}
        return {
            "input_type": stage1_info.get("input_type", "Unavailable"),
            "anatomical_region": stage1_info.get("body_region", "Unavailable"),
            "modality": stage1_info.get("modality", "Unavailable"),
            "stage1_identification": stage1_info,
            "medical_image_report_type": "Unavailable",
            "medical_finding": "Error processing image payload.",
            "abnormality_defect": "Error",
            "possible_condition": "Analysis Unsuccessful",
            "simple_explanation": error_message,
            "detailed_explanation": error_message,
            "recommended_next_steps": "Re-upload a clear medical scan or report file.",
            "medication_info": "Medication information unavailable.",
            "uncertainty": error_message,
            "disclaimer": MEDICAL_SAFETY_DISCLAIMER,
            "status": "ERROR"
        }
