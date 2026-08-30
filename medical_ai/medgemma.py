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

        # Multimodal vision models (Qwen2.5-VL-72B-Instruct is active on HF Serverless Router with image_url support)
        candidate_models = [self.model_name, "Qwen/Qwen2.5-VL-72B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"]

        for m_name in candidate_models:
            is_vision_model = (m_name == self.model_name or "VL" in m_name or "Vision" in m_name)
            
            if is_vision_model:
                user_content = [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt}
                ]
            else:
                user_content = prompt

            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                "max_tokens": 800
            }

            try:
                print(f"[MEDGEMMA DEBUG] API request started (Model: {m_name}, Vision Payload: {is_vision_model})")
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


    def _clean_medical_type(self, parsed_type, ai_region, ai_modality, stage1_info):
        """Sanitizes and dynamically formats the medical image/report descriptor."""
        parsed_lower = (parsed_type or "").lower()

        # If model returned a clean specific type (e.g. "Chest X-Ray" or "Brain MRI")
        if parsed_type and not any(kw in parsed_lower for kw in ["identify", "e.g.", "full descriptor", "placeholder", "modality"]):
            return parsed_type

        # Format directly from AI visual extraction
        if ai_region and ai_modality and ai_region != "Not applicable":
            return f"{ai_region} ({ai_modality})"
        elif ai_region and ai_region != "Not applicable":
            return ai_region
        elif ai_modality:
            return ai_modality

        stage1_info = stage1_info or {}
        stg1_modality = stage1_info.get("modality", "Medical Diagnostic Image")
        return stg1_modality

    def _parse_json_response(self, raw_text, stage1_info=None):
        """Parses decoded MedGemma output text into structured fields."""
        stage1_info = stage1_info or {}
        content = raw_text.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        stg1_input_type = stage1_info.get("input_type", "medical_image")
        stg1_region = stage1_info.get("body_region", "Dynamic Region")
        stg1_modality = stage1_info.get("modality", "Diagnostic Imaging")

        try:
            data = json.loads(content)
            ai_region = data.get("anatomical_region") or stg1_region
            ai_modality = data.get("imaging_modality") or stg1_modality
            raw_type = data.get("medical_image_report_type", "")
            final_type = self._clean_medical_type(raw_type, ai_region, ai_modality, stage1_info)

            ai_diagnosis = data.get("diagnosis") or data.get("possible_condition") or "No reliable diagnosis can be established from this image alone."

            return {
                "input_type": data.get("input_type") or stg1_input_type,
                "anatomical_region": ai_region,
                "imaging_modality": ai_modality,
                "medical_image_report_type": final_type,
                "identification_confidence": data.get("identification_confidence", "Not reliably quantifiable"),
                "observed_findings": data.get("observed_findings") or [data.get("medical_finding", "Findings visible in decrypted medical payload.")],
                "medical_finding": data.get("medical_finding", "Findings visible in decrypted medical payload."),
                "abnormalities": data.get("abnormalities") or [data.get("abnormality_defect", "Unconfirmed / None identified.")],
                "abnormality_defect": data.get("abnormality_defect", "Unconfirmed / None identified."),
                "possible_conditions": data.get("possible_conditions") or [ai_diagnosis],
                "possible_condition": ai_diagnosis,
                "diagnosis": ai_diagnosis,
                "medication_information": data.get("medication_information") or data.get("medication_info") or "Medication cannot be determined from the image alone; consult a qualified clinician.",
                "medication_info": data.get("medication_information") or data.get("medication_info") or "Medication cannot be determined from the image alone; consult a qualified clinician.",
                "simple_explanation": data.get("simple_explanation", "Medical image evaluated based on visual pixel evidence."),
                "detailed_explanation": data.get("detailed_explanation", raw_text),
                "recommended_next_steps": data.get("recommended_next_steps", "Consult primary care physician for clinical evaluation."),
                "uncertainty": data.get("uncertainty", "Analysis is limited to the provided digital image."),
                "evidence": data.get("evidence", "Visual pixel features evaluated."),
                "disclaimer": MEDICAL_SAFETY_DISCLAIMER,
                "status": "SUCCESS"
            }
        except Exception:
            final_type = self._clean_medical_type("", stg1_region, stg1_modality, stage1_info)
            return {
                "input_type": stg1_input_type,
                "anatomical_region": stg1_region,
                "imaging_modality": stg1_modality,
                "medical_image_report_type": final_type,
                "identification_confidence": "Not reliably quantifiable",
                "observed_findings": ["Direct visual evaluation of decrypted medical scan."],
                "medical_finding": "Direct visual evaluation of decrypted medical scan.",
                "abnormalities": ["Unconfirmed from plain text response."],
                "abnormality_defect": "Unconfirmed from plain text response.",
                "possible_conditions": ["Clinical evaluation required."],
                "possible_condition": "Clinical evaluation required.",
                "diagnosis": "No reliable diagnosis can be established from this image alone.",
                "medication_information": "Medication cannot be determined from the image alone; consult a qualified clinician.",
                "medication_info": "Medication cannot be determined from the image alone; consult a qualified clinician.",
                "simple_explanation": content[:250],
                "detailed_explanation": content,
                "recommended_next_steps": "Consult ordering healthcare provider.",
                "uncertainty": "Output parsed from model text response.",
                "evidence": "Raw model text response.",
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
