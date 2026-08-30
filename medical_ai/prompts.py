"""
Medical AI Analysis Instructions & Dynamic Prompts (Stage 2 Integration)
========================================================================
Defines system instructions, safety disclaimers, and dynamic prompt generators
for MedGemma medical AI analysis.
"""

MEDICAL_SAFETY_DISCLAIMER = (
    "AI-generated medical analysis is for informational and research purposes only. "
    "It is not a confirmed diagnosis or a substitute for evaluation by a qualified healthcare professional. "
    "Do not start, stop, or change medication based solely on this AI output."
)

MEDGEMMA_SYSTEM_PROMPT = (
    "You are analyzing the actual medical image or medical report supplied in this request. "
    "Identify what is visibly present in the image based ONLY on visual evidence. "
    "Do not infer the body part from filename. Do not assume body part from previous analysis. "
    "Do not reuse previous results. Do not fabricate findings, abnormalities, diagnoses, or medications. "
    "First determine whether the input is: 1. Medical image, 2. Medical report/document, 3. Non-medical image. "
    "If it is a medical image, identify the anatomical region and imaging modality from visual evidence. "
    "Only report findings supported by the visible image or text in the supplied report. "
    "If there is insufficient evidence for a diagnosis, explicitly state 'No reliable diagnosis can be established from this image alone.' "
    "Medication recommendations must never be fabricated. State 'Medication cannot be determined from the image alone; consult a qualified clinician.'"
)


def get_medgemma_dynamic_prompt(stage1_info=None):
    """
    Generates a dynamic, context-aware instruction prompt for MedGemma analysis.
    Does NOT infer organ from filename or force previous classifications.
    """
    stage1_info = stage1_info or {}
    input_type = stage1_info.get("input_type", "medical_image")

    if input_type == "medical_report":
        context_guidance = (
            "Examine this medical report / document carefully. "
            "Identify document type (e.g. Laboratory Report, Clinical Document, Radiology Report). "
            "Extract and evaluate visible laboratory measurements, test values, clinical notes, or diagnostic impressions. "
            "Do not invent missing laboratory values or unconfirmed diagnoses."
        )
    else:
        context_guidance = (
            "Examine the supplied medical image carefully based ONLY on visual pixel evidence. "
            "Determine what anatomical region (e.g. Brain, Chest/Lung, Heart, Skin, Eye, Bone, Spine, Abdomen, Liver, Kidney, etc.) "
            "and imaging modality (e.g. X-Ray, CT, MRI, Ultrasound, Dermoscopy, Fundus Photograph, Histopathology) is visibly shown. "
            "Do not assume the image is an X-ray or eye scan unless actually visible in the image pixels. "
            "Describe only findings directly supported by visual evidence."
        )

    return f"""{context_guidance}

Provide your output in strict JSON format matching these exact 15 keys:
{{
  "input_type": "Medical Image / Medical Report / Non-Medical",
  "anatomical_region": "Identify anatomical region from visual evidence (e.g. Brain, Chest/Lung, Heart, Skin, Eye, Spine, Bone, Abdomen, Kidney, Liver, etc.)",
  "imaging_modality": "Identify imaging modality (e.g. X-Ray, CT Scan, MRI Scan, Ultrasound, Dermoscopy, Fundus Photograph, Medical Document)",
  "medical_image_report_type": "Full descriptor (e.g. Chest (X-Ray / Radiograph) or Skin (Dermoscopy) or Medical Laboratory Report)",
  "identification_confidence": "Not reliably quantifiable",
  "observed_findings": ["Visual observation 1", "Visual observation 2"],
  "abnormalities": ["Visible abnormality or state none/unconfirmed"],
  "possible_conditions": ["Possible condition 1"],
  "diagnosis": "Possible condition supported by visual evidence; state 'No reliable diagnosis can be established from this image alone' if insufficient",
  "medication_information": "Medication cannot be determined from the image alone; consult a qualified clinician (unless explicitly listed in a medical report)",
  "simple_explanation": "Plain language explanation for patients without technical jargon",
  "detailed_explanation": "Comprehensive clinical explanation supported by visual evidence",
  "recommended_next_steps": "Actionable recommended follow-up actions (e.g. physician consultation)",
  "uncertainty": "Limitations of the image or state if information is insufficient",
  "evidence": "Visual pixel evidence supporting identification"
}}
Return valid JSON only. Do not include markdown code blocks or surrounding text.
"""

