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
    "You are MedGemma, Google's specialized healthcare AI model. "
    "Respond with accurate, evidence-based medical analysis of the actual input image or document."
)


def get_medgemma_dynamic_prompt(stage1_info=None):
    """
    Generates a dynamic, context-aware instruction prompt for MedGemma Stage 2 analysis.
    Does NOT hardcode a default organ or assume Chest X-Ray.
    """
    stage1_info = stage1_info or {}
    input_type = stage1_info.get("input_type", "medical_image")
    body_region = stage1_info.get("body_region", "Unknown / Unable to determine")
    modality = stage1_info.get("modality", "Medical Diagnostic Image")
    report_type = stage1_info.get("report_type")

    if input_type == "medical_report":
        context_guidance = (
            f"Examine this medical report document ({report_type or modality}) carefully. "
            "Extract and evaluate visible laboratory measurements, test values, clinical notes, or diagnostic impressions. "
            "Do not invent missing laboratory values, patient names, or unconfirmed diagnoses."
        )
    elif body_region and body_region != "Unknown / Unable to determine":
        context_guidance = (
            f"Examine the supplied medical image carefully. "
            f"The pre-analysis identification suggests an anatomical region of {body_region} ({modality}). "
            f"Perform a specialized visual evaluation of the actual image pixels corresponding to {body_region}. "
            "Describe only findings directly supported by the visual evidence in the image."
        )
    else:
        context_guidance = (
            "Examine the supplied medical image carefully. Determine what type of medical content is actually shown, "
            "identify the anatomical region and imaging modality if possible, and describe only findings supported by "
            "the supplied image. Do not assume a body region or modality that is not visible. State uncertainty explicitly if information is insufficient."
        )

    return f"""{context_guidance}

Do not assume that the image is a chest X-ray or brain scan unless actually present. Do not invent abnormalities, diagnoses, medications, symptoms, laboratory values, or anatomical structures that are not visible. If the image quality or information is insufficient, explicitly state that.

Provide your output in strict JSON format matching these exact 9 keys:
{{
  "medical_image_report_type": "Identify the exact medical image modality and anatomical region visible (e.g. {body_region if body_region != 'Unknown / Unable to determine' else 'Anatomical Region / Modality'})",
  "medical_finding": "Observed findings supported directly by the supplied image",
  "abnormality_defect": "Visible abnormality or defect, or state none / unconfirmed",
  "possible_condition": "Possible medical interpretation supported by the image; state uncertainty if insufficient",
  "simple_explanation": "Plain language explanation for patients without technical jargon",
  "detailed_explanation": "Comprehensive clinical explanation supported by visual evidence",
  "recommended_next_steps": "Actionable recommended follow-up actions (e.g. physician consultation)",
  "medication_info": "General treatment information only if supported; state unconfirmed if insufficient (do NOT invent prescriptions or dosages)",
  "uncertainty": "Limitations of the image or explicit state if information is insufficient"
}}
Return valid JSON only. Do not include markdown code blocks or surrounding text.
"""
