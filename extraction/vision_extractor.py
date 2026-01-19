"""
DocuVault - Vision-based LLM Extractor
Extracts data from prescription images using GPT-4o vision capabilities
Handles handwritten text and signature detection that OCR cannot process
"""
import json
import re
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
from openai import OpenAI
from loguru import logger
from pydantic import ValidationError

from core.config import Config
from extraction.schema import ExtractedPrescription, SignatureInfo, HandwritingAnalysis


class VisionExtractor:
    """Extract structured data from prescription images using LLM vision"""

    def __init__(self):
        """Initialize Vision extractor with OpenAI GPT-4o"""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured - required for vision extraction")

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = "gpt-4o"  # Vision-capable model

        logger.info(f"Vision Extractor initialized with {self.model}")

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode image to base64 for API transmission

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_media_type(self, image_path: Union[str, Path]) -> str:
        """Get the media type based on file extension"""
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/jpeg")

    def build_prescription_prompt(self, ocr_text: Optional[str] = None, ocr_confidence: Optional[float] = None) -> str:
        """
        Build the extraction prompt for medical prescriptions

        Args:
            ocr_text: Optional OCR-extracted text to supplement vision analysis
            ocr_confidence: Confidence score from OCR (if available)

        Returns:
            Formatted prompt string
        """
        ocr_section = ""
        if ocr_text:
            confidence_note = f" (OCR confidence: {ocr_confidence:.1%})" if ocr_confidence else ""
            ocr_section = f"""

OCR-EXTRACTED TEXT{confidence_note}:
The following text was extracted by OCR. Use this as a reference, but rely on your vision
for handwritten text, signatures, and anything the OCR might have missed or misread:

{ocr_text}

---
"""

        prompt = f"""You are an expert medical document analyzer specializing in reading medical prescriptions,
including HANDWRITTEN text and SIGNATURES that traditional OCR systems cannot process.

TASK: Analyze this medical prescription image and extract ALL information into structured JSON.

CRITICAL CAPABILITIES YOU MUST USE:
1. **SIGNATURE DETECTION**: Identify and describe any signatures present. Note:
   - Location on the document (e.g., "bottom right", "after medications list")
   - Whether it's legible or just a scribble
   - If you can read a name from the signature
   - Confidence in your signature detection (0.0 to 1.0)

2. **HANDWRITTEN TEXT READING**: Carefully read ALL handwritten portions:
   - Medication names written by hand
   - Dosage instructions
   - Patient names
   - Doctor notes
   - Any handwritten additions to printed forms

3. **MIXED CONTENT HANDLING**: Distinguish between:
   - Printed/typed text (higher OCR reliability)
   - Handwritten text (requires your vision capabilities)
   - Unclear or ambiguous text (note these in unclear_text field)
{ocr_section}
EXPECTED JSON SCHEMA:
{{
    "document_type": "prescription",
    "prescription_type": "handwritten" | "printed" | "mixed" | "digital",
    "prescription_number": "string or null",
    "barcode": "string or null",
    "issue_date": "YYYY-MM-DD or original format",

    "patient": {{
        "name": "string or null",
        "age": "string or null",
        "gender": "string or null",
        "address": "string or null",
        "phone": "string or null",
        "patient_id": "string or null"
    }},

    "doctor": {{
        "name": "string or null",
        "title": "string or null (e.g., ThS.BS, Dr.)",
        "specialty": "string or null",
        "license_number": "string or null",
        "phone": "string or null"
    }},

    "hospital": {{
        "name": "string or null",
        "department": "string or null",
        "address": "string or null",
        "phone": "string or null",
        "pharmacy_counter": "string or null"
    }},

    "diagnosis": "string or null",

    "medications": [
        {{
            "name": "medication name",
            "dosage": "e.g., 400mg",
            "quantity": "e.g., 30 tablets",
            "frequency": "e.g., twice daily after meals",
            "duration": "e.g., 7 days",
            "instructions": "special instructions",
            "is_handwritten": true/false
        }}
    ],

    "doctor_signature": {{
        "is_present": true/false,
        "signer_name": "name if legible",
        "signer_title": "title if visible",
        "location": "where on document",
        "is_legible": true/false,
        "confidence": 0.0-1.0
    }},

    "handwriting_analysis": {{
        "has_handwritten_content": true/false,
        "handwritten_sections": ["list of sections with handwriting"],
        "ocr_confidence": 0.0-1.0,
        "llm_interpretation": "your interpretation of difficult-to-read text",
        "unclear_text": ["list of text you couldn't clearly read"]
    }},

    "notes": "any additional notes",
    "total_items": number,
    "confidence_score": 0.0-1.0
}}

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown code blocks, no explanations
2. Use null for missing/unreadable fields
3. For Vietnamese prescriptions: preserve Vietnamese text exactly as written
4. ALWAYS analyze signatures even if they're just scribbles
5. Note which medications are handwritten vs printed
6. Be thorough with handwriting - this is your main advantage over OCR

Analyze the prescription image now:"""

        return prompt

    def extract_json_from_response(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response

        Args:
            text: LLM response text

        Returns:
            Parsed JSON dict
        """
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise ValueError("No valid JSON found in LLM response")

    def extract_from_image(
        self,
        image_path: Union[str, Path],
        ocr_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None
    ) -> Tuple[ExtractedPrescription, float]:
        """
        Extract prescription data directly from image using vision

        Args:
            image_path: Path to the prescription image
            ocr_text: Optional OCR text to supplement analysis
            ocr_confidence: OCR confidence score

        Returns:
            Tuple of (ExtractedPrescription, processing_time)
        """
        start_time = time.time()

        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            media_type = self.get_image_media_type(image_path)

            # Build prompt
            prompt = self.build_prescription_prompt(ocr_text, ocr_confidence)

            logger.info(f"Calling GPT-4o Vision for prescription extraction...")

            # Call GPT-4o with vision
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical prescription analyzer with advanced capabilities for reading handwritten text and detecting signatures. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": "high"  # High detail for better text reading
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1  # Low temperature for consistent extraction
            )

            response_text = response.choices[0].message.content or ""

            # Extract JSON
            raw_json = self.extract_json_from_response(response_text)

            # Add extraction method metadata
            raw_json["extraction_method"] = "llm_vision"
            raw_json["llm_enhanced"] = True
            if ocr_confidence:
                raw_json["ocr_confidence"] = ocr_confidence

            # Validate with Pydantic
            prescription = ExtractedPrescription(**raw_json)

            processing_time = time.time() - start_time

            logger.success(
                f"Vision extraction completed: "
                f"medications={len(prescription.medications)}, "
                f"signature_detected={prescription.doctor_signature.is_present if prescription.doctor_signature else False}, "
                f"time={processing_time:.2f}s"
            )

            return prescription, processing_time

        except ValidationError as e:
            logger.error(f"Validation error in vision extraction: {e}")
            processing_time = time.time() - start_time
            return ExtractedPrescription(
                document_type="prescription",
                extraction_method="llm_vision",
                llm_enhanced=True
            ), processing_time

        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            processing_time = time.time() - start_time
            return ExtractedPrescription(
                document_type="prescription",
                extraction_method="llm_vision",
                llm_enhanced=True
            ), processing_time

    def analyze_signature_only(
        self,
        image_path: Union[str, Path]
    ) -> Tuple[SignatureInfo, float]:
        """
        Analyze image specifically for signature detection

        Args:
            image_path: Path to the document image

        Returns:
            Tuple of (SignatureInfo, processing_time)
        """
        start_time = time.time()

        try:
            base64_image = self.encode_image(image_path)
            media_type = self.get_image_media_type(image_path)

            prompt = """Analyze this document image and focus ONLY on signature detection.

Return JSON with signature information:
{
    "is_present": true/false,
    "signer_name": "name if you can read it from the signature or nearby text",
    "signer_title": "title if visible (e.g., 'Doctor', 'BÁC SĨ ĐIỀU TRỊ')",
    "location": "where on the document (e.g., 'bottom right', 'bottom center')",
    "is_legible": true/false,
    "confidence": 0.0-1.0
}

Look for:
- Handwritten signatures (cursive writing that looks like a name)
- Signature lines or boxes
- Text labels like "Signature", "Ký tên", "BÁC SĨ KHÁM BỆNH"
- Any name printed near or under the signature

Return ONLY JSON, no explanations."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )

            response_text = response.choices[0].message.content or ""
            raw_json = self.extract_json_from_response(response_text)

            signature_info = SignatureInfo(**raw_json)
            processing_time = time.time() - start_time

            logger.info(f"Signature analysis complete: present={signature_info.is_present}, time={processing_time:.2f}s")

            return signature_info, processing_time

        except Exception as e:
            logger.error(f"Signature analysis failed: {e}")
            processing_time = time.time() - start_time
            return SignatureInfo(is_present=False), processing_time

    def read_handwritten_text(
        self,
        image_path: Union[str, Path],
        region_hint: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Read handwritten text from an image

        Args:
            image_path: Path to the image
            region_hint: Optional hint about where handwriting is located

        Returns:
            Tuple of (extracted_text, processing_time)
        """
        start_time = time.time()

        try:
            base64_image = self.encode_image(image_path)
            media_type = self.get_image_media_type(image_path)

            region_instruction = f"\nFocus especially on: {region_hint}" if region_hint else ""

            prompt = f"""Read ALL handwritten text in this image.{region_instruction}

Instructions:
1. Identify all handwritten portions (not printed/typed text)
2. Transcribe the handwritten text as accurately as possible
3. If text is unclear, provide your best interpretation with [unclear] marker
4. Preserve the original language (likely Vietnamese)

Return your transcription as plain text, preserving line breaks where appropriate.
Mark unclear portions with [unclear: your best guess]."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            extracted_text = response.choices[0].message.content or ""
            processing_time = time.time() - start_time

            logger.info(f"Handwritten text extraction complete: {len(extracted_text)} chars, time={processing_time:.2f}s")

            return extracted_text, processing_time

        except Exception as e:
            logger.error(f"Handwritten text extraction failed: {e}")
            processing_time = time.time() - start_time
            return "", processing_time


# Global instance
try:
    vision_extractor = VisionExtractor()
except Exception as e:
    logger.warning(f"Failed to initialize Vision Extractor: {e}")
    vision_extractor = None
