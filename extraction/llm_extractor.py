"""
DocuVault - LLM Extractor
Advanced document extraction using LLMs
"""
import json
import re
import time
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic
from loguru import logger
from pydantic import ValidationError

from core.config import Config
from extraction.schema import ExtractedPrescription


class LLMExtractor:
    """Extract structured data from OCR text using LLMs"""
    
    def __init__(self, provider: str = "openai"):
        """
        Initialize LLM extractor
        
        Args:
            provider: "openai" or "anthropic"
        """
        self.provider = provider
        
        if provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not configured")
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.model = Config.DEFAULT_LLM_MODEL
            
        elif provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            self.model = "claude-3-5-sonnet-20241022"
            
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"LLM Extractor initialized: {provider} / {self.model}")
    
    def build_extraction_prompt(self, ocr_text: str, document_type: Optional[str] = None) -> str:
        """
        Build extraction prompt with schema
        
        Args:
            ocr_text: OCR extracted text
            document_type: Optional document type hint
            
        Returns:
            Formatted prompt
        """
        schema_example = {
            "document_type": "prescription",
            "prescription_type": "printed",
            "prescription_number": "OPD6",
            "issue_date": "2023-08-30",
            "patient": {
                "name": "Patient Name",
                "age": "13 Y",
                "gender": "M",
                "address": "PUNE",
                "phone": "9423380390",
                "patient_id": "11"
            },
            "doctor": {
                "name": "Dr. Akshara",
                "title": "M.S.",
                "specialty": None,
                "license_number": "MMC 2018",
                "phone": "5465647658"
            },
            "hospital": {
                "name": "SMS hospital",
                "department": None,
                "address": "B/503, Business Center, MG Road, Pune - 411000",
                "phone": "5465647658"
            },
            "diagnosis": "MALARIA",
            "medications": [
                {
                    "name": "TAB. ABCIXIMAB",
                    "dosage": None,
                    "quantity": "8 Tab",
                    "frequency": "1 Morning",
                    "duration": "8 Days",
                    "instructions": None
                }
            ],
            "notes": "TAKE BED REST, DO NOT EAT OUTSIDE FOOD",
            "follow_up_date": "2023-09-04"
        }

        type_hint = f"\nDocument type hint: {document_type}" if document_type else ""

        prompt = f"""You are an expert medical document parser specialized in extracting structured data from medical prescriptions.

TASK: Extract ALL relevant information from the OCR text below into a structured JSON format.

RULES:
1. Return ONLY valid JSON - no markdown, no explanations, no preamble
2. Use null for missing/unknown fields
3. CRITICAL: Convert ALL dates to YYYY-MM-DD format (e.g., "30-Aug-2023" becomes "2023-08-30", "04-09-2023" becomes "2023-09-04")
4. Extract patient information: name, age, gender, address, phone, patient_id
5. Extract doctor information: name, title, specialty, license_number, phone
6. Extract hospital/clinic information: name, department, address, phone
7. Extract diagnosis if present
8. Extract ALL medications with: name, dosage, quantity, frequency, duration, instructions
9. CRITICAL: Extract "Advice" section into "notes" field - this includes any instructions like "TAKE BED REST", dietary advice, etc.
10. CRITICAL: Extract "Follow Up" date into "follow_up_date" field in YYYY-MM-DD format
11. Be thorough - capture ALL information present{type_hint}

EXPECTED JSON SCHEMA:
{json.dumps(schema_example, indent=2)}

IMPORTANT:
- "document_type" should be "prescription"
- Extract ALL patient, doctor, and hospital information visible
- ALL dates MUST be in YYYY-MM-DD format (convert from any format like DD-Mon-YYYY or DD-MM-YYYY)
- For medications array, include EVERY medication found with all details
- The "notes" field should contain ALL advice/instructions from the "Advice:" section
- The "follow_up_date" field should contain the follow-up date from "Follow Up:" section

OCR TEXT:
{ocr_text}

Return the extracted data as JSON:"""

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
    
    def call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document extraction system. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content or ""
    
    def call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=Config.LLM_MAX_TOKENS,
            temperature=Config.LLM_TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response.content[0].text
    
    def extract(
        self,
        ocr_text: str,
        document_type: Optional[str] = None
    ) -> Tuple[ExtractedPrescription, float]:
        """
        Extract structured data from OCR text

        Args:
            ocr_text: OCR extracted text
            document_type: Optional document type hint

        Returns:
            Tuple of (ExtractedPrescription, processing_time)
        """
        start_time = time.time()

        try:
            # Build prompt
            prompt = self.build_extraction_prompt(ocr_text, document_type)

            # Call LLM
            logger.info(f"Calling {self.provider} for extraction...")
            if self.provider == "openai":
                response_text = self.call_openai(prompt)
            else:
                response_text = self.call_anthropic(prompt)

            # Extract JSON
            raw_json = self.extract_json_from_response(response_text)

            # Validate with Pydantic
            document = ExtractedPrescription(**raw_json)

            processing_time = time.time() - start_time

            logger.success(
                f"Extraction completed: type={document.document_type}, "
                f"medications={len(document.medications)}, time={processing_time:.2f}s"
            )

            return document, processing_time

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            # Return minimal valid document
            processing_time = time.time() - start_time
            return ExtractedPrescription(document_type="unknown"), processing_time

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            processing_time = time.time() - start_time
            return ExtractedPrescription(document_type="unknown"), processing_time
    
    def extract_with_retry(
        self,
        ocr_text: str,
        document_type: Optional[str] = None,
        max_retries: int = 2
    ) -> Tuple[ExtractedPrescription, float]:
        """
        Extract with retry logic

        Args:
            ocr_text: OCR extracted text
            document_type: Optional document type hint
            max_retries: Maximum number of retries

        Returns:
            Tuple of (ExtractedPrescription, total_processing_time)
        """
        total_time = 0.0
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                document, processing_time = self.extract(ocr_text, document_type)
                total_time += processing_time

                # Check if extraction was successful
                if document.document_type != "unknown":
                    return document, total_time

                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")

        # All retries failed
        logger.error(f"All extraction attempts failed. Last error: {last_error}")
        return ExtractedPrescription(document_type="unknown"), total_time


# Global extractor instance
try:
    llm_extractor = LLMExtractor(provider="openai")
except Exception as e:
    logger.warning(f"Failed to initialize OpenAI extractor: {e}")
    try:
        llm_extractor = LLMExtractor(provider="anthropic")
    except Exception as e2:
        logger.error(f"Failed to initialize any LLM extractor: {e2}")
        llm_extractor = None
