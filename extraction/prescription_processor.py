"""
DocuVault - Medical Prescription Processor
Intelligent processing pipeline that combines OCR with LLM vision capabilities
for handling handwritten prescriptions and signature detection
"""
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from loguru import logger

from core.config import Config
from extraction.ocr import ocr_engine
from extraction.llm_extractor import llm_extractor
from extraction.schema import ExtractedPrescription, HandwritingAnalysis, SignatureInfo
from extraction.vision_extractor import vision_extractor


# Confidence threshold below which we trigger LLM vision fallback
HANDWRITING_CONFIDENCE_THRESHOLD = 0.6  # Handwritten text typically has lower OCR confidence
VISION_ALWAYS_FOR_SIGNATURES = True  # Always use vision for signature detection


class PrescriptionProcessor:
    """
    Intelligent prescription processor that:
    1. First attempts OCR extraction
    2. If OCR confidence is low (handwritten text), triggers LLM vision
    3. Always uses LLM vision for signature detection
    4. Combines results for best accuracy
    """

    def __init__(self):
        """Initialize prescription processor"""
        self.ocr = ocr_engine
        self.vision = vision_extractor
        self.llm_text = llm_extractor  # For text-to-JSON structuring (no vision)

        if not self.vision:
            logger.warning(
                "Vision extractor not available. "
                "Prescription processing will be limited to OCR + LLM text."
            )

        if not self.llm_text:
            logger.warning(
                "LLM text extractor not available. "
                "High-confidence OCR will fall back to regex parsing."
            )

    def classify_prescription_type(
        self,
        image_path: Union[str, Path],
        ocr_confidence: float
    ) -> str:
        """
        Classify prescription type based on source folder or OCR confidence

        Args:
            image_path: Path to the image
            ocr_confidence: OCR confidence score

        Returns:
            Prescription type: 'handwritten', 'printed', 'mixed', or 'digital'
        """
        path = Path(image_path)
        parent_folder = path.parent.name.lower()

        # Check folder name for hints
        if "handwrit" in parent_folder:
            return "handwritten"
        elif "print" in parent_folder:
            return "printed"
        elif "mixed" in parent_folder:
            return "mixed"
        elif "screen" in parent_folder or "digital" in parent_folder:
            return "digital"

        # Infer from OCR confidence
        if ocr_confidence < 0.5:
            return "handwritten"
        elif ocr_confidence < 0.7:
            return "mixed"
        else:
            return "printed"

    def needs_vision_enhancement(
        self,
        ocr_confidence: float,
        prescription_type: str
    ) -> bool:
        """
        Determine if LLM vision should be used to enhance extraction

        Args:
            ocr_confidence: OCR confidence score
            prescription_type: Type of prescription

        Returns:
            True if vision enhancement is recommended
        """
        # Always enhance handwritten prescriptions
        if prescription_type == "handwritten":
            return True

        # Enhance mixed prescriptions
        if prescription_type == "mixed":
            return True

        # Enhance if OCR confidence is below threshold
        if ocr_confidence < HANDWRITING_CONFIDENCE_THRESHOLD:
            return True

        # Always use vision for signature detection (handled separately)
        return False

    def process(
        self,
        image_path: Union[str, Path],
        force_vision: bool = False
    ) -> Tuple[ExtractedPrescription, Dict[str, Any]]:
        """
        Process a prescription image with intelligent OCR/Vision fallback

        Args:
            image_path: Path to the prescription image
            force_vision: Force using LLM vision regardless of OCR confidence

        Returns:
            Tuple of (ExtractedPrescription, processing_metadata)
        """
        start_time = time.time()
        image_path = Path(image_path)

        logger.info(f"Processing prescription: {image_path.name}")

        metadata = {
            "file_name": image_path.name,
            "file_path": str(image_path),
            "ocr_used": True,
            "vision_used": False,
            "extraction_method": "ocr_only",
            "processing_stages": []
        }

        # Stage 1: OCR Extraction
        logger.info("Stage 1: Running OCR extraction...")
        ocr_start = time.time()

        # Added logging to capture raw OCR text and metadata for debugging purposes
        ocr_text, ocr_confidence, ocr_metadata = self.ocr.extract_with_fallback(str(image_path))
        ocr_time = time.time() - ocr_start

        # Log raw OCR text and metadata for debugging
        logger.debug(f"Raw OCR Text: {ocr_text}")
        logger.debug(f"OCR Metadata: {ocr_metadata}")

        metadata["processing_stages"].append({
            "stage": "ocr",
            "time": ocr_time,
            "confidence": ocr_confidence,
            "text_length": len(ocr_text)
        })

        logger.info(f"OCR completed: confidence={ocr_confidence:.2%}, lines={ocr_metadata.get('total_lines', 0)}")

        # Classify prescription type
        prescription_type = self.classify_prescription_type(image_path, ocr_confidence)
        logger.info(f"Prescription type detected: {prescription_type}")

        # Stage 2: Determine if vision enhancement is needed
        use_vision = force_vision or self.needs_vision_enhancement(ocr_confidence, prescription_type)

        if use_vision and self.vision:
            logger.info("Stage 2: OCR confidence low or handwritten content detected - using LLM Vision...")
            vision_start = time.time()

            # Use vision extractor with OCR text as supplementary context
            prescription, vision_time = self.vision.extract_from_image(
                image_path,
                ocr_text=ocr_text,
                ocr_confidence=ocr_confidence
            )

            metadata["vision_used"] = True
            metadata["extraction_method"] = "ocr_plus_llm_vision"
            metadata["processing_stages"].append({
                "stage": "vision",
                "time": vision_time,
                "reason": f"prescription_type={prescription_type}, ocr_confidence={ocr_confidence:.2%}"
            })

            # Update prescription metadata
            prescription.prescription_type = prescription_type
            prescription.extraction_method = "ocr_plus_llm"
            prescription.ocr_confidence = ocr_confidence
            prescription.llm_enhanced = True

        else:
            # OCR confidence is high - use LLM text extractor (text-to-JSON, no vision)
            # This is cheaper than vision but more accurate than regex
            if self.llm_text:
                logger.info("Stage 2: OCR confidence sufficient - using LLM text structuring (no vision)...")
                llm_start = time.time()

                # Use LLM to structure OCR text into JSON
                prescription, llm_time = self.llm_text.extract(ocr_text, document_type="prescription")

                metadata["extraction_method"] = "ocr_plus_llm_text"
                metadata["vision_used"] = False
                metadata["llm_text_used"] = True
                metadata["processing_stages"].append({
                    "stage": "llm_text_structuring",
                    "time": llm_time,
                    "reason": f"OCR confidence sufficient ({ocr_confidence:.2%}), using LLM for JSON structuring"
                })

                # Update prescription metadata
                prescription.prescription_type = prescription_type
                prescription.extraction_method = "ocr_plus_llm_text"
                prescription.ocr_confidence = ocr_confidence
                prescription.llm_enhanced = True  # LLM was used (text mode, not vision)

            else:
                # Fallback to regex parsing if LLM text extractor not available
                logger.warning("Stage 2: LLM text extractor not available - falling back to regex parsing...")
                from extraction.ocr_parser import ocr_parser
                prescription = ocr_parser.parse(ocr_text, ocr_confidence)
                prescription.prescription_type = prescription_type

                metadata["extraction_method"] = "ocr_only_regex"
                metadata["vision_used"] = False
                metadata["processing_stages"].append({
                    "stage": "ocr_parsing_regex",
                    "time": 0.0,
                    "reason": f"LLM not available, using regex fallback"
                })

        # Assign OCR text to the prescription object
        prescription.ocr_text = ocr_text

        # Stage 3: Signature detection (always use vision if available)
        if VISION_ALWAYS_FOR_SIGNATURES and self.vision and not prescription.doctor_signature:
            logger.info("Stage 3: Running dedicated signature detection...")
            sig_start = time.time()
            signature_info, sig_time = self.vision.analyze_signature_only(image_path)

            prescription.doctor_signature = signature_info

            metadata["processing_stages"].append({
                "stage": "signature_detection",
                "time": sig_time,
                "signature_found": signature_info.is_present
            })

        # Finalize metadata
        total_time = time.time() - start_time
        metadata["total_processing_time"] = total_time
        metadata["final_confidence"] = prescription.confidence_score or ocr_confidence

        logger.success(
            f"Prescription processing complete: "
            f"method={metadata['extraction_method']}, "
            f"medications={len(prescription.medications)}, "
            f"signature={'detected' if prescription.doctor_signature and prescription.doctor_signature.is_present else 'not detected'}, "
            f"time={total_time:.2f}s"
        )

        return prescription, metadata

    def process_batch(
        self,
        image_paths: list,
        force_vision: bool = False
    ) -> list:
        """
        Process multiple prescription images

        Args:
            image_paths: List of image paths
            force_vision: Force vision for all

        Returns:
            List of (prescription, metadata) tuples
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths, 1):
            logger.info(f"Processing {i}/{total}: {Path(path).name}")

            try:
                prescription, metadata = self.process(path, force_vision)
                results.append((prescription, metadata))
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append((
                    ExtractedPrescription(document_type="prescription"),
                    {"error": str(e), "file_path": str(path)}
                ))

        return results


# Global processor instance
prescription_processor = PrescriptionProcessor()


def process_prescription(
    image_path: Union[str, Path],
    force_vision: bool = False
) -> Tuple[ExtractedPrescription, Dict[str, Any]]:
    """
    Convenience function to process a single prescription

    Args:
        image_path: Path to prescription image
        force_vision: Force LLM vision usage

    Returns:
        Tuple of (ExtractedPrescription, metadata)
    """
    return prescription_processor.process(image_path, force_vision)
