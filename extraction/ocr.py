"""
DocuVault - OCR Engine
Robust OCR processing with PaddleOCR
"""
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
from loguru import logger

from core.config import Config
from extraction.pdf_converter import pdf_converter


class OCREngine:
    """OCR processing with PaddleOCR"""
    
    def __init__(self):
        """Initialize OCR engine"""
        logger.info("Initializing PaddleOCR engine...")
        self.engine = PaddleOCR(
            lang=Config.OCR_LANGUAGE,
            use_gpu=Config.OCR_USE_GPU,
            show_log=False,
            use_angle_cls=True,  # Enable text orientation detection
            det_db_thresh=0.3,   # Detection threshold
            det_db_box_thresh=0.5,  # Box threshold
            rec_batch_num=6      # Batch size for recognition
        )
        logger.success("OCR engine initialized")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}. Using original.")
            return cv2.imread(image_path)
    
    # Expected minimums for a typical prescription document
    EXPECTED_MIN_LINES = 8  # Prescription usually has at least 8 lines
    EXPECTED_MIN_CHARS = 150  # Prescription usually has at least 150 characters

    def calculate_confidence(self, results: list, extracted_text: str = "") -> float:
        """
        Calculate multi-factor confidence score from OCR results.

        Combines three factors:
        1. Base OCR confidence - average confidence of detected text lines
        2. Detection density - penalizes when too few lines are detected
        3. Text length - penalizes when extracted text is too short

        Args:
            results: PaddleOCR results
            extracted_text: The full extracted text (for length calculation)

        Returns:
            Combined confidence score (0-1)
        """
        if not results or not results[0]:
            return 0.0

        # Factor 1: Base OCR confidence (average of per-line confidences)
        confidences = []
        for page in results:
            for line in page:
                if len(line) >= 2 and len(line[1]) >= 2:
                    conf = line[1][1]
                    confidences.append(conf)

        if not confidences:
            return 0.0

        base_confidence = sum(confidences) / len(confidences)
        num_lines = len(confidences)
        text_length = len(extracted_text)

        # Factor 2: Detection density score
        # Penalize when fewer lines than expected are detected
        density_score = min(num_lines / self.EXPECTED_MIN_LINES, 1.0)

        # Factor 3: Text length score
        # Penalize when text is shorter than expected
        length_score = min(text_length / self.EXPECTED_MIN_CHARS, 1.0) if text_length > 0 else 0.0

        # Combine factors with weights:
        # - 50% base confidence (quality of what was detected)
        # - 25% density score (did we detect enough lines?)
        # - 25% length score (did we extract enough text?)
        combined_confidence = (
            base_confidence * 0.5 +
            density_score * 0.25 +
            length_score * 0.25
        )

        logger.debug(
            f"Confidence breakdown: base={base_confidence:.2%}, "
            f"density={density_score:.2%} ({num_lines} lines), "
            f"length={length_score:.2%} ({text_length} chars) → "
            f"combined={combined_confidence:.2%}"
        )

        return combined_confidence
    
    def extract_text(
        self, 
        image_path: str, 
        preprocess: bool = True
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text from image using OCR
        
        Args:
            image_path: Path to image file
            preprocess: Whether to preprocess image
            
        Returns:
            Tuple of (extracted_text, confidence_score, metadata)
        """
        start_time = time.time()
        
        try:
            # Validate file
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Handle PDF files
            if Path(image_path).suffix.lower() == '.pdf':
                return self._extract_from_pdf(image_path)
            
            # Preprocess if requested
            if preprocess:
                logger.info(f"Preprocessing image: {image_path}")
                processed_img = self.preprocess_image(image_path)
                # Save temporarily
                temp_path = str(Path(image_path).parent / f"_temp_{Path(image_path).name}")
                cv2.imwrite(temp_path, processed_img)
                ocr_input = temp_path
            else:
                ocr_input = image_path
            
            # Run OCR
            logger.info(f"Running OCR on: {Path(image_path).name}")
            results = self.engine.ocr(ocr_input, cls=True)
            
            # Extract text
            text_lines = []
            boxes = []
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        # line[0] = bounding box, line[1] = (text, confidence)
                        text = line[1][0]
                        text_lines.append(text)
                        boxes.append(line[0])
            
            extracted_text = "\n".join(text_lines)
            confidence = self.calculate_confidence(results, extracted_text)
            processing_time = time.time() - start_time
            
            # Metadata
            metadata = {
                "processing_time": processing_time,
                "confidence": confidence,
                "total_lines": len(text_lines),
                "preprocessed": preprocess,
                "boxes_count": len(boxes)
            }
            
            # Cleanup temp file
            if preprocess and Path(temp_path).exists():
                Path(temp_path).unlink()
            
            logger.success(
                f"OCR completed: {len(text_lines)} lines, "
                f"confidence: {confidence:.2%}, "
                f"time: {processing_time:.2f}s"
            )
            
            return extracted_text, confidence, metadata
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            processing_time = time.time() - start_time
            
            return "", 0.0, {
                "processing_time": processing_time,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _extract_from_pdf(self, pdf_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text from PDF by converting to images first
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, confidence_score, metadata)
        """
        start_time = time.time()
        
        if not pdf_converter:
            raise RuntimeError("PDF converter not available. Install PyMuPDF: pip install PyMuPDF")
        
        try:
            logger.info(f"Processing PDF: {Path(pdf_path).name}")
            
            # Convert PDF to images
            image_paths = pdf_converter.convert_to_images(pdf_path, dpi=300)
            
            if not image_paths:
                raise ValueError("PDF conversion produced no images")
            
            # Extract text from each page
            all_text = []
            all_confidences = []
            total_lines = 0
            
            for i, image_path in enumerate(image_paths, 1):
                logger.info(f"Processing page {i}/{len(image_paths)}")
                
                # Run OCR on page image
                results = self.engine.ocr(image_path, cls=True)
                
                # Extract text
                if results and results[0]:
                    page_lines = []
                    for line in results[0]:
                        if len(line) >= 2:
                            text = line[1][0]
                            page_lines.append(text)
                    
                    if page_lines:
                        all_text.append(f"\n--- Page {i} ---\n")
                        all_text.extend(page_lines)
                        total_lines += len(page_lines)

                    # Calculate page confidence (pass page text for length scoring)
                    page_text = "\n".join(page_lines)
                    page_confidence = self.calculate_confidence(results, page_text)
                    all_confidences.append(page_confidence)
            
            # Cleanup temporary images
            pdf_converter.cleanup_temp_images(image_paths)
            
            # Combine results
            extracted_text = "\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "confidence": avg_confidence,
                "total_lines": total_lines,
                "total_pages": len(image_paths),
                "preprocessed": False,
                "is_pdf": True
            }
            
            logger.success(
                f"PDF OCR completed: {len(image_paths)} pages, "
                f"{total_lines} lines, "
                f"confidence: {avg_confidence:.2%}, "
                f"time: {processing_time:.2f}s"
            )
            
            return extracted_text, avg_confidence, metadata
            
        except Exception as e:
            logger.error(f"PDF OCR extraction failed: {e}")
            processing_time = time.time() - start_time
            
            return "", 0.0, {
                "processing_time": processing_time,
                "confidence": 0.0,
                "error": str(e),
                "is_pdf": True
            }
    
    def extract_with_fallback(self, image_path: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text with fallback strategy
        
        First tries with preprocessing, then without if confidence is low
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (extracted_text, confidence_score, metadata)
        """
        # Try with preprocessing
        text, confidence, metadata = self.extract_text(image_path, preprocess=True)
        
        # If confidence is low, try without preprocessing
        if confidence < Config.OCR_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Low confidence ({confidence:.2%}), "
                f"retrying without preprocessing..."
            )
            text_alt, confidence_alt, metadata_alt = self.extract_text(
                image_path, 
                preprocess=False
            )
            
            # Use better result
            if confidence_alt > confidence:
                logger.info(f"Fallback improved confidence: {confidence_alt:.2%}")
                return text_alt, confidence_alt, metadata_alt
        
        return text, confidence, metadata


# Global OCR instance
ocr_engine = OCREngine()