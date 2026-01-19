"""
DocuVault - Document Processor
Orchestrates OCR and LLM extraction pipeline
"""
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from core.config import Config
from extraction.ocr import ocr_engine
from extraction.llm_extractor import llm_extractor
from storage.database import db_manager, Document


class DocumentProcessor:
    """Process documents through OCR and LLM extraction pipeline"""
    
    def __init__(self):
        """Initialize document processor"""
        self.ocr = ocr_engine
        self.llm = llm_extractor
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to file
            
        Returns:
            True if valid, False otherwise
        """
        path = Path(file_path)
        
        # Check existence
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            logger.error(
                f"File too large: {file_size_mb:.2f}MB "
                f"(max: {Config.MAX_FILE_SIZE_MB}MB)"
            )
            return False
        
        # Check format
        file_ext = path.suffix.lower().lstrip('.')
        if file_ext not in Config.SUPPORTED_FORMATS:
            logger.error(
                f"Unsupported format: {file_ext}. "
                f"Supported: {Config.SUPPORTED_FORMATS}"
            )
            return False
        
        return True
    
    def save_file(self, source_path: str, original_filename: str) -> tuple[str, str]:
        """
        Save uploaded file to storage
        
        Args:
            source_path: Source file path
            original_filename: Original filename
            
        Returns:
            Tuple of (saved_path, unique_filename)
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(original_filename).suffix
        unique_filename = f"{timestamp}_{original_filename}"
        
        # Save to uploads directory
        dest_path = Config.UPLOADS_DIR / unique_filename
        shutil.copy2(source_path, dest_path)
        
        logger.info(f"File saved: {unique_filename}")
        
        return str(dest_path), unique_filename
    
    def process_document(
        self, 
        file_path: str, 
        original_filename: str,
        document_type_hint: Optional[str] = None
    ) -> Optional[Document]:
        """
        Process a single document through full pipeline
        
        Args:
            file_path: Path to document file
            original_filename: Original filename
            document_type_hint: Optional document type hint
            
        Returns:
            Document database record or None if failed
        """
        logger.info(f"Processing document: {original_filename}")
        
        # Validate file
        if not self.validate_file(file_path):
            return None
        
        # Create database record
        session = db_manager.get_session()
        
        try:
            # Save file
            saved_path, unique_filename = self.save_file(file_path, original_filename)
            
            # Create DB entry
            doc = Document(
                filename=unique_filename,
                original_filename=original_filename,
                file_path=saved_path,
                file_size=Path(saved_path).stat().st_size,
                file_type=Path(saved_path).suffix.lower().lstrip('.'),
                status="processing"
            )
            
            session.add(doc)
            session.commit()
            
            # Ensure OCR extraction works and add JSON data to details
            # Step 1: OCR Extraction
            logger.info(f"Step 1/2: OCR extraction for doc_id={doc.id}")
            ocr_text, ocr_confidence, ocr_metadata = self.ocr.extract_with_fallback(saved_path)

            if not ocr_text:
                logger.error(f"OCR failed for doc_id={doc.id}")
                doc.status = "failed"
                session.commit()
                return doc

            # Update OCR results
            doc.ocr_text = ocr_text
            doc.ocr_confidence = ocr_confidence
            doc.metadata = ocr_metadata
            doc.status = "completed"
            session.commit()

            # Add JSON data to details
            if doc.ocr_text:
                with st.expander("📄 View Details", expanded=False):
                    st.json(doc.ocr_text)

            return doc
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            if 'doc' in locals():
                doc.status = "failed"
                session.commit()
            return None
            
        finally:
            session.close()
    
    def process_batch(
        self, 
        file_paths: list[tuple[str, str]],
        document_type_hint: Optional[str] = None
    ) -> list[Document]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of (file_path, original_filename) tuples
            document_type_hint: Optional document type hint
            
        Returns:
            List of processed Document records
        """
        logger.info(f"Processing batch of {len(file_paths)} documents")
        
        results = []
        for i, (file_path, original_filename) in enumerate(file_paths, 1):
            logger.info(f"Processing {i}/{len(file_paths)}: {original_filename}")
            
            doc = self.process_document(file_path, original_filename, document_type_hint)
            if doc:
                results.append(doc)
        
        logger.success(f"Batch processing completed: {len(results)}/{len(file_paths)} successful")
        
        return results
    
    def reprocess_document(self, doc_id: int) -> Optional[Document]:
        """
        Reprocess an existing document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Updated Document record or None
        """
        session = db_manager.get_session()
        
        try:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            
            if not doc:
                logger.error(f"Document not found: doc_id={doc_id}")
                return None
            
            logger.info(f"Reprocessing document: doc_id={doc_id}")
            
            # Clear previous results
            doc.status = "processing"
            doc.ocr_text = None
            doc.extracted_data = None
            session.commit()
            
            # Reprocess
            return self.process_document(
                doc.file_path, 
                doc.original_filename
            )
            
        finally:
            session.close()


# Global processor instance
document_processor = DocumentProcessor()
