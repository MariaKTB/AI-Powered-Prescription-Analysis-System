"""
DocuVault - PDF Converter
Convert PDF pages to images for OCR processing
"""
import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF support will be limited.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available. Falling back to PyMuPDF.")

from core.config import Config


class PDFConverter:
    """Convert PDF documents to images for OCR processing"""
    
    def __init__(self):
        """Initialize PDF converter"""
        if not PYMUPDF_AVAILABLE and not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "No PDF library available. Install PyMuPDF (pip install PyMuPDF) "
                "or pdf2image (pip install pdf2image)"
            )
        
        self.method = "pymupdf" if PYMUPDF_AVAILABLE else "pdf2image"
        logger.info(f"PDF Converter initialized using: {self.method}")
    
    def is_pdf(self, file_path: str) -> bool:
        """
        Check if file is a PDF
        
        Args:
            file_path: Path to file
            
        Returns:
            True if PDF, False otherwise
        """
        return Path(file_path).suffix.lower() == '.pdf'
    
    def convert_pdf_to_images_pymupdf(
        self, 
        pdf_path: str,
        output_dir: Optional[str] = None,
        dpi: int = 300
    ) -> List[str]:
        """
        Convert PDF to images using PyMuPDF (faster, better quality)
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for images
            dpi: Resolution for conversion
            
        Returns:
            List of image file paths
        """
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF not available")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Config.CACHE_DIR / "pdf_pages"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open PDF
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Convert each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to pixmap
            zoom = dpi / 72  # 72 is the default DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            base_name = Path(pdf_path).stem
            image_path = output_dir / f"{base_name}_page_{page_num + 1}.png"
            pix.save(str(image_path))
            
            image_paths.append(str(image_path))
            logger.debug(f"Converted page {page_num + 1}/{len(doc)}: {image_path.name}")
        
        doc.close()
        
        logger.success(f"Converted PDF to {len(image_paths)} images using PyMuPDF")
        return image_paths
    
    def convert_pdf_to_images_pdf2image(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        dpi: int = 300
    ) -> List[str]:
        """
        Convert PDF to images using pdf2image (requires poppler)
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for images
            dpi: Resolution for conversion
            
        Returns:
            List of image file paths
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image not available")
        
        # Setup output directory
        if output_dir is None:
            output_dir = Config.CACHE_DIR / "pdf_pages"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=dpi)
        image_paths = []
        
        base_name = Path(pdf_path).stem
        
        for i, image in enumerate(images):
            image_path = output_dir / f"{base_name}_page_{i + 1}.png"
            image.save(str(image_path), 'PNG')
            image_paths.append(str(image_path))
            logger.debug(f"Converted page {i + 1}/{len(images)}: {image_path.name}")
        
        logger.success(f"Converted PDF to {len(image_paths)} images using pdf2image")
        return image_paths
    
    def convert_to_images(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        dpi: int = 300
    ) -> List[str]:
        """
        Convert PDF to images using available method
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Output directory for images
            dpi: Resolution for conversion
            
        Returns:
            List of image file paths
        """
        if not self.is_pdf(pdf_path):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Converting PDF to images: {Path(pdf_path).name}")
        
        try:
            if self.method == "pymupdf":
                return self.convert_pdf_to_images_pymupdf(pdf_path, output_dir, dpi)
            else:
                return self.convert_pdf_to_images_pdf2image(pdf_path, output_dir, dpi)
        
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def cleanup_temp_images(self, image_paths: List[str]):
        """
        Clean up temporary image files
        
        Args:
            image_paths: List of image paths to delete
        """
        for image_path in image_paths:
            try:
                Path(image_path).unlink()
                logger.debug(f"Cleaned up: {Path(image_path).name}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {image_path}: {e}")


# Global converter instance
try:
    pdf_converter = PDFConverter()
except Exception as e:
    logger.error(f"Failed to initialize PDF converter: {e}")
    pdf_converter = None