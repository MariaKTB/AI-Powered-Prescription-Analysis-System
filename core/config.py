"""
DocuVault - Configuration Module
Centralized configuration management
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class Config:
    """Main configuration class for DocuVault"""
    
    # Project
    PROJECT_NAME = "DocuVault"
    VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    STORAGE_DIR = BASE_DIR / "storage"
    UPLOADS_DIR = STORAGE_DIR / "uploads"
    EXPORTS_DIR = STORAGE_DIR / "exports"
    CACHE_DIR = STORAGE_DIR / "cache"
    DB_PATH = STORAGE_DIR / "docuvault.db"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM Settings
    DEFAULT_LLM_MODEL = "gpt-4o-mini"
    FALLBACK_LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0
    LLM_MAX_TOKENS = 4096
    
    # OCR Settings
    OCR_LANGUAGE = "en"
    OCR_USE_GPU = False
    OCR_CONFIDENCE_THRESHOLD = 0.5
    
    # Document Processing
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "pdf", "tiff"]
    BATCH_SIZE = 10
    
    # Database
    DB_ECHO = False
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.STORAGE_DIR, cls.UPLOADS_DIR, cls.EXPORTS_DIR, cls.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            errors.append("No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
        
        return True
    
    @classmethod
    def initialize(cls):
        """Initialize configuration"""
        cls.setup_directories()
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            format=cls.LOG_FORMAT,
            level=cls.LOG_LEVEL,
            colorize=True
        )
        logger.info(f"{cls.PROJECT_NAME} v{cls.VERSION} initialized")


# Initialize on import
Config.initialize()
