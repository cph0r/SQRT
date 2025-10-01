"""
Production configuration for SQRT application.
"""

import os
import logging
from typing import Dict, Any


class Config:
    """Application configuration."""
    
    # Application settings
    APP_NAME = "SQRT - Selfie Quality Rater"
    APP_VERSION = "1.0.0"
    
    # Performance settings
    REALTIME_ANALYSIS_INTERVAL = float(os.getenv("REALTIME_INTERVAL", "2.0"))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))
    REALTIME_IMAGE_SIZE = (640, 480)  # Optimized for real-time
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Gradio settings
    GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    # Face detection settings
    FACE_MIN_SIZE = (40, 40)
    FACE_MAX_SIZE = (300, 300)
    FACE_SCALE_FACTOR = 1.2
    FACE_MIN_NEIGHBORS = 6
    
    @classmethod
    def setup_logging(cls):
        """Configure application logging."""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format=cls.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
            ]
        )
        
        # Reduce noise from external libraries
        logging.getLogger("gradio").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.INFO)
    
    @classmethod
    def get_launch_kwargs(cls) -> Dict[str, Any]:
        """Get Gradio launch configuration."""
        return {
            "server_name": cls.GRADIO_SERVER_NAME,
            "server_port": cls.GRADIO_SERVER_PORT,
            "share": cls.GRADIO_SHARE,
            "show_api": False,
            "show_error": True,
            "quiet": cls.LOG_LEVEL != "DEBUG",
        }

