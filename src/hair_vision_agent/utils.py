import base64
import os
from typing import Optional


class ImageUtils:
    """Utility functions for image processing."""

    @staticmethod
    def encode_image_to_base64(image_path: str) -> Optional[str]:
        """
        Encode an image file to base64 string.
        """
        try:
            if not os.path.exists(image_path):
                return None

            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            return None

    @staticmethod
    def validate_image_path(image_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate that an image path exists and is accessible.
        Returns (is_valid, error_message)
        """
        if not image_path:
            return False, "Image path is required"

        if not os.path.exists(image_path):
            return False, f"Image file not found: {image_path}"

        if not os.path.isfile(image_path):
            return False, f"Path is not a file: {image_path}"

        # Check if it's a valid image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        _, ext = os.path.splitext(image_path.lower())
        if ext not in valid_extensions:
            return False, f"Unsupported image format: {ext}"

        return True, None
