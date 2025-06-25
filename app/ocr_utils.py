from __future__ import annotations

from io import BytesIO
from PIL import Image
import pytesseract


def ocr_image(data: bytes) -> str:
    """Return text recognized from an image using Tesseract."""
    try:
        img = Image.open(BytesIO(data))
    except Exception:
        return ""
    try:
        text = pytesseract.image_to_string(img)
    except Exception:
        return ""
    return text.strip()
