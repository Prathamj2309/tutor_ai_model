import io
from PIL import Image
import pytesseract

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Returns the extracted text as a string.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Improve contrast for math notation
        image = image.convert('L')  # Grayscale
        text = pytesseract.image_to_string(image, config='--psm 6')
        return text.strip()
    except Exception as e:
        print(f"[OCR] Error extracting text: {e}")
        return ""
