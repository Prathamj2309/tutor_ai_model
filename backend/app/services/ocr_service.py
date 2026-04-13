"""
OCR Service — backed by Image_To_Text_Classifier/main.py
---------------------------------------------------------
Imports load_model / run_inference / parse_output from the
Image_To_Text_Classifier package that lives one level above
the backend directory.
"""

import io
import os
import sys
from PIL import Image

# ── Make the project root importable so we can find Image_To_Text_Classifier ──
_BACKEND_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../backend
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)  # .../tutor_ai_model
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Image_To_Text_Classifier.main import load_model, run_inference, parse_output  # noqa: E402

# ── Lazy singletons ────────────────────────────────────────────────────────────
_model     = None
_processor = None
_is_loaded = False


def _ensure_loaded():
    global _model, _processor, _is_loaded
    if _is_loaded:
        return
    _model, _processor = load_model()
    _is_loaded = True


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract subject + question text from raw image bytes.
    Returns the raw model output string; the caller can
    optionally call parse_output() on it for structured data.
    """
    try:
        _ensure_loaded()

        # Resize large images for faster inference (same as standalone script)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(image.size) > 1024:
            ratio = 1024 / max(image.size)
            image = image.resize(
                (int(image.width * ratio), int(image.height * ratio)),
                Image.LANCZOS,
            )

        raw = run_inference(_model, _processor, image)
        return raw

    except Exception as e:
        print(f"[OCR] Error during extraction: {e}")
        return f"Extraction Error: {str(e)}"


def extract_parsed_from_image(image_bytes: bytes) -> dict:
    """
    Convenience wrapper — returns { subject, question, raw }.
    """
    raw = extract_text_from_image(image_bytes)
    if raw.startswith("Extraction Error:"):
        return {"subject": "Unknown", "question": raw, "raw": raw}
    return parse_output(raw)
