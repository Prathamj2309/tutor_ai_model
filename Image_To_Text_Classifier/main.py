"""
Image-to-Text Extractor using Qwen2.5-VL-3B-Instruct
------------------------------------------------------
Can be used as a standalone script:
  python main.py                          # prompts for image path
  python main.py "C:\\path\\to\\image.png"  # pass path directly

Or imported as a module by the backend:
  from Image_To_Text_Classifier.main import load_model, run_inference, parse_output
"""

import sys
import os
import torch
from PIL import Image
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# ── Shared constants ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = """You are an image-to-text extractor. Your ONLY job is to read and transcribe what is written in the image.

STRICT RULES — You MUST follow all of them:
1. Identify the subject: Physics, Chemistry, or Mathematics.
2. Extract the FULL question text exactly as written in the image, including all LaTeX math.
3. List all answer options (A), (B), (C), (D) exactly as written. Do NOT mark, highlight, or indicate which one is correct.
4. If there is a diagram, circuit, or figure — describe it in full detail.
5. Use IUPAC nomenclature for any chemical compounds.

ABSOLUTE PROHIBITIONS — violating any of these is WRONG:
- DO NOT solve the question.
- DO NOT calculate or evaluate any expression.
- DO NOT reveal, hint at, mark, or mention the correct answer.
- DO NOT add any explanation, analysis, or reasoning.
- DO NOT include any text that was not present in the image.

Your output format MUST be exactly:
Subject: <Physics | Chemistry | Mathematics>
Question: <full question text with LaTeX>
Options:
(A) <option A text>
(B) <option B text>
(C) <option C text>
(D) <option D text>
"""

# ── 1. Load Model & Processor ─────────────────────────────────────────────────
def load_model():
    """
    Load Qwen2.5-VL-3B-Instruct with 4-bit quantization.
    Returns (model, processor).
    """
    print(f"[OCR] Loading model: {MODEL_NAME} (4-bit quantized)...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("[OCR] Model loaded successfully.")
    return model, processor


# ── 2. Run Inference ──────────────────────────────────────────────────────────
def run_inference(model, processor, image: Image.Image, max_new_tokens: int = 512) -> str:
    """
    Run the Qwen2.5-VL model on a PIL image and return the raw text output.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs_kwargs = {
        "text":           [text],
        "images":         image_inputs,
        "padding":        True,
        "return_tensors": "pt",
    }
    if video_inputs:
        inputs_kwargs["videos"] = video_inputs

    inputs = processor(**inputs_kwargs).to(model.device)

    print("[OCR] Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    result = output_text[0].strip()
    print(f"[OCR] Done. Output length: {len(result)} chars")
    return result


# ── 3. Parse Output ───────────────────────────────────────────────────────────
def parse_output(raw: str) -> dict:
    """
    Parse the model output into { subject, question, options, raw }.
    Handles the structured format:
        Subject: ...
        Question: ...
        Options:
        (A) ...
        (B) ...
    """
    subject = ""
    question_lines = []
    options = {}
    in_question = False
    in_options = False

    for line in raw.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("subject:"):
            subject = stripped.split(":", 1)[-1].strip()
            in_question = False
            in_options = False

        elif lower.startswith("question:"):
            q_text = stripped.split(":", 1)[-1].strip()
            if q_text:
                question_lines.append(q_text)
            in_question = True
            in_options = False

        elif lower.startswith("options"):
            in_question = False
            in_options = True

        elif in_options and len(stripped) > 3 and stripped[0] == "(" and stripped[2] == ")":
            key = stripped[1]           # A / B / C / D
            val = stripped[3:].strip()  # text after "(X)"
            options[key] = val

        elif in_question and stripped:
            question_lines.append(stripped)

    # Normalize subject
    sl = subject.lower()
    if "math" in sl:
        subject = "Mathematics"
    elif "phys" in sl:
        subject = "Physics"
    elif "chem" in sl:
        subject = "Chemistry"

    # Build clean question string — append options block if parsed
    question = "\n".join(question_lines)
    if options:
        opts_block = "\n".join(f"({k}) {v}" for k, v in sorted(options.items()))
        question = (question + "\n\n" + opts_block).strip() if question else opts_block

    return {
        "subject":  subject or "Unknown",
        "question": question or raw,
        "options":  options,
        "raw":      raw,
    }



# ── 4. Standalone entrypoint ──────────────────────────────────────────────────
def _get_image_path() -> str:
    if len(sys.argv) > 1:
        path = " ".join(sys.argv[1:]).strip().strip('"').strip("'")
    else:
        print("=" * 55)
        print("  Qwen2.5-VL Image-to-Text Extractor (Local)")
        print("=" * 55)
        path = input("\nEnter the full path to your image file:\n> ").strip().strip('"').strip("'")

    path = os.path.expandvars(os.path.expanduser(path))
    if not os.path.isfile(path):
        print(f"\n❌ File not found: {path}")
        sys.exit(1)

    ext = Path(path).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        print(f"\n⚠️  Unsupported file type: {ext}")
        sys.exit(1)

    return path


def main():
    image_path = _get_image_path()
    print(f"\n📷 Image: {image_path}")

    image = Image.open(image_path).convert("RGB")
    orig = image.size
    if max(image.size) > 1024:
        ratio = 1024 / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS
        )
        print(f"🔄 Resized {orig} → {image.size}")

    model, processor = load_model()
    raw = run_inference(model, processor, image)
    parsed = parse_output(raw)

    print("\n" + "=" * 55)
    print("  EXTRACTION RESULT")
    print("=" * 55)
    print(f"Subject : {parsed['subject']}")
    print(f"Question: {parsed['question']}")
    print("=" * 55)


if __name__ == "__main__":
    main()