import io
import torch
import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

_vision_model = None
_vision_processor = None
_is_loaded = False

def load_vision_model():
    """Lazy-load the deep vision model into VRAM."""
    global _vision_model, _vision_processor, _is_loaded
    if _is_loaded:
        return
    
    print("[OCR] Loading Qwen2.5-VL-3B-Instruct Model (Optimized)...")
    
    # Use standard 4-bit quant with CPU offload if VRAM is tight
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    _vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.float16, 
        device_map="auto", # auto handles CPU offload if needed
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    # Optimized resolution settings for speed
    # max_pixels=512*28*28 ≈ 400,000 pixels
    _vision_processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=256*28*28,
        max_pixels=768*28*28 
    )
    _is_loaded = True

def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from an image using Qwen2.5-VL-3B.
    Includes resolution optimization for faster inference.
    """
    try:
        from qwen_vl_utils import process_vision_info

        if not _is_loaded:
            load_vision_model()
            
        # Pre-resize large images to speed up processing
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if max(image.size) > 1024:
            ratio = 1024 / max(image.size)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text", 
                        "text": "Identify the text, math expressions, and questions in this image. Format with LaTeX for math. Extract only the question and options."
                    },
                ],
            }
        ]
        
        text_prompt = _vision_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = _vision_processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            # Ensure model is on correct device if possible
            inputs = {k: v.to(_vision_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print("[OCR] Running optimized inference...")
        with torch.no_grad():
            generated_ids = _vision_model.generate(
                **inputs, 
                max_new_tokens=400, # Reduce tokens for faster response
                do_sample=False     # Greedy search is faster
            )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = _vision_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        result = output_text[0].strip()
        print(f"[OCR] Done. Result length: {len(result)}")
        return result
        
    except Exception as e:
        print(f"[OCR] Error during extraction: {e}")
        return f"Extraction Error: {str(e)}"
