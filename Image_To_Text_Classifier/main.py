# # Install required packages first
# !pip install -U bitsandbytes>=0.46.1
# !pip install -U transformers
# !pip install qwen-vl-utils

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# 1. Setup Model and Processor
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Use Qwen2_5_VLForConditionalGeneration specifically
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config
)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# 2. Load and Prepare Image
image_path = "Math.png"  # Make sure this file is uploaded to Colab
image = Image.open(image_path).convert("RGB")

# 3. Create the Prompt
prompt = """
Analyze this image and do the following:
1. Identify subject (Physics / Chemistry / Mathematics)
2. Extract the full question clearly
3. Describe the diagram in the question clearly , Do not miss any detail and describe properly , if any 
4. If there are any chemical compounds then use IUPAC nomenclature properly  
5. Do not solve question just describe it clearly

Give answer in this format:
Subject: <subject>
Question: <question>
"""

# 4. Prepare Inputs using Chat Template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

inputs_kwargs = {
    "text": [text],
    "images": image_inputs,
    "padding": True,
    "return_tensors": "pt",
}

if video_inputs:
    inputs_kwargs["videos"] = video_inputs

inputs = processor(**inputs_kwargs)
inputs = inputs.to(model.device)

# 5. Generate Output
print("Processing image... please wait.")
generated_ids = model.generate(**inputs, max_new_tokens=250)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n===== OUTPUT =====\n")
print(output_text[0])