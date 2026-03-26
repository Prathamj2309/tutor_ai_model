from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("1. Downloading base model (This will cache ~2.5GB locally)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Phi-4-mini-instruct", 
    torch_dtype=torch.float16, 
    device_map="cpu" # Using CPU so we don't crash your local GPU VRAM
)

print("2. Applying your JEE Math LoRA adapter...")
# MAKE SURE THIS PATH MATCHES YOUR EXTRACTED FOLDER NAME
model = PeftModel.from_pretrained(base_model, "./phi-4-jee-math-lora")

print("3. Merging weights permanently (This takes a few minutes of heavy CPU work)...")
model = model.merge_and_unload()

print("4. Saving the standalone model...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-4-mini-instruct")
model.save_pretrained("./JEE_Math_Merged")
tokenizer.save_pretrained("./JEE_Math_Merged")

print("Merge complete! You can now point Ollama to the JEE_Math_Merged folder.")