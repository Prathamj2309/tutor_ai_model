# Generated from: lora_implementation_notebook.ipynb
# Converted at: 2026-03-27T11:31:03.001Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

max_seq_length = 4096
dtype = None
load_in_4bit = True

print("Downloading Phi-4-mini...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "microsoft/phi-4-mini-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("Injecting LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)
print("Model ready!")

from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)

def formatting_prompts_func(examples):
    conversations = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in conversations]
    return {"text": texts}

# Load the file you uploaded to the sidebar
dataset = load_dataset("json", data_files="jee_augmented_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5% Validation Split
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training on {len(train_dataset)} examples. Validating on {len(eval_dataset)} examples.")

from trl import SFTTrainer
from transformers import TrainingArguments

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 15,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        eval_strategy = "steps", # <--- THE FIX IS HERE
        eval_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("Starting Training...")
trainer_stats = trainer.train()

import shutil

# Save the adapter locally in Colab
model.save_pretrained("phi-4-jee-math-lora")
tokenizer.save_pretrained("phi-4-jee-math-lora")

# Zip the folder so you can download it easily
shutil.make_archive("phi-4-jee-math-lora", 'zip', "phi-4-jee-math-lora")
print("Saved and zipped! Look in the left sidebar for 'phi-4-jee-math-lora.zip'.")
print("Right-click it and press 'Download' to save it to your laptop.")

from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# 1. Flip Unsloth into 2x Faster Inference Mode
FastLanguageModel.for_inference(model)

test_question = "Find the shortest distance between the parallel lines 3x - 4y + 7 = 0 and 3x - 4y - 5 = 0."

messages = [
    {"role": "user", "content": test_question}
]

# 2. THE FIX: Separate formatting and tokenization
# First, render the raw text with the <|im_start|> tags
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize = False, # Keep it as a string first
    add_generation_prompt = True,
)

# Second, tokenize it explicitly into a PyTorch dictionary
inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

print("\n" + "="*50)
print(f"USER QUESTION: {test_question}")
print("="*50 + "\n")
print("MODEL OUTPUT:\n")

text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# 3. Unpack the inputs dictionary using **inputs
_ = model.generate(
    **inputs, # <--- This safely unpacks the input_ids and attention_mask tensors
    streamer = text_streamer,
    max_new_tokens = 2000,
    temperature = 0.1,
    use_cache = True
)

from google.colab import drive
import shutil

# 1. Mount your Google Drive
drive.mount('/content/drive')

# 2. Copy the zip file directly to the root of your Google Drive
source_file = "phi-4-jee-math-lora.zip"
destination = "/content/drive/MyDrive/phi-4-jee-math-lora.zip"

try:
    shutil.copy(source_file, destination)
    print("-> SUCCESS! The file is now safely in your Google Drive.")
except FileNotFoundError:
    print("-> ERROR: The zip file wasn't found. Did you run the shutil.make_archive cell?")