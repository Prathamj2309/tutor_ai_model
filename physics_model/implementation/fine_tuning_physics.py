import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template # <-- Added import
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

# ==========================================-
# 1. MODEL & LORA SETUP
# ==========================================-
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Downloading Phi-4-mini for Physics...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4-mini-instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Apply the standard ChatML template used by Phi-4
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

print("Injecting LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# ==========================================-
# 2. DATASET PREPARATION (THE FIX IS HERE)
# ==========================================-
print("Loading physics_training_data.jsonl...")
dataset = load_dataset("json", data_files="/content/drive/MyDrive/physics_model/physics_training_data.jsonl", split="train")

# Define how to convert the "messages" into a single training string
def apply_template(examples):
    messages = examples["messages"]
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    return {"text": texts}

# Map the function across your 3,457 rows
print("Formatting conversational data...")
dataset = dataset.map(apply_template, batched=True)

# ==========================================-
# 3. TRAINING CONFIGURATION
# ==========================================-
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text", # Now the trainer will successfully find the "text" column
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        # max_steps = 60, # Uncomment for a quick test run
        num_train_epochs = 1, # Run for 1 full epoch over your data
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_physics",
        report_to = "none",
    ),
)

print("Starting Physics SFT Training...")
trainer_stats = trainer.train()

print("Training complete. Saving physics adapter...")
model.save_pretrained("phi-4-jee-physics-lora")
tokenizer.save_pretrained("phi-4-jee-physics-lora")

# ==========================================-
# 4. IMMEDIATE INFERENCE TEST
# ==========================================-
if __name__ == "__main__":
    print("\nFlipping to Inference Mode for a quick test...")
    FastLanguageModel.for_inference(model)

    test_question = "A particle starts from rest and accelerates uniformly at 2 m/s^2 for 10 seconds. Calculate the final velocity and the distance covered."
    user_content = f"Solve the following numerical JEE Physics problem step-by-step:\n\nQuestion: {test_question}"

    messages = [{"role": "user", "content": user_content}]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True,
    )

    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

    print("\n" + "="*50)
    print(f"USER QUESTION: {test_question}")
    print("="*50 + "\n")
    print("MODEL OUTPUT:\n")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    _ = model.generate(
        **inputs,
        streamer = text_streamer,
        max_new_tokens = 1024,
        temperature = 0.1,
        use_cache = True
    )