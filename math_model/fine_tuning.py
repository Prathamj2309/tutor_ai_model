import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, TextStreamer

# ==========================================
# 1. MODEL & LORA SETUP
# ==========================================
max_seq_length = 4096 
dtype = None 
load_in_4bit = True 

print("Downloading Phi-4-mini...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-4-mini-instruct", # Using the stable Unsloth repo
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

# ==========================================
# 2. DATASET PREPARATION
# ==========================================
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)

def formatting_prompts_func(examples):
    conversations = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in conversations]
    return {"text": texts}

print("Loading and mapping dataset...")
dataset = load_dataset("json", data_files="jee_augmented_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5% Validation Split
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training on {len(train_dataset)} examples. Validating on {len(eval_dataset)} examples.")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
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
        eval_strategy = "steps", 
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

# Save the model locally after training finishes
print("Training complete. Saving adapter...")
model.save_pretrained("phi-4-jee-math-lora")
tokenizer.save_pretrained("phi-4-jee-math-lora")

# ==========================================
# 4. INFERENCE TEST
# ==========================================
if __name__ == "__main__":
    
    print("\nFlipping to Inference Mode for final test...")
    # 1. Flip Unsloth into 2x Faster Inference Mode
    FastLanguageModel.for_inference(model)

    test_question = "Find the shortest distance between the parallel lines 3x - 4y + 7 = 0 and 3x - 4y - 5 = 0."

    messages = [
        {"role": "user", "content": test_question}
    ]

    # 2. Separate formatting and tokenization
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

    # 3. Generate output
    _ = model.generate(
        **inputs, 
        streamer = text_streamer,
        max_new_tokens = 2000,
        temperature = 0.1, 
        use_cache = True
    )