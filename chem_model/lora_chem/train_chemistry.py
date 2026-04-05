import os
import json
import torch
import pandas as pd
import argparse
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

# 1. HELPER FUNCTIONS
def format_options(options_raw):
    """Parses JSON-like strings from CSV into a clean readable format."""
    try:
        # Standardizing JSON format for terminal/local storage
        # Replace single quotes with double quotes for valid JSON parsing
        opts = json.loads(str(options_raw).replace("'", '"'))
        return "\n".join([f"{o['identifier']}: {o['content']}" for o in opts])
    except:
        return str(options_raw)

def create_text_col(row):
    """Formats the data into the ChatML structure for Phi-4."""
    formatted_opts = format_options(row['options'])
    return (
        f"<|im_start|>system\nYou are a Chemistry Expert. Solve the doubt using step-by-step reasoning (Chain of Thought).<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {row['question']}\nOptions:\n{formatted_opts}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['solution']}<|im_end|>"
    )

def main():
    # 2. CONFIGURATION
    # You can change these variables or use argparse to pass them via terminal
    MODEL_NAME = "unsloth/phi-4-mini-instruct"
    CSV_FILE = "chemistry_ft_optimal_cot.csv"
    OUTPUT_DIR = "phi4_chemistry_lora"
    LOG_DIR = "train_logs"

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Ensure the CSV is in the same directory.")
        return

    # 3. DATA PREPARATION
    print("Loading and splitting dataset...")
    df = pd.read_csv(CSV_FILE)
    df['text'] = df.apply(create_text_col, axis=1)
    
    # 90/10 Split for Validation Loss tracking
    train_df, eval_df = train_test_split(df[['text']], test_size=0.10, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # 4. LOAD MODEL & TOKENIZER
    print(f"Loading {MODEL_NAME} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 5. TRAINING ARGUMENTS
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100, # Set to 300-500 for a full training run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 10,
        per_device_eval_batch_size = 1,
        output_dir = LOG_DIR,
        optim = "adamw_8bit",
        seed = 3407,
        report_to = "none", # Disables WandB/Tensorboard prompts in terminal
    )

    # 6. INITIALIZE TRAINER
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = training_args,
        packing = True, 
    )

    # 7. EXECUTE TRAINING
    print("Starting training loop...")
    trainer.train()

    # 8. SAVE THE ADAPTERS
    print(f"Saving fine-tuned adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete. You can now run test_chemistry.py.")

if __name__ == "__main__":
    main()