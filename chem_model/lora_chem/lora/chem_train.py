import os
import json
import torch
import pandas as pd
import shutil
import argparse
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

def format_options(options_raw):
    try:
        # Standardizing JSON format for terminal/local storage
        opts = json.loads(options_raw.replace("'", '"'))
        return "\n".join([f"{o['identifier']}: {o['content']}" for o in opts])
    except:
        return str(options_raw)

def create_text_col(row):
    formatted_opts = format_options(row['options'])
    return (
        f"<|im_start|>system\nYou are a Chemistry Expert. Solve the doubt using step-by-step reasoning (Chain of Thought).<|im_end|>\n"
        f"<|im_start|>user\nQuestion: {row['question']}\nOptions:\n{formatted_opts}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['solution']}<|im_end|>"
    )

def main():
    # 1. SETUP PARAMETERS
    model_name = "unsloth/phi-4-mini-instruct"
    dataset_file = "chemistry_ft_optimal_cot.csv"
    output_dir = "phi4_chemistry_lora"
    
    if not os.path.exists(dataset_file):
        print(f"Error: {dataset_file} not found in current directory.")
        return

    # 2. LOAD & PREPROCESS DATA
    print("Pre-processing dataset...")
    df = pd.read_csv(dataset_file)
    df['text'] = df.apply(create_text_col, axis=1)
    
    train_df, eval_df = train_test_split(df[['text']], test_size=0.10, random_state=42)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # 3. LOAD MODEL
    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # 4. TRAINING CONFIG
    training_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust for full training
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 10,
        output_dir = "train_outputs",
        optim = "adamw_8bit",
        report_to = "none", # Prevents unwanted cloud logging prompts
    )

    # 5. TRAIN
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

    print("Starting training...")
    trainer.train()

    # 6. SAVE LOCALLY
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()