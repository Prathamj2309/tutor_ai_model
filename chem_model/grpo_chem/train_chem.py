"""
Chemistry GRPO RL Training Pipeline

This script performs Group Relative Policy Optimization (GRPO) on a
chemistry-focused LLM (Qwen-based, via Unsloth LoRA adapters) to improve
its chain-of-thought reasoning and answer accuracy on chemistry MCQs.

Usage (Kaggle / Colab with T4 GPU):
    python train.py \
        --model_path /kaggle/input/datasets/risl345345/dataset2/lora/lora \
        --data_path /kaggle/input/datasets/risl345345/dataset2/chemistry_ft_optimal_cot.csv \
        --output_dir /kaggle/working/grpo_chemistry
"""

import argparse
import os

import torch
import transformers.utils.hub
import unsloth
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

from rewards import (
    reward_format,
    chem_soft_accuracy_reward,
    equation_usage_reward,
    thought_length_reward,
)
from data import prepare_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chemistry GRPO RL Training with Unsloth"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/kaggle/input/datasets/risl345345/dataset2/lora/lora",
        help="Path to the pre-trained SFT LoRA adapters.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/kaggle/input/datasets/risl345345/dataset2/chemistry_ft_optimal_cot.csv",
        help="Path to the chemistry fine-tuning CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/kaggle/working/grpo_chemistry",
        help="Directory to save the final trained model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate for GRPO training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ==========================================
    # 1. PATCH UNSLOTH FOR GRPO
    # ==========================================
    PatchFastRL("GRPO", FastLanguageModel)
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv(
        "HF_HOME", "~/.cache/huggingface/hub"
    )

    # ==========================================
    # 2. LOAD MODEL (SFT LoRA ADAPTERS)
    # ==========================================
    print(f"Loading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )
    print("✅ Model loaded successfully!")

    # ==========================================
    # 3. PREPARE DATASET
    # ==========================================
    rl_train, rl_eval, test_dataset = prepare_dataset(
        csv_path=args.data_path,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    # ==========================================
    # 4. GRPO CONFIGURATION
    # ==========================================
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=512,
        use_vllm=False,
        report_to="none",
        optim="adamw_8bit",
        fp16=True,   # T4 GPUs MUST use fp16
        bf16=False,  # T4 GPUs physically cannot run bf16
        logging_steps=10,
        max_steps=args.max_steps,
    )

    # TRL bug fix: manually create the dictionary TRL is looking for
    model.warnings_issued = {}

    # ==========================================
    # 5. TRAIN
    # ==========================================
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            reward_format,
            chem_soft_accuracy_reward,
            equation_usage_reward,
            thought_length_reward,
        ],
        args=training_args,
        train_dataset=rl_train,
        eval_dataset=rl_eval,
    )

    print("Starting GRPO Chemistry Training...")
    torch.cuda.empty_cache()
    trainer.train()

    # ==========================================
    # 6. SAVE MODEL
    # ==========================================
    final_model_dir = os.path.join(args.output_dir, "final_grpo_chemistry_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✅ Training complete! Model saved to: {final_model_dir}")


if __name__ == "__main__":
    main()
