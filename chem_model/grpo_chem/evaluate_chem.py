"""
Evaluation script for the GRPO-trained Chemistry model.

Runs greedy inference on a held-out test set and computes accuracy.
"""

import argparse
import re

import torch
from unsloth import FastLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the GRPO Chemistry model on a held-out test set."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained GRPO model directory.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the chemistry CSV file.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed (must match training to get same test split).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here to reuse the same split logic
    from data import prepare_dataset

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
    )

    # Get the locked test split
    _, _, test_dataset = prepare_dataset(
        csv_path=args.data_path,
        tokenizer=tokenizer,
        seed=args.seed,
    )

    # Evaluate
    print("\n--- Final Evaluation on Held-Out Test Set ---")
    model.eval()
    correct = 0
    total = len(test_dataset)

    for example in test_dataset:
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # greedy for evaluation
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # Check if answer appears in boxed output
        pred_match = re.search(r"\\boxed\{(.*?)\}", decoded)
        pred = pred_match.group(1).strip().lower() if pred_match else ""
        gt = str(example["answer"]).strip().lower()

        is_correct = pred == gt
        if is_correct:
            correct += 1

        print(f"Expected : {gt}")
        print(f"Predicted: {pred}")
        print(f"Correct  : {is_correct}")
        print("---")

    print(f"\nFinal Accuracy: {correct}/{total} = {correct / total * 100:.1f}%")


if __name__ == "__main__":
    main()
