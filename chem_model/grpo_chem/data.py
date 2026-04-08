"""
Dataset preparation for Chemistry GRPO RL Training.

Loads the chemistry fine-tuning CSV, formats prompts using a chat template,
and splits into RL train / eval / test sets.
"""

import pandas as pd
from datasets import Dataset


def prepare_dataset(csv_path: str, tokenizer, seed: int = 3407):
    """
    Load and prepare the chemistry dataset for GRPO training.

    Args:
        csv_path: Path to the chemistry_ft_optimal_cot.csv file.
        tokenizer: HuggingFace tokenizer (with chat template support).
        seed: Random seed for reproducible splits.

    Returns:
        rl_train: ~270 rows for GRPO training
        rl_eval:  ~30 rows for live reward monitoring
        test_dataset: ~30 rows locked away for final evaluation
    """
    df = pd.read_csv(csv_path)

    def prepare_rl_example(row):
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a Chemistry Professor. Break down the molecular interaction "
                    "in <think> tags, then provide the final result in \\boxed{}."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {row['question']}\nOptions: {row['options']}",
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            ),
            "answer": row["solution"],
        }

    dataset = Dataset.from_pandas(df).map(prepare_rl_example)

    # Full dataset split — 90% was used for SFT, we only work with the 10% holdout
    all_splits = dataset.train_test_split(test_size=0.1, seed=seed)
    rl_pool = all_splits["test"]  # ~330 rows — our entire RL budget

    # Step 1: carve out ~30 rows as final test set (never touched during training)
    temp = rl_pool.train_test_split(test_size=0.09, seed=seed)
    test_dataset = temp["test"]  # ~30 rows — locked away for final evaluation

    # Step 2: split remainder into rl_train and rl_eval
    rl_split = temp["train"].train_test_split(test_size=0.1, seed=seed)
    rl_train = rl_split["train"]  # ~270 rows — GRPO trains on this
    rl_eval = rl_split["test"]    # ~30 rows  — monitors reward live

    print(
        f"RL Train: {len(rl_train)} | RL Eval: {len(rl_eval)} | Test (locked): {len(test_dataset)}"
    )

    return rl_train, rl_eval, test_dataset
