"""
Reward functions for GRPO Chemistry RL Training.

Each function takes a list of model completions (and optionally ground-truth answers)
and returns a list of float rewards scored per completion.
"""

import re


def reward_format(completions: list, **kwargs) -> list[float]:
    """
    Reward for following the expected output format:
    <think>...</think> followed by \\boxed{...}
    
    Returns 0.5 if the format is correct, 0.0 otherwise.
    """
    pattern = r"<think>.*?</think>\s*\\boxed\{.*?\}"
    return [0.5 if re.search(pattern, str(c), re.DOTALL) else 0.0 for c in completions]


def chem_soft_accuracy_reward(completions: list, answer: list, **kwargs) -> list[float]:
    """
    Soft accuracy reward for chemistry answers.
    
    - Exact match:   2.0
    - Partial match:  1.5 (cleaned answer is a substring of prediction or vice versa)
    - No match:       0.0
    """
    rewards = []
    for comp, ans in zip(completions, answer):
        match = re.search(r"\\boxed\{(.*?)\}", str(comp))
        if not match:
            rewards.append(0.0)
            continue

        pred = match.group(1).strip().lower()
        gt = str(ans).strip().lower()

        if pred == gt:
            rewards.append(2.0)
            continue

        def clean(text):
            text = re.sub(r"_\{.*?\}", "", text)  # remove _{...} subscripts fully
            text = text.replace("$", "").replace("\\", "")
            text = text.replace("moles", "mol").replace("grams", "g").replace("joules", "j")
            return text.strip()

        if clean(gt) in clean(pred) or clean(pred) in clean(gt):
            rewards.append(1.5)
        else:
            rewards.append(0.0)
    return rewards


def equation_usage_reward(completions: list, **kwargs) -> list[float]:
    """
    Reward for using chemical equations/arrows in the thinking process.
    
    Returns 0.5 if arrows (→, ⇌, etc.) are found in <think> block, 0.0 otherwise.
    """
    rewards = []
    for comp in completions:
        think_match = re.search(r"<think>(.*?)</think>", str(comp), re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue
        thought = think_match.group(1)
        if any(arrow in thought for arrow in ["->", "\\rightarrow", "\\rightleftharpoons", "⇌"]):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def thought_length_reward(completions: list, **kwargs) -> list[float]:
    """
    Reward for producing sufficiently detailed reasoning.
    
    - >500 chars: 0.3
    - >100 chars: 0.1
    - Otherwise:  0.0
    """
    rewards = []
    for comp in completions:
        think_match = re.search(r"<think>(.*?)</think>", str(comp), re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue
        length = len(think_match.group(1).strip())
        if length > 500:
            rewards.append(0.3)
        elif length > 100:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards
