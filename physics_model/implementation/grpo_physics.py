# ==========================================
# DEPENDENCY FIXES FOR NEW COLAB ACCOUNT
# ==========================================
import os
import transformers.utils.hub
# Manually define the missing cache variable to prevent llm_blender from crashing
transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "~/.cache/huggingface/hub")

import re
import pandas as pd
from datasets import Dataset
import torch
from sympy import sympify, simplify
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. LOAD YOUR NEWLY TRAINED SFT ADAPTER
# ==========================================
max_seq_length = 1024

print("Loading the Physics SFT Adapter from Google Drive...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/content/drive/MyDrive/physics_model/phi-4-jee-physics-lora",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

# ==========================================
# 2. PREPARE GRPO DATASET
# ==========================================
print("Loading Dataset from Google Drive...")
df = pd.read_csv("/content/drive/MyDrive/physics_model/physics_with_cot.csv")
grpo_data = []

for index, row in df.iterrows():
    question = str(row['question']).strip() if pd.notna(row['question']) else ""
    raw_options = str(row['options']).strip() if pd.notna(row['options']) else ""
    is_mcq = bool(raw_options and raw_options != "[]" and raw_options.lower() != "nan")

    if is_mcq:
        raw_correct = str(row['correct_option']).strip() if pd.notna(row['correct_option']) else ""
        truth = raw_correct.replace('["', '').replace('"]', '').replace("['", "").replace("']", "")
        prompt = f"Solve the following JEE Physics problem step-by-step:\n\nQuestion: {question}\nOptions: {raw_options}"
    else:
        truth = str(row['answer']).strip() if pd.notna(row['answer']) else ""
        prompt = f"Solve the following numerical JEE Physics problem step-by-step:\n\nQuestion: {question}"

    if question and truth:
        grpo_data.append({
            "prompt": [{"role": "user", "content": prompt}],
            "answer": truth
        })

dataset = Dataset.from_pandas(pd.DataFrame(grpo_data))

# ==========================================
# 3. MATH-STYLE REWARD FUNCTIONS
# ==========================================
def clean_math(expr):
    expr = str(expr)
    expr = expr.replace("$", "")
    expr = expr.replace("\\boxed{", "").replace("}", "")
    expr = expr.replace("\\frac", "")
    expr = expr.replace("\\over", "/")
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.strip()
    return expr

def symbolic_equal(a, b):
    try:
        a_expr = sympify(clean_math(a))
        b_expr = sympify(clean_math(b))
        return simplify(a_expr - b_expr) == 0
    except:
        return False

def format_reward_func(completions, **kwargs):
    rewards = []
    for output in completions:
        # TRL passes a list of dicts, we extract the content
        text = output[0]["content"]
        if "<think>" in text and "</think>" in text and "\\boxed{" in text:
            rewards.append(0.4)
        else:
            rewards.append(0.0)
    return rewards

def completion_reward_func(completions, **kwargs):
    rewards = []
    for output in completions:
        text = output[0]["content"]
        if re.search(r"\\boxed\{.*?\}\s*$", text.strip()):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, answer=None, prompts=None, **kwargs):
    rewards = []
    for i in range(len(completions)):
        text = completions[i][0]["content"]
        ground_truth = str(answer[i]).strip() if answer is not None else None

        match = re.search(r"\\boxed\{(.*?)\}", text)

        if match and ground_truth is not None:
            generated = match.group(1)

            # 1. Check symbolic equivalence first
            if symbolic_equal(generated, ground_truth):
                rewards.append(3.0)
                continue

            # 2. Check float eval equivalence
            try:
                if abs(float(eval(clean_math(generated))) - float(eval(clean_math(ground_truth)))) < 1e-6:
                    rewards.append(3.0)
                    continue
            except:
                pass

            # Partial reward if formatted correctly but math is wrong
            rewards.append(0.15)
        else:
            # Full penalty if no box is found
            rewards.append(0.1)

    return rewards

# ==========================================
# 4. GRPO TRAINING CONFIGURATION
# ==========================================
print("Starting GRPO Reinforcement Learning...")
training_args = GRPOConfig(
    output_dir = "/content/drive/MyDrive/physics_model/outputs_grpo_physics",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4,
    max_completion_length = 1024, # <-- ADDED BACK TO PREVENT 0.0 LOSS BUG
    max_steps = 250,
    optim = "paged_adamw_8bit",
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 5,
    report_to = "none",
)

# ---> THE FIX FOR THE TRL BUG <---
# We manually create this empty dictionary so TRL has a place to hide its warning
if not hasattr(model, "warnings_issued"):
    model.warnings_issued = {}

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [format_reward_func, accuracy_reward_func, completion_reward_func],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()

# ==========================================
# 5. SAVE THE ULTIMATE MODEL
# ==========================================
print("GRPO Complete. Saving the final Physics Model to Drive...")
final_path = "/content/drive/MyDrive/physics_model/jee-physics-grpo-final"
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Success! Model securely saved to {final_path}")