# Generated from: grpo_implementation_notebook.ipynb
# Converted at: 2026-03-27T11:30:45.734Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from datasets import load_dataset

ds = load_dataset("UtkarshM005/grafite-jee-mains-qna-no-img")

df = ds["train"].to_pandas()

df = ds["train"].to_pandas()
df.to_csv("jee_data_all.csv", index=False)

import pandas as pd
df_new = pd.read_csv('/content/jee_data_all.csv')

df['subject'].unique()

df_math = df[df['subject'] == 'maths']

df_math.head()

df_math = df_math.reset_index(drop = 'True')

df_math

import json
import pandas as pd

# 1. Build the prompt
def build_prompt(row):
    question_text = str(row['question'])
    options_data = str(row['options'])

    prompt = question_text + "\n\nOptions:\n"
    try:
        opts = json.loads(options_data)
        for opt in opts:
            prompt += f"{opt['identifier']}) {opt['content']}\n"
    except Exception:
        prompt += options_data

    return prompt

print("Formatting prompts...")
df_math['prompt'] = df_math.apply(build_prompt, axis=1)

# 2. THE FIX: Drop the existing 'answer' column before renaming 'solution'
if 'answer' in df_math.columns and 'solution' in df_math.columns:
    df_math = df_math.drop(columns=['answer'])

df_math = df_math.rename(columns={'solution': 'answer'})

# 3. Clean up and isolate columns
df_math = df_math.dropna(subset=['prompt', 'answer', 'chapter'])
rl_df = df_math[['prompt', 'answer', 'chapter', 'topic']].copy()

# 4. Stratified Sampling (Warning Fixed)
print("Sampling 150 balanced questions...")
num_chapters = rl_df['chapter'].nunique()
samples_per_chapter = max(1, 200 // num_chapters)

# Added include_groups=False to silence the DeprecationWarning
train_df = rl_df.groupby('chapter', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_chapter), random_state=42),
    include_groups=False
)

# Force exactly 200
if len(train_df) > 200:
    train_df = train_df.sample(200, random_state=42)
elif len(train_df) < 200:
    needed = 200 - len(train_df)
    filler = rl_df.drop(train_df.index, errors='ignore').sample(needed, random_state=42)
    train_df = pd.concat([train_df, filler])

# Ensure only the required columns are exported
train_df = train_df[['prompt', 'answer', 'chapter', 'topic']]

print(f"Final Train Set Size: {len(train_df)} questions.")

# 5. Export
train_df.to_json("train_rl_colab.jsonl", orient="records", lines=True)
print("Success! File train_rl_colab.jsonl is ready.")

# Install Unsloth and standard RL/PEFT libraries
!pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes xformers triton datasets

!unzip phi-4-jee-math-lora.zip -d phi-4-jee-math-lora

!pip install mergekit

!pip install llm-blender tyro rich wandb

!pip install weave

!pip install sympy pandas



import os
import unsloth
import transformers.utils.hub

from sympy import sympify, simplify
# Bypass cache bug
transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "~/.cache/huggingface/hub")

# MUST be first
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import torch
import re
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# 1. Load model
max_seq_length = 4096
print("Loading your SFT adapter weights...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="phi-4-jee-math-lora",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ---------------- REWARD FUNCTIONS ---------------- #

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


# ================== REWARD FUNCTIONS ==================

def format_reward_func(completions, **kwargs):
    rewards = []
    for output in completions:
        if "<think>" in output and "</think>" in output and "\\boxed{" in output:
            rewards.append(0.4)
        else:
            rewards.append(0.0)
    return rewards


def completion_reward_func(completions, **kwargs):
    rewards = []
    for output in completions:
        if re.search(r"\\boxed\{.*?\}\s*$", output.strip()):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward_func(completions, answer=None, prompts=None, **kwargs):
    rewards = []

    for i in range(len(completions)):
        output = completions[i]
        ground_truth = answer[i] if answer is not None else None

        match = re.search(r"\\boxed\{(.*?)\}", output)

        if match and ground_truth is not None:
            generated = match.group(1)

            if symbolic_equal(generated, ground_truth):
                rewards.append(3.0)
                continue

            try:
                if abs(float(eval(clean_math(generated))) - float(eval(clean_math(ground_truth)))) < 1e-6:
                    rewards.append(3.0)
                    continue
            except:
                pass

            rewards.append(0.15)

        else:
            rewards.append(0.1)

    return rewards

# ---------------- DATA ---------------- #

dataset = load_dataset("json", data_files="train_rl_colab.jsonl", split="train")

def enforce_chat_format(example):
    clean_prompt = re.sub(r"<.*?>", "", example["prompt"])

    formatted_user_prompt = f"""
Solve the following math problem.

Respond EXACTLY in this format:
<think>
step-by-step reasoning
</think>
\\boxed{{final answer}}

Question:
{clean_prompt}
"""

    return {
        "prompt": tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        ),
        "answer": example["answer"],
    }

dataset = dataset.map(enforce_chat_format)

# ---------------- TRAINING ---------------- #

training_args = GRPOConfig(
    output_dir="grpo_outputs",
    learning_rate=5e-6,
    num_train_epochs=1,

    per_device_train_batch_size=6,
    gradient_accumulation_steps=3,

    num_generations=3,

    max_completion_length=400,

    max_prompt_length=768,
    logging_steps=5,
    optim="adamw_8bit",
    report_to="none",
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        format_reward_func,
        accuracy_reward_func,
        completion_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

print("Starting GRPO Training...")
trainer.train()

# ---------------- SAVE ---------------- #

print("Saving RL-optimized model...")
model.save_pretrained("phi-4-jee-math-grpo")
tokenizer.save_pretrained("phi-4-jee-math-grpo")
print("Done!")

from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/phi-4-jee-math-grpo /content/drive/MyDrive/phi-4-jee-math-grpo-final

print("Transfer complete! Check your Google Drive.")

import pandas as pd
df_t = pd.read_csv('/content/jee_data_all.csv')
df_m = df_t[df_t['subject'] == 'maths']

import pandas as pd

print("Preparing the 50-question benchmark dataset...")

# 1. Grab the 50 random rows first
sampled_df = df_m.sample(n=50, random_state=42)

# 2. Extract ONLY the two columns we care about into a fresh DataFrame
# This completely destroys any duplicate column errors
clean_test_df = pd.DataFrame({
    "prompt": sampled_df["question"].tolist(),
    "answer": sampled_df["solution"].tolist()
})

# 3. Save it to the exact filename
file_name = "test_50_colab.jsonl"
clean_test_df.to_json(file_name, orient="records", lines=True)

print(f"SUCCESS: Saved 50 clean questions to '{file_name}'.")
print("You are clear to run the benchmark script now!")

# Convert your new GRPO model to a 4-bit GGUF file
model.save_pretrained_gguf(
    "phi-4-jee-math-grpo-gguf",
    tokenizer,
    quantization_method = "q4_k_m"
)

# This uses a wildcard (*.gguf) to grab the exact file Unsloth just created
# and drops it directly into the main folder of your Google Drive.
!cp *.gguf /content/drive/MyDrive/

print("GGUF successfully backed up to Google Drive!")

import pandas as pd
import torch
import re
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from sympy import sympify, simplify # Added the missing imports from your logic

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Run this script twice:
# First time -> MODEL_PATH = "phi-4-jee-math-lora" (Outputs SFT results)
# Second time -> MODEL_PATH = "phi-4-jee-math-grpo" (Outputs GRPO results)
MODEL_PATH = "phi-4-jee-math-grpo"
OUTPUT_CSV_NAME = f"benchmark_results_{MODEL_PATH.split('-')[-1]}.csv"
TEST_DATA_FILE = "test_50_colab.jsonl"

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print(f"Loading model: {MODEL_PATH} for inference...")
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 2x faster inference mode

# ==========================================
# 3. YOUR EXACT EVALUATION LOGIC
# ==========================================
def clean_math(expr):
    if expr is None: return ""
    expr = str(expr)
    expr = expr.replace("$", "").replace("\\boxed{", "").replace("}", "")
    expr = expr.replace("\\frac", "").replace("\\over", "/")
    expr = expr.replace("{", "").replace("}", "").strip()
    return expr

def symbolic_equal(a, b):
    if not a or not b: return False
    try:
        a_expr = sympify(clean_math(a))
        b_expr = sympify(clean_math(b))
        return simplify(a_expr - b_expr) == 0
    except:
        return False

def check_math_accuracy(generated, ground_truth):
    """Replicates the accuracy_reward_func logic."""
    if symbolic_equal(generated, ground_truth):
        return True
    try:
        if abs(float(eval(clean_math(generated))) - float(eval(clean_math(ground_truth)))) < 1e-6:
            return True
    except:
        pass
    return False

# ==========================================
# 4. DATA LOADING & PROMPT FORMATTING
# ==========================================
print(f"Loading test questions from {TEST_DATA_FILE}...")
test_df = pd.read_json(TEST_DATA_FILE, lines=True)

def format_test_prompt(question_text):
    """Exactly matches the enforce_chat_format function from your training."""
    clean_prompt = re.sub(r"<.*?>", "", question_text)
    formatted_user_prompt = f"""
Solve the following math problem.

Respond EXACTLY in this format:
<think>
step-by-step reasoning
</think>
\\boxed{{final answer}}

Question:
{clean_prompt}
"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

# ==========================================
# 5. THE INFERENCE LOOP
# ==========================================
results = []
correct_math_count = 0
correct_format_count = 0

print(f"Starting evaluation on {len(test_df)} questions...")

for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Benchmarking"):
    prompt = format_test_prompt(row['prompt'])
    ground_truth = str(row['answer']).strip()

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the response (Matching max_completion_length from training)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 450,
        temperature = 0.1,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id
    )

    # Decode only the generated response
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # 1. Check Formatting
    has_think = "<think>" in generated_text and "</think>" in generated_text
    match = re.search(r"\\boxed\{(.*?)\}", generated_text)
    extracted_ans = match.group(1).strip() if match else None

    is_format_correct = has_think and (extracted_ans is not None)

    # 2. Check Math Accuracy
    is_math_correct = check_math_accuracy(extracted_ans, ground_truth)

    if is_format_correct: correct_format_count += 1
    if is_math_correct: correct_math_count += 1

    results.append({
        "Question": row['prompt'],
        "Ground_Truth": ground_truth,
        "Model_Answer": extracted_ans,
        "Format_Correct": is_format_correct,
        "Math_Correct": is_math_correct,
        "Full_Generation": generated_text
    })

# ==========================================
# 6. SAVE AND REPORT
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV_NAME, index=False)

accuracy = (correct_math_count / len(test_df)) * 100
format_rate = (correct_format_count / len(test_df)) * 100

print("\n" + "="*50)
print(f"BENCHMARK REPORT: {MODEL_PATH}")
print("="*50)
print(f"Total Questions Evaluated: {len(test_df)}")
print(f"Formatting Success Rate:   {format_rate:.2f}%")
print(f"Mathematical Accuracy:     {accuracy:.2f}%")
print("="*50)
print(f"Detailed logs saved to: {OUTPUT_CSV_NAME}")



import pandas as pd
import torch
import re
import json
from tqdm import tqdm
from unsloth import FastLanguageModel
from sympy import sympify, simplify # Added the missing imports from your logic

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Run this script twice:
# First time -> MODEL_PATH = "phi-4-jee-math-lora" (Outputs SFT results)
# Second time -> MODEL_PATH = "phi-4-jee-math-grpo" (Outputs GRPO results)
MODEL_PATH = "phi-4-jee-math-lora"
OUTPUT_CSV_NAME = f"benchmark_results_{MODEL_PATH.split('-')[-1]}.csv"
TEST_DATA_FILE = "test_50_colab.jsonl"

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print(f"Loading model: {MODEL_PATH} for inference...")
max_seq_length = 4096
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # 2x faster inference mode

# ==========================================
# 3. YOUR EXACT EVALUATION LOGIC
# ==========================================
def clean_math(expr):
    if expr is None: return ""
    expr = str(expr)
    expr = expr.replace("$", "").replace("\\boxed{", "").replace("}", "")
    expr = expr.replace("\\frac", "").replace("\\over", "/")
    expr = expr.replace("{", "").replace("}", "").strip()
    return expr

def symbolic_equal(a, b):
    if not a or not b: return False
    try:
        a_expr = sympify(clean_math(a))
        b_expr = sympify(clean_math(b))
        return simplify(a_expr - b_expr) == 0
    except:
        return False

def check_math_accuracy(generated, ground_truth):
    """Replicates the accuracy_reward_func logic."""
    if symbolic_equal(generated, ground_truth):
        return True
    try:
        if abs(float(eval(clean_math(generated))) - float(eval(clean_math(ground_truth)))) < 1e-6:
            return True
    except:
        pass
    return False

# ==========================================
# 4. DATA LOADING & PROMPT FORMATTING
# ==========================================
print(f"Loading test questions from {TEST_DATA_FILE}...")
test_df = pd.read_json(TEST_DATA_FILE, lines=True)

def format_test_prompt(question_text):
    """Exactly matches the enforce_chat_format function from your training."""
    clean_prompt = re.sub(r"<.*?>", "", question_text)
    formatted_user_prompt = f"""
Solve the following math problem.

Respond EXACTLY in this format:
<think>
step-by-step reasoning
</think>
\\boxed{{final answer}}

Question:
{clean_prompt}
"""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

# ==========================================
# 5. THE INFERENCE LOOP
# ==========================================
results = []
correct_math_count = 0
correct_format_count = 0

print(f"Starting evaluation on {len(test_df)} questions...")

for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Benchmarking"):
    prompt = format_test_prompt(row['prompt'])
    ground_truth = str(row['answer']).strip()

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Generate the response (Matching max_completion_length from training)
    outputs = model.generate(
        **inputs,
        max_new_tokens = 450,
        temperature = 0.1,
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id
    )

    # Decode only the generated response
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # 1. Check Formatting
    has_think = "<think>" in generated_text and "</think>" in generated_text
    match = re.search(r"\\boxed\{(.*?)\}", generated_text)
    extracted_ans = match.group(1).strip() if match else None

    is_format_correct = has_think and (extracted_ans is not None)

    # 2. Check Math Accuracy
    is_math_correct = check_math_accuracy(extracted_ans, ground_truth)

    if is_format_correct: correct_format_count += 1
    if is_math_correct: correct_math_count += 1

    results.append({
        "Question": row['prompt'],
        "Ground_Truth": ground_truth,
        "Model_Answer": extracted_ans,
        "Format_Correct": is_format_correct,
        "Math_Correct": is_math_correct,
        "Full_Generation": generated_text
    })

# ==========================================
# 6. SAVE AND REPORT
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV_NAME, index=False)

accuracy = (correct_math_count / len(test_df)) * 100
format_rate = (correct_format_count / len(test_df)) * 100

print("\n" + "="*50)
print(f"BENCHMARK REPORT: {MODEL_PATH}")
print("="*50)
print(f"Total Questions Evaluated: {len(test_df)}")
print(f"Formatting Success Rate:   {format_rate:.2f}%")
print(f"Mathematical Accuracy:     {accuracy:.2f}%")
print("="*50)
print(f"Detailed logs saved to: {OUTPUT_CSV_NAME}")