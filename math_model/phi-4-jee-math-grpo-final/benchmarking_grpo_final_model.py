# Generated from: notebook3528349734.ipynb
# Converted at: 2026-03-27T16:05:26.065Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Kaggle needs Unsloth installed fresh since it's not a native library there
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
!pip install sympy pandas tqdm

import os
import pandas as pd
import torch
import re
from tqdm import tqdm
from sympy import sympify, simplify

# Bypass cache bug natively for Kaggle's working directory
os.environ["HF_HOME"] = "/kaggle/working/huggingface_cache"

# MANDATORY: Unsloth must be imported first
from unsloth import FastLanguageModel

# ==========================================
# 1. KAGGLE PATH CONFIGURATION
# ==========================================
# Change "jee-math-assets" to whatever you named your dataset in step 3
DATASET_NAME = "model data"

# Path to your test file in Kaggle's read-only input folder
TEST_DATA_FILE = f"/kaggle/input/datasets/prathamjain2309/model-data/test_50_colab.jsonl"

# Path to your LoRA/GRPO folder in Kaggle's read-only input folder
# Example: /kaggle/input/jee-math-assets/phi-4-jee-math-grpo
MODEL_PATH = f"/kaggle/input/datasets/prathamjain2309/model-data/phi-4-jee-math-grpo-final-20260326T184609Z-3-001/phi-4-jee-math-grpo-final" 

# Save outputs to Kaggle's writable working directory so you can download it later
OUTPUT_CSV_NAME = "/kaggle/working/benchmark_results_grpo.csv"

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
FastLanguageModel.for_inference(model) # Turns on 2x faster inference

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
    
    outputs = model.generate(
        **inputs,
        max_new_tokens = 450, 
        temperature = 0.1, 
        use_cache = True,
        pad_token_id = tokenizer.eos_token_id
    )
    
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
import time

# Load the benchmark you already ran
df = pd.read_csv("/kaggle/input/datasets/prathamjain2309/benchmark/benchmark_results_grpo.csv")

# 1. Formatting Success
format_rate = df['Format_Correct'].mean() * 100

# 2. Latency / Token generation estimation
# We know the model generated roughly ~918 characters per response on average.
# Assuming ~4 characters per token, that's ~230 tokens per response.
# If your benchmark took ~10 minutes (600 seconds) for 50 questions:
total_time_seconds = 5261 # Adjust this to how long your script actually took
avg_time_per_question = total_time_seconds / 50
estimated_tps = 230 / avg_time_per_question

print("=== FINAL PIPELINE METRICS ===")
print(f"Format Adherence Rate: {format_rate:.1f}%")
print(f"Average Inference Time per Question: {avg_time_per_question:.1f} seconds")
print(f"Estimated Generation Speed: {estimated_tps:.1f} Tokens/Second")
print("==============================")