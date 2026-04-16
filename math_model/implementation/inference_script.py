# Generated from: inference-script (2).ipynb
# Converted at: 2026-04-15T15:03:27.950Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# To run the file, go on kaggle, and split the code into several blocks as i have numbered. T4 gpu will be needed and the base sft/grpo model will be needed during execution. The dataset is on github.

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

# !kaggle kernels output prathamjain2309/grpo-implementation-notebook-v2 -p /kaggle/working/

%%capture
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1" # Extra 30% context lengths
!pip install --upgrade -qqq uv

# Exact dependency block provided
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth vllm
else:
    try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
    except: _numpy = "numpy"; _pil = "pillow"
    try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
    except: is_t4 = False
    _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
    !uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth
    !uv pip install -qqq {_triton}
!uv pip install transformers==4.56.2
!uv pip install --no-deps trl==0.22.2



# ==========================================
# 1. IMPORTS & CONFIGURATION
# ==========================================
import ast
import pandas as pd
import re
import torch
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MAX_SEQ_LENGTH = 8092
NUM_TEST_QUESTIONS = 10 # Adjust to run on the full unseen set



import re
import ast
import torch
import sympy
import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from sklearn.model_selection import train_test_split
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
# Using the local GRPO final model
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
MODEL_PATH = os.path.join(project_root, "math_model", "grpo_final_model_updated")
DATA_PATH = os.path.join(project_root, "math_model", "data", "jee_augmented_dataset_v2.jsonl") # Updated to point to local data dir
OUTPUT_PATH = os.path.join(script_dir, "phi-4-inference_results_v2.3.csv")

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print(f"Loading SFT adapter weights from: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
    attn_implementation="eager",
)

FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)
print("Model loaded successfully!")

# ==========================================
# 3. EXACT DATA PREPARATION FUNCTIONS
# ==========================================
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "": return None
    text = str(text)
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<\/?p>", "", text)
    text = re.sub(r"<br\s*\/?>", "\n", text)
    # FIXED: Preserve newlines but clean up horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text) 
    return text.strip()

def format_inference_row(row):
    conversations = row.get('conversations', [])
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None

    # Extract user and assistant messages from the ShareGPT format
    user_content = next((msg.get('content', '') for msg in conversations if msg.get('role') == 'user'), "")
    assistant_content = next((msg.get('content', '') for msg in conversations if msg.get('role') == 'assistant'), "")
    
    user_content = clean_text(user_content)
    if not user_content:
        return None

    # 1. Extract Ground Truth from the Assistant's response
    box_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", assistant_content)
    expected_ans = box_match.group(1).strip().replace('$', '') if box_match else ""

    # 2. Map Options directly from the User's text
    options_map = {}
    # Looks for "A: value", "B) value" recognizing newlines
    option_matches = re.finditer(r"([A-D])[:\)]\s*(.+?)(?=(?:\n[A-D][:\)]|$))", user_content, re.IGNORECASE)
    for opt_match in option_matches:
        letter = opt_match.group(1).upper()
        val = opt_match.group(2).strip().replace('$', '')
        options_map[letter] = val

    if options_map:
        instruction = "This is a multiple-choice question. Reason concisely, do not show redundant calculations. Respond EXACTLY with the correct option letter enclosed in a tag (e.g., \\boxed{A}). Refer to the options and only answer from one of them strictly. Make sure that a previous step in calculation is not the same as the current step.(INTERNAL: maximum sequence length is 4096 tokens, so you must answer before you finish these tokens)"
    else:
        instruction = "This is an integer question. Respond EXACTLY with strictly the integer mathematical value enclosed in a LaTeX box (e.g., \\boxed{42}). Do not show redundant calculations. Make sure that a previous step in calculation is not the same as the current step.(INTERNAL: maximum sequence length is 4096 tokens, so you must answer before you finish these tokens)"

    formatted_user_prompt = f"""{instruction}

Respond EXACTLY in this format:
<think>
step-by-step reasoning
</think>
\\boxed{{final answer}}

Question:
{user_content}"""

    prompt_chatml = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return {
        "prompt": prompt_chatml,
        "expected_answer": expected_ans,
        "question": user_content,
        "options_map": options_map
    }
# ==========================================
# 4. EVALUATION FUNCTION (Mirrors Training)
# ==========================================
def evaluate_answer(generated, gt, options_map):
    if generated == "No \\boxed{} answer found.":
        return False
        
    acceptable_targets = [gt]
    
    if gt in options_map:
        acceptable_targets.append(options_map[gt])
    else:
        for letter, val in options_map.items():
            if val == gt:
                acceptable_targets.append(letter)
                break

    for target in acceptable_targets:
        if generated == target:
            return True
        if target in ['A', 'B', 'C', 'D'] and re.search(rf"\b{target}\b", generated, re.IGNORECASE):
            return True
        try:
            gen_clean = generated.replace('^', '**').replace('\\frac', '').replace('}{', '/').replace('{', '').replace('}', '').replace('\\', '')
            tgt_clean = target.replace('^', '**').replace('\\frac', '').replace('}{', '/').replace('{', '').replace('}', '').replace('\\', '')
            
            gen_expr = sympy.sympify(gen_clean)
            tgt_expr = sympy.sympify(tgt_clean)
            
            if sympy.simplify(gen_expr - tgt_expr) == 0:
                return True
        except Exception:
            pass 
            
    return False

# ==========================================
# 5. DATA SPLITTING & SAMPLING
# ==========================================
print("Loading dataset...")
df_math = pd.read_json(DATA_PATH, lines=True)

# FIXED: Drop rows where 'conversations' is missing, not 'question'
df_math = df_math.dropna(subset=['conversations'])

_, sft_eval_df = train_test_split(df_math, test_size=0.1, random_state=42)
test_df = sft_eval_df.sample(n=min(NUM_TEST_QUESTIONS, len(sft_eval_df)), random_state=79)
print(f"Running inference on {len(test_df)} unseen questions.")

# FIXED: Filter out any None returns from formatting
test_data = [res for res in (format_inference_row(row) for _, row in test_df.iterrows()) if res is not None]
import gc
from collections import Counter

# ==========================================
# 6. INFERENCE LOOP (MAJORITY VOTING)
# ==========================================
results = []
correct_count = 0
NUM_GENERATIONS = 5 # 5 passes per question

for item in tqdm(test_data, total=len(test_data)):
    inputs = tokenizer([item["prompt"]], return_tensors="pt").to("cuda")
    
    extracted_answers = []
    
    for i in range(NUM_GENERATIONS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3000, # Reduced from 4500 to prevent OOM
                temperature=0.6,     # Increased for creative diversity
                do_sample=True,      # REQUIRED when temp > 0
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Immediate Memory Cleanup to prevent OOM
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        # Extract the answer for this pass
        box_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", generated_text)
        extracted_answer = box_match.group(1).strip().replace('$', '') if box_match else "No \\boxed{} answer found."
        
        extracted_answers.append(extracted_answer)

    # --- MAJORITY VOTING LOGIC ---
    # Filter out failures so they don't accidentally win the vote
    valid_answers = [ans for ans in extracted_answers if ans != "No \\boxed{} answer found."]
    
    if not valid_answers:
        final_majority_answer = "No \\boxed{} answer found."
    else:
        # Count frequencies and pick the most common valid answer
        counter = Counter(valid_answers)
        final_majority_answer = counter.most_common(1)[0][0]
    
    # Grade the final majority answer
    is_correct = evaluate_answer(final_majority_answer, item["expected_answer"], item["options_map"])
    if is_correct:
        correct_count += 1
    
    results.append({
        "Question": item["question"],
        "Expected_Answer": item["expected_answer"],
        "Majority_Answer": final_majority_answer,
        "All_Extracted_Answers": str(extracted_answers), # Log all 5 so you can analyze variance later
        "Is_Correct": is_correct
    })
    
    # Clean up inputs before the next question
    del inputs
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 7. SAVE OUTPUT
# ==========================================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

accuracy = (correct_count / len(test_data)) * 100
print(f"\nInference Complete!")
print(f"Final Accuracy: {accuracy:.2f}% ({correct_count}/{len(test_data)})")
print(f"Results saved to {OUTPUT_PATH}")

# ==========================================
# 7. EVALUATE ACCURACY
# ==========================================
print("\n" + "="*50)
print("Evaluating Accuracy on Unseen Data...")

# Standardize strings to prevent false negatives from stray spaces or casing
results_df['Expected_Clean'] = results_df['Expected_Answer'].astype(str).str.strip().str.lower()
results_df['Model_Clean'] = results_df['Model_Extracted_Answer'].astype(str).str.strip().str.lower()

# Calculate matches
results_df['Correct'] = results_df['Expected_Clean'] == results_df['Model_Clean']

correct_count = results_df['Correct'].sum()
total_count = len(results_df)
accuracy = (correct_count / total_count) * 100

print(f"Total Questions Evaluated: {total_count}")
print(f"Total Correct: {correct_count}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print("="*50)

# Optional: Print a quick debug view of the failures
wrong_df = results_df[~results_df['Correct']]
if not wrong_df.empty:
    print("\nPreview of incorrect predictions (First 5):")
    # Displaying Expected vs Model side-by-side to see where it failed
    print(wrong_df[['Expected_Answer', 'Model_Extracted_Answer']].head(5).to_string(index=False))