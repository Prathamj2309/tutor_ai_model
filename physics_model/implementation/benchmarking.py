import os
import re
import ast
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_PATH = "/jee-physics-grpo-final"
DATA_PATH = "/content/drive/MyDrive/physics_model/physics_with_cot.csv" # <-- UPDATE THIS TO YOUR PHYSICS CSV
OUTPUT_PATH = "/content/drive/MyDrive/physics_model/physics_benchmark_results.csv"
MAX_SEQ_LENGTH = 2048
NUM_TEST_QUESTIONS = 50

# ==========================================
# 2. LOAD MODEL & TOKENIZER
# ==========================================
print(f"Loading Physics SFT adapter weights from: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False,
)

FastLanguageModel.for_inference(model)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},
)
print("Model loaded successfully!")

# ==========================================
# 3. DATA PREPARATION FUNCTIONS
# ==========================================
def clean_text(text):
    if pd.isna(text) or str(text).strip() == "": return None
    text = str(text)
    text = re.sub(r"<img[^>]*>", "", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<\/?p>", "", text)
    text = re.sub(r"<br\s*\/?>", "\n", text)
    return re.sub(r"\s+", " ", text).strip()

def format_inference_row(row):
    question = clean_text(row.get('question', ''))
    options_raw = row.get('options', None)
    user_content = str(question) + "\n\n"

    parsed_options = []
    if pd.notna(options_raw) and str(options_raw).strip() != "":
        try: parsed_options = ast.literal_eval(str(options_raw))
        except: pass

    if parsed_options:
        user_content += "Options:\n"
        for opt in parsed_options:
            user_content += f"{opt.get('identifier', '')}: {opt.get('content', '')}\n"
        try:
            expected_ans = ast.literal_eval(str(row['correct_option']))
            expected_ans = expected_ans[0] if isinstance(expected_ans, list) else expected_ans
        except:
            expected_ans = str(row['correct_option']).strip('[]"\'')
        instruction = "This is a multiple-choice physics question. Respond EXACTLY with the correct option letter enclosed in a LaTeX box (e.g., \\boxed{A})."
    else:
        expected_ans = str(row.get('solution', '')).strip()
        instruction = "This is a numerical physics question. Respond EXACTLY with strictly the integer or decimal mathematical value enclosed in a LaTeX box (e.g., \\boxed{42})."

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
        "expected_answer": str(expected_ans).strip(),
        "question": question
    }

# ==========================================
# 4. LOAD & SAMPLE DATASET
# ==========================================
print(f"Loading dataset from {DATA_PATH}...")
try:
    df_physics = pd.read_csv(DATA_PATH)
    df_physics = df_physics.dropna(subset=['question'])

    # Split to ensure we test on unseen data
    _, test_df = train_test_split(df_physics, test_size=0.1, random_state=42)
    test_df = test_df.sample(n=min(NUM_TEST_QUESTIONS, len(test_df)), random_state=99)

    print(f"Running inference on {len(test_df)} unseen physics questions.")
    test_data = [format_inference_row(row) for _, row in test_df.iterrows()]
except FileNotFoundError:
    print(f"ERROR: Could not find dataset at {DATA_PATH}. Please update the path.")
    test_data = []

# ==========================================
# 5. EXECUTE BENCHMARK
# ==========================================
results = []
if test_data:
    print(f"\n🚀 Benchmarking model logic on {len(test_data)} questions...\n")

    for item in tqdm(test_data, total=len(test_data)):
        inputs = tokenizer([item["prompt"]], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract reasoning and boxed answer
        think_match = re.search(r"<think>(.*?)</think>", generated_text, flags=re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else "No <think> tags found."

        box_match = re.search(r"\\boxed\{(.*?)\}", generated_text)
        extracted_answer = box_match.group(1).strip() if box_match else "No \\boxed{} answer found."

        results.append({
            "Question": item["question"][:60] + "...",
            "Expected_Answer": item["expected_answer"],
            "Model_Extracted_Answer": extracted_answer,
            "Result": "✅" if item["expected_answer"].lower() == extracted_answer.lower() else "❌",
            "Raw_Model_Output": generated_text
        })

# ==========================================
# 6. FINAL REPORT
# ==========================================
if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    results_df['Correct'] = results_df['Result'] == "✅"
    correct_count = results_df['Correct'].sum()
    accuracy = (correct_count / len(results_df)) * 100

    print("\n" + "="*50)
    print(f"📊 PHYSICS BENCHMARK RESULTS")
    print("="*50)
    print(results_df[['Question', 'Expected_Answer', 'Model_Extracted_Answer', 'Result']].to_string(index=False))
    print(f"\n🎯 FINAL LOGIC ACCURACY: {accuracy:.1f}%")
    print("="*50)