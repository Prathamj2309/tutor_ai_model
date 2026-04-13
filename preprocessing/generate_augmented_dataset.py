import os
import re
import json
import random
import time
import ast
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Nvidia Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def generate_batched_explanations(batch_data):
    """
    Takes a list of dictionaries containing question data.
    """
    problems_text = ""
    for item in batch_data:
        problems_text += f"\n=== [Problem ID: {item['id']}] ===\n"
        problems_text += f"Question:\n{item['question_prompt']}\n"
        problems_text += f"Final Solution: {item['solution']}\n"
        problems_text += f"Old Explanation: {item['explanation']}\n"

    prompt = f"""
You are an expert JEE Mathematics professor. I am going to give you a batch of {len(batch_data)} math problems.
For EVERY SINGLE problem, generate a comprehensive, step-by-step chain of thought to teach a student model. Mention laws/theorems. Reason through elimination if options exist.

CRITICAL INSTRUCTION:
You MUST output your response EXACTLY in this XML format for each problem. Do not skip any problems. Do not combine them.

<result id="THE_PROBLEM_ID">
<think>
Step-by-step mathematical reasoning...
</think>
<answer>
The final concise answer here (e.g., \\boxed{{A}} or \\boxed{{42}}).
</answer>
</result>

Here are the {len(batch_data)} problems:
{problems_text}
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3500 # Adjusted for 5 questions
            )
            return parse_batch_response(response.choices[0].message.content, batch_data)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Warning] API Error on batch. Retrying... ({e})")
                time.sleep(5)
            else:
                print(f"[Error] Batch failed completely.")
                return []

def parse_batch_response(raw_text, batch_data):
    successful_results = []
    
    result_blocks = re.findall(r'<result id="(\d+)">(.*?)</result>', raw_text, re.DOTALL)
    original_map = {str(item['id']): item['question_prompt'] for item in batch_data}
    
    for str_id, block_content in result_blocks:
        think_match = re.search(r"<think>(.*?)</think>", block_content, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", block_content, re.DOTALL)
        
        if think_match and answer_match and str_id in original_map:
            think_text = think_match.group(1).strip()
            answer_text = answer_match.group(1).strip()
            original_q = original_map[str_id]
            
            successful_results.append({
                "conversations": [
                    {"role": "user", "content": original_q},
                    {"role": "assistant", "content": f"<think>\n{think_text}\n</think>\n{answer_text}"}
                ],
                "source_id": int(str_id)
            })
        else:
            print(f"  [Parse Error] Dropped Problem ID {str_id} due to bad formatting.")
            
    return successful_results

def main():
    try:
        df = pd.read_csv("C:/Users/Asus/tutor_ai_model/preprocessing/df_math (1).csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("C:/Users/Asus/tutor_ai_model/preprocessing/df_math (1).csv", encoding="cp1252")

    if 'solution' not in df.columns: df['solution'] = ""
    if 'explanation' not in df.columns: df['explanation'] = ""
    df['solution'] = df['solution'].fillna("")
    df['explanation'] = df['explanation'].fillna("")
    
    completed_indices = set()
    output_file = "jee_augmented_dataset_v2.jsonl"
    
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: completed_indices.add(json.loads(line).get("source_id"))
                    except: pass
                        
    print(f"\nFound {len(completed_indices)} already completed questions.")
    
    valid_indices = df[df['question'].notna()].index.tolist()
    uncompleted_indices = list(set(valid_indices) - completed_indices)
    
    target_total = 3000
    needed_questions = target_total - len(completed_indices)
    
    if needed_questions <= 0:
        print("You have already reached your target!")
        return
        
    random.seed(42)
    target_indices = random.sample(uncompleted_indices, min(needed_questions, len(uncompleted_indices)))
    
    # === MICRO-BATCHING LOGIC ===
    BATCH_SIZE = 3
    batches = [target_indices[i:i + BATCH_SIZE] for i in range(0, len(target_indices), BATCH_SIZE)]
    
    print(f"Divided {len(target_indices)} questions into {len(batches)} batches of {BATCH_SIZE}.")
    print("Starting generation on Nvidia API (Llama 3.1 70B)...\n")

    with open(output_file, "a", encoding="utf-8") as f:
        for batch_num, current_batch_indices in enumerate(batches, 1):
            print(f"Processing Batch {batch_num}/{len(batches)} (Contains {len(current_batch_indices)} questions)...")
            
            batch_data = []
            for idx in current_batch_indices:
                row = df.loc[idx]
                q_text = str(row['question'])
                
                options_raw = row.get('options', None)
                if pd.notna(options_raw) and str(options_raw).strip() != "":
                    try: 
                        parsed_options = ast.literal_eval(str(options_raw))
                        if parsed_options:
                            q_text += "\n\nOptions:\n"
                            for opt in parsed_options:
                                q_text += f"{opt.get('identifier', '')}: {opt.get('content', '')}\n"
                    except: pass 
                
                batch_data.append({
                    "id": idx,
                    "question_prompt": q_text.strip(),
                    "question": str(row['question']).strip(),
                    "solution": str(row['solution']).strip(),
                    "explanation": str(row['explanation']).strip()
                })
            
            results = generate_batched_explanations(batch_data)
            
            if results:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                print(f"-> Saved {len(results)}/{len(current_batch_indices)} items successfully.")
            else:
                print("-> Batch failed entirely. Moving to next.")


    print("\nDataset generation finished!")

if __name__ == "__main__":
    main()