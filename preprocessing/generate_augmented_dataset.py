import os
import re
import json
import random
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Nvidia Client (uses standard OpenAI SDK)
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def generate_augmentations(original_question, question_id):
    prompt = f"""
You are an expert JEE Mathematics professor.
Analyze the following original math problem, solve it internally, and then generate 3 NEW, isomorphic problems that test the exact same concepts but with different numbers, algebraic functions, or geometric orientations.

For EACH of the 3 new problems, you MUST format your output EXACTLY like this using XML tags:
<variation>
<question>The full text of the new question here...</question>
<think>Your step-by-step mathematical reasoning here...</think>
<answer>The final concise answer here (e.g., \\boxed{{4}})</answer>
</variation>

Original Problem:
{original_question}
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            return parse_variations(response.choices[0].message.content, question_id)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[Warning] Failed on Question {question_id}. Retrying... Error: {e}")
                time.sleep(5)
            else:
                print(f"[Error] Question {question_id} failed completely: {e}")
                return []

def parse_variations(raw_text, original_id):
    variations = []
    pattern = r"<variation>.*?<question>(.*?)</question>.*?<think>(.*?)</think>.*?<answer>(.*?)</answer>.*?</variation>"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    
    for match in matches:
        variations.append({
            "conversations": [
                {"role": "user", "content": match[0].strip()},
                {"role": "assistant", "content": f"<think>\n{match[1].strip()}\n</think>\n{match[2].strip()}"}
            ],
            "source_id": original_id
        })
    return variations

def main():
    try:
        df = pd.read_csv("jee_math.csv", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv("jee_math.csv", encoding="cp1252")

    jee_questions = df['question'].dropna().tolist()
    
    # --- SMART RESUME & SAMPLING LOGIC ---
    completed_indices = set()
    if os.path.exists("jee_augmented_dataset.jsonl"):
        with open("jee_augmented_dataset.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        completed_indices.add(data.get("source_id"))
                    except:
                        pass
                        
    print(f"\nFound {len(completed_indices)} already completed questions.")
    
    all_indices = set(range(len(jee_questions)))
    uncompleted_indices = list(all_indices - completed_indices)
    
    target_total = 300
    needed_questions = target_total - len(completed_indices)
    
    if needed_questions <= 0:
        print("You have already reached your 300-question target!")
        return
        
    # Pick random remaining questions for topic diversity
    random.seed(42) # Keeps the random selection consistent if you restart
    target_indices = random.sample(uncompleted_indices, min(needed_questions, len(uncompleted_indices)))
    
    print(f"Randomly selected {len(target_indices)} new questions to reach the 50% mark.")
    print("Starting generation on Nvidia API (Llama 3.1 70B)...\n")

    with open("jee_augmented_dataset.jsonl", "a", encoding="utf-8") as f:
        for count, idx in enumerate(target_indices, 1):
            q_text = jee_questions[idx]
            print(f"Processing target {count}/{len(target_indices)} (Original CSV Row {idx})...")
            
            result = generate_augmentations(q_text, idx)
            
            if result: 
                for item in result:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                print(f"-> Success! Saved 3 variations.")
            
            time.sleep(2) # Polite delay to respect free API limits

    print("\nDataset target reached! You are ready to train.")

if __name__ == "__main__":
    main()