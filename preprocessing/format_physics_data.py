import pandas as pd
import json
import ast

print("Loading physics_with_cot.csv...")
df = pd.read_csv("physics_with_cot.csv")

formatted_data = []
skipped = 0
mcq_count = 0
integer_count = 0

for index, row in df.iterrows():
    question = str(row['question']).strip() if pd.notna(row['question']) else ""
    cot = str(row['cot']).strip() if pd.notna(row['cot']) else ""
    
    # Check if options exist and aren't just an empty list '[]' or 'NaN'
    raw_options = str(row['options']).strip() if pd.notna(row['options']) else ""
    is_mcq = bool(raw_options and raw_options != "[]" and raw_options.lower() != "nan")

    if not question or not cot:
        skipped += 1
        continue

    # ==========================================
    # ROUTE 1: MULTIPLE CHOICE QUESTIONS (MCQ)
    # ==========================================
    if is_mcq:
        # Prompt includes the options
        user_content = f"Solve the following JEE Physics problem step-by-step:\n\nQuestion: {question}\nOptions: {raw_options}"
        
        # Clean the correct_option (e.g., converts '["B"]' to 'B')
        raw_correct = str(row['correct_option']).strip() if pd.notna(row['correct_option']) else ""
        final_answer = raw_correct.replace('["', '').replace('"]', '').replace("['", "").replace("']", "")
        
        if final_answer:
            mcq_count += 1
        else:
            skipped += 1
            continue

    # ==========================================
    # ROUTE 2: INTEGER / NUMERICAL QUESTIONS
    # ==========================================
    else:
        # Prompt explicitly states it is numerical and omits options
        user_content = f"Solve the following numerical JEE Physics problem step-by-step:\n\nQuestion: {question}"
        
        # Pulls directly from the answer column
        final_answer = str(row['answer']).strip() if pd.notna(row['answer']) else ""
        
        if final_answer:
            integer_count += 1
        else:
            skipped += 1
            continue

    # 3. Build the Assistant Response
    assistant_content = f"<think>\n{cot}\n</think>\n\n\\boxed{{{final_answer}}}"
    
    # 4. Construct the standard JSONL format
    conversation = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }
    formatted_data.append(conversation)

# Save the final dataset
output_filename = "physics_training_data.jsonl"
with open(output_filename, "w", encoding="utf-8") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Successfully formatted {len(formatted_data)} total problems.")
print(f"   -> MCQs preserved: {mcq_count}")
print(f"   -> Integer problems preserved: {integer_count}")
if skipped > 0:
    print(f"⚠️ Skipped {skipped} rows due to missing critical data.")
print(f"📁 Saved to: {output_filename}")