import pandas as pd
from openai import OpenAI
import time

# 1. Setup NVIDIA Client
# Replace 'YOUR_NVIDIA_API_KEY' with your actual key
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def generate_cot(row):
    """
    Constructs a prompt and fetches the Chain of Thought from the NVIDIA model.
    """
    # Constructing the prompt using available columns
    prompt = f"""
    As an expert Physics tutor, provide a step-by-step 'Chain of Thought' derivation for this JEE Mains question.
    
    Question: {row.get('question', 'N/A')}
    Options: {row.get('options', 'N/A')}
    Correct Solution/Answer: {row.get('solution', 'N/A')}
    
    Instructions:
    1. Identify the core physics concepts involved.
    2. List the given variables.
    3. Show the step-by-step mathematical derivation or logical reasoning.
    4. Conclude with the final answer.
    
    Chain of Thought:
    """

    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct", # You can change this to your preferred NVIDIA NIM model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# 2. Load your data
df = pd.read_csv("physics_data.csv")

# 3. Apply the function
# Note: For large datasets, use df.head(10) first to test costs and speed.
print("Generating Chain of Thought... this may take a while.")
df['cot'] = df.apply(generate_cot, axis=1)

# 4. Save the updated dataset
df.to_csv("physics_with_cot.csv", index=False)
print("Done! Saved to physics_with_cot.csv")