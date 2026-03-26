import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Setup NVIDIA Client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

def generate_cot(row):
    """
    Constructs a prompt and fetches the Chain of Thought from the NVIDIA model.
    """
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
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing row: {e}")
        return None

# 3. Load your data
# Using a try-block in case the file name changed or isn't found
try:
    df = pd.read_csv("physics_data.csv")
    
    # 4. Apply the function
    print("Generating Chain of Thought... this may take a while.")
    # Testing on first 5 rows to ensure everything works before running full file
    # df = df.head(5) 
    df['cot'] = df.apply(generate_cot, axis=1)

    # 5. Save the updated dataset
    df.to_csv("physics_with_cot.csv", index=False)
    print("Done! Saved to physics_with_cot.csv")

except FileNotFoundError:
    print("Error: 'physics_data.csv' not found. Please check the file path.")