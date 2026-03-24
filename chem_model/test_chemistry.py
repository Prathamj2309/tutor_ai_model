import torch
from unsloth import FastLanguageModel

def run_test():
    # 1. Configuration
    # Ensure this path matches where you saved your model
    model_path = "phi4_chemistry_lora" 
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    print(f"Loading fine-tuned model from: {model_path}...")

    # 2. Load the model and tokenizer
    # We load the adapters directly using FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path, 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Prepare for inference (Optimizes memory and speed)
    FastLanguageModel.for_inference(model)

    # 4. Define your test case
    question = "What is the main product formed when ethanol is heated with excess concentrated sulfuric acid at 443 K?"
    options = "A: Ethane\nB: Ethene\nC: Diethyl ether\nD: Ethyl hydrogen sulfate"

    # 5. Format using ChatML Template
    messages = [
        {"role": "system", "content": "You are a Chemistry Expert. Solve the doubt using step-by-step reasoning (Chain of Thought)."},
        {"role": "user", "content": f"Question: {question}\nOptions:\n{options}"},
    ]

    # Convert to tokens
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    # 6. Generate reasoning
    print("\n--- Generating Chemistry Reasoning ---\n")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 512,
            use_cache = True,
            temperature = 0.1, # Keep temperature low for factual accuracy
            top_p = 0.9
        )

    # 7. Decode and Clean Output
    response = tokenizer.batch_decode(outputs)
    
    # Extract only the assistant's new response
    final_text = response[0].split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "")
    
    print(final_text.strip())

if __name__ == "__main__":
    run_test()