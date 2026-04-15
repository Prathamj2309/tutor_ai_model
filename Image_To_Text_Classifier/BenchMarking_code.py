# ============================================================
# STEP 1: Install dependencies
# # ============================================================
# !pip install -U bitsandbytes>=0.46.1 transformers qwen-vl-utils
# !pip install pandas matplotlib seaborn

# ============================================================
# STEP 2: Imports
# ============================================================
import torch, time, gc, os, zipfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass, field
from typing import List

# ============================================================
# STEP 3: Config
# ============================================================
ZIP_PATH     = "BENCHMARK_DATASET.zip"
EXTRACT_DIR  = "BENCHMARK_DATASET"
IMAGE_EXTS   = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

@dataclass
class BenchmarkSample:
    image_path: str
    sample_id: str = ""
    expected_subject: str = "Unknown"      # no ground truth in flat zip
    expected_keywords: List[str] = field(default_factory=list)

# ============================================================
# STEP 4: Extract ZIP & Auto-Build Dataset (flat folder)
# ============================================================
def extract_zip(zip_path, extract_to):
    print(f"📦 Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"✅ Extracted to: {extract_to}/")

def auto_build_dataset(extract_dir):
    """
    Flat zip — all images mixed, no subfolders, no subject labels.
    Recursively finds every image file and creates one sample per image.
    """
    extract_path = Path(extract_dir)
    all_images = sorted([
        p for p in extract_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ])

    if not all_images:
        print("⚠️  No images found inside the zip!")
        return []

    samples = [
        BenchmarkSample(
            image_path=str(img),
            sample_id=img.stem,           # filename (no extension) as ID
            expected_subject="Unknown",   # flat zip → no ground truth
            expected_keywords=[],
        )
        for img in all_images
    ]

    print(f"✅ Found {len(samples)} images in flat zip.")
    print(f"   Sample IDs (first 5): {[s.sample_id for s in samples[:5]]}")
    return samples

# ============================================================
# STEP 5: Load Model
# ============================================================
def load_model(mode="4bit"):
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"\n🤖 Loading model in [{mode}] mode...")

    if mode == "4bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, quantization_config=quant_cfg
        )
    elif mode == "8bit":
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True, quantization_config=quant_cfg
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Model ready.")
    return model, processor

# ============================================================
# STEP 6: Inference
# ============================================================
PROMPT = """
Analyze this image and do the following:
1. Identify subject (Physics / Chemistry / Mathematics)
2. Extract the full question clearly

Give answer in this format:
Subject: <subject>
Question: <question>
"""

def run_inference(model, processor, image: Image.Image, max_new_tokens=128):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs_kwargs = {
        "text": [text], "images": image_inputs,
        "padding": True, "return_tensors": "pt"
    }
    if video_inputs:
        inputs_kwargs["videos"] = video_inputs

    inputs = processor(**inputs_kwargs).to(model.device)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    latency = time.perf_counter() - t_start

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output, latency, sum(len(t) for t in trimmed)

# ============================================================
# STEP 7: Parse model output → subject & question
# ============================================================
def parse_output(output: str):
    """Extract subject and question from model's formatted output."""
    parsed_subject  = "Unknown"
    parsed_question = ""

    for line in output.split("\n"):
        line = line.strip()
        if line.lower().startswith("subject:"):
            parsed_subject = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("question:"):
            parsed_question = line.split(":", 1)[-1].strip()

    # Normalize subject
    sub_lower = parsed_subject.lower()
    if "math" in sub_lower:
        parsed_subject = "Mathematics"
    elif "physics" in sub_lower or "phy" in sub_lower:
        parsed_subject = "Physics"
    elif "chem" in sub_lower:
        parsed_subject = "Chemistry"

    return parsed_subject, parsed_question

def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

# ============================================================
# STEP 8: Main Benchmark Runner
# ============================================================
def run_benchmark(modes=["4bit"], max_new_tokens=128, warmup_runs=1):
    # Extract zip
    if not os.path.exists(EXTRACT_DIR):
        extract_zip(ZIP_PATH, EXTRACT_DIR)
    else:
        print(f"📂 '{EXTRACT_DIR}' already exists, skipping extraction.")

    dataset = auto_build_dataset(EXTRACT_DIR)
    if not dataset:
        print("❌ Empty dataset. Aborting.")
        return pd.DataFrame()

    all_results = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()}  |  Total images: {len(dataset)}")
        print(f"{'='*60}")

        model, processor = load_model(mode)

        # Warmup pass
        print(f"\n🔥 Warmup ({warmup_runs} run(s))...")
        dummy = Image.new("RGB", (224, 224), color=(100, 100, 100))
        for _ in range(warmup_runs):
            run_inference(model, processor, dummy, max_new_tokens=32)
        print("✅ Warmup done.\n")

        for idx, sample in enumerate(dataset):
            print(f"[{idx+1}/{len(dataset)}] {sample.sample_id}", end=" ... ")

            try:
                image      = Image.open(sample.image_path).convert("RGB")
                mem_before = get_gpu_memory_mb()

                output, latency, num_tokens = run_inference(
                    model, processor, image, max_new_tokens
                )

                mem_after       = get_gpu_memory_mb()
                parsed_sub, parsed_q = parse_output(output)
                tokens_per_sec  = round(num_tokens / latency, 2) if latency > 0 else 0

                result = {
                    "mode"             : mode,
                    "sample_id"        : sample.sample_id,
                    "image_path"       : sample.image_path,
                    "image_size"       : f"{image.size[0]}x{image.size[1]}",
                    "predicted_subject": parsed_sub,
                    "extracted_question": parsed_q,
                    "latency_sec"      : round(latency, 3),
                    "tokens_generated" : num_tokens,
                    "tokens_per_sec"   : tokens_per_sec,
                    "gpu_mem_before_mb": round(mem_before, 1),
                    "gpu_mem_after_mb" : round(mem_after, 1),
                    "gpu_mem_delta_mb" : round(mem_after - mem_before, 1),
                    "full_output"      : output.strip(),
                    "status"           : "success",
                }

                print(f"✅  Subject: {parsed_sub} | {latency:.2f}s | {tokens_per_sec} tok/s")

            except Exception as e:
                print(f"❌ ERROR: {e}")
                result = {
                    "mode": mode, "sample_id": sample.sample_id,
                    "image_path": sample.image_path, "status": f"error: {e}"
                }

            all_results.append(result)

        del model, processor
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return pd.DataFrame(all_results)

# ============================================================
# STEP 9: Visualization
# ============================================================
def plot_results(df: pd.DataFrame):
    df_ok = df[df["status"] == "success"].copy()
    if df_ok.empty:
        print("No successful results to plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Qwen2.5-VL Benchmark — Flat Dataset", fontsize=16, fontweight="bold")

    # 1. Latency distribution
    ax = axes[0, 0]
    ax.hist(df_ok["latency_sec"], bins=20, color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_title("Latency Distribution"); ax.set_xlabel("Seconds"); ax.set_ylabel("Count")

    # 2. Tokens per second distribution
    ax = axes[0, 1]
    ax.hist(df_ok["tokens_per_sec"], bins=20, color="coral", edgecolor="black", alpha=0.8)
    ax.set_title("Tokens/sec Distribution"); ax.set_xlabel("Tokens/sec"); ax.set_ylabel("Count")

    # 3. GPU Memory Delta
    ax = axes[0, 2]
    ax.hist(df_ok["gpu_mem_delta_mb"], bins=20, color="mediumseagreen", edgecolor="black", alpha=0.8)
    ax.set_title("GPU Memory Delta (MB)"); ax.set_xlabel("MB"); ax.set_ylabel("Count")

    # 4. Predicted subject distribution (from model output)
    ax = axes[1, 0]
    df_ok["predicted_subject"].value_counts().plot(
        kind="bar", ax=ax, color="mediumpurple", edgecolor="black")
    ax.set_title("Predicted Subject Distribution")
    ax.set_ylabel("Count"); ax.tick_params(axis='x', rotation=30)

    # 5. Latency per sample (scatter)
    ax = axes[1, 1]
    ax.scatter(range(len(df_ok)), df_ok["latency_sec"].values,
               color="tomato", alpha=0.6, s=25)
    ax.axhline(df_ok["latency_sec"].mean(), color="black", linestyle="--", label="Mean")
    ax.set_title("Latency per Sample")
    ax.set_xlabel("Sample Index"); ax.set_ylabel("Seconds"); ax.legend()

    # 6. Success vs Failed
    ax = axes[1, 2]
    status_counts = df["status"].apply(lambda x: "Success" if x == "success" else "Failed").value_counts()
    status_counts.plot(kind="pie", ax=ax, autopct="%1.0f%%",
                       colors=["mediumseagreen", "tomato"], startangle=90)
    ax.set_title("Success vs Failed"); ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📁 Plot saved → benchmark_results.png")

# ============================================================
# STEP 10: Summary
# ============================================================
def print_summary(df: pd.DataFrame):
    df_ok = df[df["status"] == "success"]
    total, success, failed = len(df), len(df_ok), len(df) - len(df_ok)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"  Total: {total} | Success: {success} | Failed: {failed}")
    print(f"{'='*60}")
    print(f"  Avg Latency        : {df_ok['latency_sec'].mean():.3f}s")
    print(f"  Avg Tokens/sec     : {df_ok['tokens_per_sec'].mean():.2f}")
    print(f"  Avg GPU Mem Delta  : {df_ok['gpu_mem_delta_mb'].mean():.1f} MB")
    print(f"  Total Tokens Gen   : {df_ok['tokens_generated'].sum()}")
    print(f"\n  Predicted Subject Breakdown:")
    for subj, cnt in df_ok["predicted_subject"].value_counts().items():
        print(f"    {subj:<15}: {cnt} images ({cnt/success*100:.1f}%)")
    print("="*60)

    df.to_csv("benchmark_full_results.csv", index=False)
    print("\n📁 Full results → benchmark_full_results.csv")

# ============================================================
# STEP 11: RUN
# ============================================================
results_df = run_benchmark(modes=["4bit"], max_new_tokens=128, warmup_runs=1)
print_summary(results_df)
plot_results(results_df)
