# TutorAI: Personalized IIT-JEE Doubt Resolution & Assessment Platform 🎓

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

TutorAI is a full-stack, multimodal AI web application designed to help IIT-JEE students master Physics, Chemistry, and Mathematics (PCM). By serving highly optimized, domain-specifically fine-tuned Small Language Models (SLMs) using a **Mixture-of-Experts (MoE)** architecture, TutorAI acts as a personalized digital tutor.

---

## 📊 Model Accuracy Benchmarks
The platform uses specialized LoRA adapters for each subject. Current performance metrics on JEE-level benchmarks are as follows:

| Subject | Accuracy | Model Base |
| :--- | :--- | :--- |
| **Chemistry** | **52%** | Phi-4-mini + Chem-GRPO |
| **Physics** | **34%** | Phi-4-mini + Physics-GRPO |
| **Mathematics** | **12%** | Phi-4-mini + Math-LORA |

*Note: Accuracy is measured on high-difficulty JEE Advanced style problems. Continuous fine-tuning is ongoing.*

---

## 🚀 Key Features

*   **Dynamic Mixture-of-Experts (MoE):** Automatically routes questions to specialized Physics, Chemistry, or Math adapters using a MiniLM-based context router.
*   **Vision-OCR Intelligence:** Powered by `Qwen2.5-VL-3B-Instruct`, the system can "see" textbook images, handle complex LaTeX equations, and extract textbook problems with high precision.
*   **150k+ Question JEE Dataset:** Integrated Mock Test Generator pulls from a verified database of over 150,000 JEE questions with chapter-wise tagging.
*   **Custom Exam Sessions:** Support for timed mock tests (up to 90 mins) with automated scoring and LaTeX-supported explanations.
*   **Long-Term Memory:** Tracks user performance and weak topics via Supabase, enabling personalized study paths.

---

## 🏗️ System Architecture & Tech Stack

### 1. Frontend (User Interface)
*   **Framework:** React (Vite)
*   **Styling:** Vanilla CSS + Tailwind, Framer Motion (for chat animations)
*   **Math Rendering:** React-Markdown + KaTeX (rehype-katex)

### 2. Backend (Main Brain & AI Inference)
*   **Framework:** Python, FastAPI
*   **LLM Engine:** `transformers` + `bitsandbytes` (4-bit quantization)
*   **Inference:** Phi-4-Mini-Instruct (Base) + Custom Adapters
*   **Vision:** Qwen2.5-VL-3B-Instruct (Optimized resolution & VRAM management)

### 3. Database & Authentication
*   **Platform:** Supabase
*   **Database:** PostgreSQL (Messages, Conversations, Quiz data)
*   **Auth:** Supabase Auth (Email/JWT)

---

## 📂 Project Structure

```text
TutorAI/
│
├── frontend/                 # React (Vite) Web Application
│   ├── src/
│   │   ├── components/       # Chat UI, Quiz UI, Image Uploader
│   │   ├── pages/            # Dashboard, Quiz, Vision OCR
│   │   └── hooks/            # useChat, useWeakness logic
│
├── backend/                  # Python FastAPI Server
│   ├── app/
│   │   ├── routers/          # API endpoints (chat, quiz, ocr)
│   │   ├── services/         # LLM (MoE), OCR (Qwen), Mock Test services
│   │   └── core/             # Configuration & DB connection
│   └── requirements.txt      # Python dependencies
│
├── physics_model/            # LoRA Adapters for Physics
├── chem_model/               # LoRA Adapters for Chemistry
├── math_model/               # LoRA Adapters for Mathematics
└── README.md
```

## 🛠️ Usage
1.  **Backend:** `python -m app.main` (Requires GPU with 6GB+ VRAM for full MoE + Vision capability)
2.  **Frontend:** `npm run dev`