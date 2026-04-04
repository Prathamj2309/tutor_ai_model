# TutorAI: Personalized IIT-JEE Doubt Resolution & Assessment Platform 🎓

![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

TutorAI is a full-stack, multimodal AI web application designed to help IIT-JEE students master Physics, Chemistry, and Mathematics (PCM). By serving highly optimized, domain-specifically fine-tuned Small Language Models (SLMs) entirely locally, TutorAI acts as a personalized, cost-free digital tutor. 

It solves complex doubts via text and image uploads, tracks student weaknesses over time using long-term memory, and dynamically generates custom mock tests to improve performance.

---

## 🚀 Key Features

* **Domain-Specific Experts:** Powered by custom fine-tuned `phi-4-mini` models, trained heavily on advanced JEE-level reasoning and Chain-of-Thought (CoT) problem-solving.
* **Multimodal Doubt Resolution:** Students can type questions or upload images of textbook problems, diagrams, and equations (processed via Python OCR/Vision pipeline).
* **Long-Term Memory & User Profiles:** Integrated with Supabase, the system tracks individual student progress, remembers past concepts they struggled with, and tailors future explanations to their specific learning curve.
* **Dynamic Test Generation:** Based on historical query data, the FastAPI backend analyzes weak points and prompts the local AI to instantly generate customized, JEE-pattern multiple-choice quizzes.
* **100% Local AI Inference:** Models are quantized to `Q4_K_M` GGUF format and served via `llama.cpp` in Python, allowing the AI to run locally on consumer hardware without expensive third-party API calls.

---

## 🏗️ System Architecture & Tech Stack

The project utilizes a clean separation of concerns, ensuring high performance and scalability.

### 1. Frontend (User Interface)
* **Framework:** React (Vite)
* **Styling:** Tailwind CSS, Framer Motion (for smooth chat animations)
* **State Management:** React Hooks & Context API

### 2. Backend (Main Brain & AI Inference)
* **Framework:** Python, FastAPI
* **AI Engine:** `llama-cpp-python` (for running local `.gguf` models)
* **Computer Vision:** Python-based OCR libraries (e.g., `pytesseract` or `surya-ocr`) for extracting math from images.

### 3. Database & Authentication
* **Platform:** Supabase
* **Database:** PostgreSQL (Stores users, chat history, and quiz attempts)
* **Auth:** Supabase Auth (Email/Password & OAuth)
* **Integration:** `supabase-py` client used directly inside the FastAPI backend.

---

## 📂 Project Structure

```text
TutorAI/
│
├── frontend/                 # React (Vite) Web Application
│   ├── src/
│   │   ├── components/       # Chat UI, Quiz UI, Image Uploader
│   │   ├── pages/            # Dashboard, Login, Chat
│   │   ├── utils/            # Supabase JS Client for Auth only
│   │   └── App.jsx
│   ├── package.json
│   └── tailwind.config.js
│
├── backend/                  # Python FastAPI Server
│   ├── models/               # Place your .gguf models here!
│   ├── routes/               # API endpoints (chat, quiz, history)
│   ├── services/             # LLM Inference, OCR, Supabase DB logic
│   ├── main.py               # FastAPI entry point
│   └── requirements.txt      # Python dependencies
│
├── supabase/                 # Supabase SQL Schemas and migrations
│   └── schema.sql            # Table definitions (Users, Messages, Quizzes)
│
└── README.md