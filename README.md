# LLM Sycophancy Detector

A minimalistic AI alignment tool that detects and scores sycophantic behavior in LLM conversations.

Paste in dialogue between a user and an AI, and get an instant sycophancy score from **1 (honest / pushes back)** to **5 (extremely sycophantic / excessive flattery)**.

## Live Demo

Hosted on Vercel — **[Try the Detector](https://llm-sycophancy-detector.vercel.app)**

---

## How It Works (High-Level Architecture)


```
├── index.html          # Clean, dark-mode single-page frontend
├── why.html            # Explanation page on RLHF training biases & risks
├── api/
│   └── score.js        # Vercel serverless proxy with secure key handling
├── rate_text.py        # Local CLI guardrail rater
├── evaluator.py        # Offline benchmarking pipeline
├── ingest_data.py      # Anthropic evals data ingestion
└── data/               # Anthropic model-written sycophancy dataset
```

### 1. Data Curation (`ingest_data.py`)
Fetches and cleans Anthropic’s `model-written-evals` sycophancy dataset (~10,000 synthetic questions spanning NLP opinions, philosophy, and factual claims designed to test if an LLM flips its stance to match a user's stated bias).

### 2. Offline Model Benchmarking (`evaluator.py`)
An automated local testing pipeline that feeds trap questions to open-weights models and measures their baseline sycophancy score against ground-truth honest answers.

### 3. Real-Time Guardrail (`api/score.js` + `index.html`)
A production-ready serverless scoring endpoint:
- **Model**: `openai/gpt-oss-20b` (served via Groq's high-speed LPU inference)
- **Prompt Engineering**: Structured zero-shot alignment judge that evaluates conversational dynamics, flattery, and uncritical agreement
- **Security**: Complete separation between public client and API credentials stored in encrypted environment variables

---

