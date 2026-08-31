# LLM Sycophancy Detector

A minimalistic web tool that detects sycophantic behavior in AI conversations.

Paste in a conversation between a user and an AI, and get an instant sycophancy score from 1 (honest) to 5 (extremely sycophantic).

## Live Demo

Hosted on Vercel — [Try it here](https://llm-sycophancy-detector.vercel.app)

## Architecture

- **Frontend:** Static HTML/CSS/JS — clean, dark-mode, single-page interface
- **Backend:** Vercel Serverless Function (`api/score.js`) — securely proxies requests to a free LLM inference API (Groq Cloud) with the API key hidden in environment variables
- **Model:** Llama 3.1 8B Instruct (via Groq) — prompted to score sycophancy on a 1–5 scale

## Project Structure

```
├── index.html          # Frontend UI
├── api/
│   └── score.js        # Vercel serverless function (backend)
├── rate_text.py         # CLI tool for local sycophancy rating
├── evaluator.py         # Batch evaluation on Anthropic sycophancy dataset
├── ingest_data.py       # Dataset ingestion script
└── data/                # Anthropic model-written-evals sycophancy data
```

## Local Development

```bash
# Install Python dependencies
pip install transformers torch datasets pandas tqdm

# Run the CLI sycophancy rater
python rate_text.py

# Run the batch evaluator on the Anthropic dataset
python evaluator.py
```

## Deployment

1. Fork/clone this repo
2. Import into [Vercel](https://vercel.com)
3. Add `GROQ_API_KEY` as an environment variable (get a free key at [console.groq.com](https://console.groq.com/keys))
4. Deploy
