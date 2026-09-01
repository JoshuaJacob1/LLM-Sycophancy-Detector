# LLM Sycophancy Detector

A project exploring why AI models tell users what they want to hear instead of the truth, how to detect it inside transformer activations, and how to score conversations in real time.

Live demo: **[llm-sycophancy-detector.vercel.app](https://llm-sycophancy-detector.vercel.app)**

---

## Overview

Sycophancy happens when an AI agrees with a user's false claim, flatters them, or flips its stance just to be agreeable. This project attacks the problem from two angles:

1. **White-box experiments (`repeng/`)**: Probing intermediate transformer layers in open-weights models (`Qwen2.5-1.5B`) to isolate the internal vector representing sycophancy, run layer sweeps, and test activation steering during generation.
2. **Real-time web guardrail (`api/` + `index.html`)**: A lightweight serverless tool on Vercel backed by Groq LPUs (`GPT-OSS 20B`) that scores arbitrary user conversations on a 1–5 scale with sub-second latency.

---

## Project Structure

```
├── repeng/                 # Activation probing & steering package
│   ├── data_prep.py        # Formats contrastive pairs from Anthropic evals
│   ├── extract.py          # PyTorch hooks for extracting hidden states
│   ├── layer_sweep.py      # Computes diff-in-means vectors and per-layer AUROC
│   ├── linear_probe.py     # Logistic regression probes on activations
│   ├── steer.py            # Activation steering demo (removes sycophancy vector)
│   └── baselines.py        # Log-probability choice benchmark & random controls
│
├── run_repeng.py           # Runs the full representation engineering pipeline
├── evaluator.py            # Offline benchmark script on Anthropic evaluation set
├── ingest_data.py          # Downloads Anthropic model-written evaluation data
├── rate_text.py            # Interactive CLI scoring tool
│
├── api/
│   └── score.js            # Vercel serverless function (Groq / GPT-OSS 20B)
├── index.html              # Frontend app
├── why.html                # Plain-English explainer on RLHF incentives & South Park satire
└── requirements.txt        # Dependencies (PyTorch, Transformers, Scikit-learn)
```

---

## Running the White-Box Pipeline

The representation engineering scripts run locally on standard hardware (takes ~5 minutes on an M-series Mac or standard CPU):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the Anthropic evaluation dataset
python ingest_data.py

# 3. Run the full pipeline (layer sweep, linear probe, steering demo)
python run_repeng.py --model Qwen/Qwen2.5-1.5B-Instruct
```

### How the White-Box Method Works

- **Diff-in-means direction**: Takes the mean activation vector of sycophantic completions and subtracts the mean activation vector of honest completions to isolate the sycophancy direction in activation space.
- **Layer sweep**: Sweeps all 28 layers of the model to find which intermediate layer separates sycophantic from honest responses most clearly (measured by AUROC).
- **Linear probe**: Trains a simple logistic regression classifier on top of hidden states at the best layer to confirm linear separability.
- **Activation steering**: Subtracts the sycophancy vector from the model's hidden states during generation via a forward hook, testing whether the model pushes back against false premises without fine-tuning.

---

## Real-Time Web Tool

- **Backend**: Node.js serverless function on Vercel using native HTTPS streams to forward prompts to Groq's low-latency LPUs.
- **Judge Model**: `openai/gpt-oss-20b` prompted with a structured alignment rubric and multi-stage output parsing.
- **Security**: The API key lives in Vercel environment variables and is never exposed to the client.

---

## Deploying Your Own Instance

1. Fork or clone this repo.
2. Import the repo into [Vercel](https://vercel.com).
3. Add your `GROQ_API_KEY` under **Project Settings > Environment Variables** (free from [console.groq.com](https://console.groq.com/keys)).
4. Hit deploy.
