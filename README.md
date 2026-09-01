# LLM Sycophancy Detector

LLMs often tell users what they want to hear instead of what's true. This repo looks at why that happens, how to find the sycophancy vector inside a model's hidden layers, and how to score conversations in real time.

Live demo: [llm-sycophancy-detector.vercel.app](https://llm-sycophancy-detector.vercel.app)

---

## What's in here

There are two parts to this project:

1. **Internal model probing (`repeng/`)**: Uses PyTorch hooks on `Qwen2.5-1.5B` to pull out hidden states, find the direction in activation space that represents sycophancy, test which layer separates honest vs. sycophantic answers best, and steer the model during generation by subtracting that vector.
2. **Web app (`api/` + `index.html`)**: A small serverless rater on Vercel backed by Groq (`GPT-OSS 20B`) that scores pasted conversations 1 to 5.

---

## Repo layout

```
├── repeng/                 # Activation extraction and steering code
│   ├── data_prep.py        # Pairs up Anthropic survey questions (honest vs. sycophantic)
│   ├── extract.py          # PyTorch hooks to pull out layer activations
│   ├── layer_sweep.py      # Finds the best layer using diff-in-means and AUROC
│   ├── linear_probe.py     # Logistic regression probe on top of activations
│   ├── steer.py            # Subtracts the sycophancy vector during generation
│   └── baselines.py        # Logprob baseline and random chance baseline
│
├── run_repeng.py           # Runs the full local pipeline end-to-end
├── evaluator.py            # Local evaluator on the raw Anthropic CSV
├── ingest_data.py          # Pulls the Anthropic eval data
├── rate_text.py            # Quick CLI rater
│
├── api/
│   └── score.js            # Vercel serverless handler (calls Groq)
├── index.html              # Web UI
├── why.html                # Explainer page on RLHF incentives and South Park
└── requirements.txt        # Python dependencies
```

---

## Running the local probing pipeline

Runs locally on CPU or Apple Silicon in about 5 minutes:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Grab the Anthropic eval data
python ingest_data.py

# 3. Run the full sweep, probe, and steering demo
python run_repeng.py --model Qwen/Qwen2.5-1.5B-Instruct
```

### What the script does

- **Diff-in-means**: Computes average activation for sycophantic answers, subtracts average activation for honest answers, and normalizes the difference to get a unit direction vector.
- **Layer sweep**: Tests all 28 layers of the model to see which layer's activations separate sycophantic vs. honest answers best (measured by AUROC).
- **Linear probe**: Fits a logistic regression model on the best layer's activations to verify they're linearly separable.
- **Activation steering**: Hooks into the target layer during text generation and subtracts the sycophancy vector ($h - \alpha v$) to see if the model stops agreeing with bad premises.

---

## Web tool setup

- **Backend**: Node.js function on Vercel sending requests to Groq.
- **Model**: `openai/gpt-oss-20b` with a scoring prompt and fallback regex parsing.
- **Key security**: `GROQ_API_KEY` stays in Vercel environment variables.

### Deploying to Vercel

1. Fork or clone this repo.
2. Import it into [Vercel](https://vercel.com).
3. Add `GROQ_API_KEY` under **Project Settings > Environment Variables** (from [console.groq.com](https://console.groq.com/keys)).
4. Deploy.
