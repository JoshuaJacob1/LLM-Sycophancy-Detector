# LLM Sycophancy Detector & Representation Engineering

A dual-tier AI alignment project: a white-box **Representation Engineering (RepE)** research suite for extracting and steering sycophancy vectors in transformer activations, paired with a production-grade **real-time serverless guardrail**.

---

## Live Demo

Hosted on Vercel — **[Try the Real-Time Detector](https://llm-sycophancy-detector.vercel.app)**

---

## System Architecture

```
├── repeng/                     # Representation Engineering package (White-box)
│   ├── data_prep.py            # Builds contrastive pairs (Sycophantic vs. Honest)
│   ├── extract.py              # Hook-based residual stream activation extraction
│   ├── layer_sweep.py          # Diff-in-means vector extraction & AUROC layer sweeps
│   ├── linear_probe.py         # Logistic regression probing & direction comparisons
│   ├── steer.py                # Inference-time activation steering (intervention)
│   └── baselines.py            # Log-prob model sycophancy rate & random baselines
│
├── run_repeng.py               # Single entry point for representation engineering pipeline
├── evaluator.py                # Black-box offline evaluation on Anthropic evals
├── ingest_data.py              # Ingests Anthropic model-written evaluation dataset
├── rate_text.py                # Interactive CLI guardrail rater
│
├── api/
│   └── score.js                # Production serverless proxy (Groq LPU / GPT-OSS 20B)
├── index.html                  # Minimalist web guardrail UI
├── why.html                    # Deep-dive on RLHF incentives & South Park satire
└── requirements.txt            # PyTorch, Transformers, Scikit-learn dependencies
```

---

## 1. Representation Engineering Suite (`repeng/`)

Based on the Representation Engineering framework ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)), this module inspects the internal hidden states of open-weights models (`Qwen/Qwen2.5-1.5B-Instruct`) to detect and steer sycophancy at the layer activation level.

### Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest Anthropic dataset
python ingest_data.py

# 3. Run full RepEng pipeline (Layer sweep, probe, steering demo)
python run_repeng.py --model Qwen/Qwen2.5-1.5B-Instruct
```

### Key Technical Methods:
1. **Diff-in-Means Direction Extraction**:
   $$\vec{v}_l = \frac{\boldsymbol{\mu}_{\text{syc}}^{(l)} - \boldsymbol{\mu}_{\text{honest}}^{(l)}}{\|\boldsymbol{\mu}_{\text{syc}}^{(l)} - \boldsymbol{\mu}_{\text{honest}}^{(l)}\|_2}$$
   Extracts layer-wise centroids and calculates normalized concept vectors separating sycophantic from honest completions.
2. **Layer Sweeping**: Measures AUROC across all 28 transformer layers to pinpoint where the sycophancy representation peaks in the residual stream.
3. **Linear Probing**: Fits $L_2$-regularized logistic classifiers on activation tensors and validates alignment with diff-in-means vectors via cosine similarity.
4. **Activation Steering (Inference Intervention)**:
   $$h_l \leftarrow h_l - \alpha \cdot \vec{v}_l$$
   Dynamically modifies hidden states via forward hooks during autoregressive generation to suppress sycophancy without fine-tuning.

---

## 2. Production Serverless Guardrail (`api/` + `index.html`)

For zero-cost real-time inference on arbitrary user-provided conversations:
- **Inference Engine**: `openai/gpt-oss-20b` via Groq LPU hardware acceleration (sub-300ms latency).
- **Backend Architecture**: Stateless Vercel Serverless Functions using native Node.js `https.request` streams for deterministic TLS handshakes.
- **Resilient Multi-Stage Parsing**: Robust JSON AST parsing with fallback regex token matching to handle internal chain-of-thought outputs.

---

## Deployment

1. Fork or clone this repository.
2. Import the project into [Vercel](https://vercel.com).
3. Set `GROQ_API_KEY` in **Environment Variables** (free tier key from [console.groq.com](https://console.groq.com/keys)).
4. Deploy!
