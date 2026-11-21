# ✦ LexiFix AI — Spelling Intelligence Studio

<div align="center">

**A Next-Generation Explainable Spelling Correction & Grammar Evaluation Platform**  
*Benchmarking Multi-Signal Deterministic NLP against State-of-the-Art Deep Learning Transformers*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passing-brightgreen?style=flat)](#-testing--validation)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## 📖 Table of Contents
* [Overview](#-overview)
* [Key Features](#-key-features)
* [Architectural Highlights](#-architectural-highlights)
* [Design System (60-20-20 Rule)](#-design-system-60-20-20-rule)
* [System Requirements](#-system-requirements)
* [Quickstart Guide](#-quickstart-guide)
* [Interactive Studio Modules](#-interactive-studio-modules)
* [CLI Batch Benchmarking](#-cli-batch-benchmarking)
* [REST API Catalog](#-rest-api-catalog)
* [Project Structure](#-project-structure)
* [Testing & Validation](#-testing--validation)
* [Detailed Technical Documentation](#-detailed-technical-documentation)

---

## 🔭 Overview

**LexiFix AI** is an advanced spelling intelligence studio designed to evaluate, explain, and benchmark text correction systems. It pairs a **sub-millisecond, multi-signal deterministic Python NLP baseline** with **deep learning Seq2Seq Transformer models** (`vennify/t5-base-grammar-correction`, `Unbabel/gec-t5_small`, `google/flan-t5-small`, `oliverguhr/spelling-correction-english-base`).

Unlike black-box spellcheckers, LexiFix AI exposes the internal mechanics behind every correction:
* Candidate rankings and edit distances (Levenshtein & Damerau-Levenshtein)
* American Soundex phonetic encoding
* QWERTY physical keyboard layout proximity
* Contextual bigram transition probabilities
* Precision, Recall, F1, Character Error Rate (CER), and Word Error Rate (WER)

---

## ⚡ Key Features

* **🏎️ Sub-Millisecond NLP Baseline:** Completely offline, deterministic spell checker operating at $<2\text{ms}$ latency powered by Peter Norvig's 1.1M-token corpus (`data/big.txt`).
* **🧠 Deep Learning Transformer Suite:** Integrated Hugging Face Seq2Seq bridge with automatic CUDA GPU acceleration and beam search decoding.
* **🔍 Explainable Diagnostics:** Live single-word inspection showing candidate probabilities, Soundex phonetic hashes, and transposition penalties.
* **🎓 Live Dynamic Training:** Train the baseline model in real-time with custom text or vocabulary files without restarting the server.
* **📊 Visual Token Diffing:** Instant side-by-side comparison between original text, corrected outputs, and expected ground truth.
* **📈 Batch Evaluator & CSV Explorer:** Run automated multi-model benchmarks on custom datasets (e.g., `my_cases.csv`) with progress tracking and CSV inspection.
* **🖥️ Dual Deployment:** Runs as a native desktop application (via Microsoft Edge WebView2 / `pywebview`) or as a local web app in any browser.

---

## 🧬 Architectural Highlights

| Component | Technical Implementation | Purpose |
| :--- | :--- | :--- |
| **Language Model Prior** | Peter Norvig 1.1M-Token Corpus (`data/big.txt`, 29,161 unique words) | Prior probability $P(\text{candidate})$ |
| **Phonetic Hashing** | American Soundex Algorithm | Captures sound-alike typos (`frend` $\rightarrow$ `friend`) |
| **Transposition Model** | Damerau-Levenshtein Matrix | Treats letter swaps (`recieve` $\rightarrow$ `receive`) as 1 edit |
| **Typo Geometry** | 2D QWERTY Coordinate Map | Rewards substitutions of physically adjacent keys |
| **Context Model** | Bigram Transition Matrix | Evaluates $P(w_i \mid w_{i-1})$ and $P(w_{i+1} \mid w_i)$ |
| **Transformer Bridge** | PyTorch + Hugging Face Transformers | High-accuracy GEC and Seq2Seq generation |
| **Decoding** | Deterministic Beam Search (`num_beams=2`) | Avoids greedy search errors and hallucinations |

---

## 🎨 Design System (60-20-20 Rule)

The user interface follows a modern, proportional color scheme:

```
┌────────────────────────────────────────────────────────┐
│  Pebble Silk (#D9D3CC) — 60% Dominant Canvas           │
│  Main surfaces, workbench cards, and dataset tables    │
├────────────────────────────┬───────────────────────────┤
│  Eclipse Plum (#2B1C33)    │  Amber Flare (#FF8C12)    │
│  20% Structural Framing    │  20% Radiant Accent       │
│  Header, typography, logs  │  CTAs, diffs, metrics     │
└────────────────────────────┴───────────────────────────┘
```

* **Pebble Silk (`#D9D3CC`) — 60%:** The tactile, natural background canvas, card surfaces, and input containers.
* **Eclipse Plum (`#2B1C33`) — 20%:** Structural contrast, authoritative brand header, dark terminal log box, and crisp typography.
* **Amber Flare (`#FF8C12`) — 20%:** High-energy action buttons, visual diff insertions, active tab indicators, and benchmark highlights.

---

## 💻 System Requirements

* **Operating System:** Windows 10/11, macOS, or Linux
* **Python Version:** Python 3.10 or 3.11
* **GPU (Optional):** NVIDIA GPU with CUDA for accelerated Transformer inference (CPU fallback supported automatically)
* **Web Engine:** Microsoft Edge WebView2 (installed by default on Windows 10/11)

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/your-username/LexiFix-AI.git
cd "LexiFix AI"

# Option A: In Conda environment
conda create -n nlp python=3.11 -y
conda activate nlp
pip install -r requirements.txt

# Option B: Standard Python venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Application

```powershell
# Launch native desktop application (WebView2)
python app.py

# Launch in default web browser
python app.py --browser

# Launch headless server on specific port
python app.py --port 8765 --no-open
```

Once started, the application is accessible at `http://127.0.0.1:8765/`.

---

## 🎛️ Interactive Studio Modules

The interface is organized into 5 dedicated views:

1. **View 01: Live Studio & Visual Diff**
   - Interactive multi-token spelling correction.
   - Live side-by-side comparison between Deterministic Baseline and Transformer models.
   - Interactive token diff badges showing deletions and replacements.
   - Ground truth evaluation with real-time Accuracy, Precision, Recall, and F1 calculations.

2. **View 02: Benchmark CSV Explorer**
   - Live browsing and filtering of benchmark evaluation datasets.
   - Instant search, column sorting, pagination, and row-level modal inspection.
   - Live summary KPI cards showing Macro Accuracy, F1 Score, and Total Processed Cases.

3. **View 03: Batch Evaluator**
   - Select dataset files (e.g., `my_cases.csv`) and run comparative multi-model evaluations.
   - Real-time animated progress bar and detailed execution logs.
   - Automatically saves and reloads generated benchmark results.

4. **View 04: NLP Sandbox & Live Trainer**
   - **Explainability Inspector:** Enter any word (e.g., `recieve`) to view its vocabulary status, Soundex code, Damerau distance, and full ranked candidate list.
   - **Live Model Trainer:** Ingest custom domain text or external vocabulary files dynamically to expand the model's vocabulary without rebooting.

5. **View 05: Environment & System Diagnostics**
   - Displays hardware acceleration status, PyTorch version, active CUDA device, and memory usage.

---

## 🔬 CLI Batch Benchmarking

You can run automated benchmarks from the terminal using `run_comparison.py`:

```powershell
# Basic evaluation on default test cases
python run_comparison.py

# Comprehensive comparative benchmark on custom dataset
python run_comparison.py `
  --cases .\my_cases.csv `
  --llm-models `
    "vennify/t5-base-grammar-correction" `
    "Unbabel/gec-t5_small" `
    "google/flan-t5-small" `
    "oliverguhr/spelling-correction-english-base" `
  --max-distance 2 `
  --output .\multi_model_results.csv `
  --summary-output .\multi_model_summary.csv
```

---

## 🌐 REST API Catalog

The backend provides a clean JSON REST API:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/info` | Hardware acceleration, CUDA status, and Python runtime info |
| `GET` | `/api/vocabulary` | Active vocabulary size, total token counts, and top words |
| `GET` | `/api/csv/list` | Returns list of benchmark CSV files in workspace |
| `GET` | `/api/csv/read?file=...` | Reads CSV headers and rows for data tables |
| `POST` | `/api/correct` | Corrects text across baseline & LLMs with evaluation metrics |
| `POST` | `/api/explain/word` | Returns candidate ranking breakdown, Soundex hash, and distances |
| `POST` | `/api/model/train` | Ingests new text into the active model vocabulary |
| `POST` | `/api/benchmark/run` | Executes batch evaluation over dataset file |
| `POST` | `/api/csv/save` | Updates and saves modified dataset records |

---

## 📁 Project Structure

```
LexiFix AI/
├── app.py                      # Multi-threaded HTTP server & desktop window entrypoint
├── spelling_correction.py      # Core NLP engine: NoisyChannel, Soundex, Damerau, LLMCorrector
├── run_comparison.py           # CLI benchmark runner and dataset evaluator
├── HOW_IT_WORKS.md             # In-depth technical architecture and algorithmic guide
├── README.md                   # Project overview, installation, and usage guide
├── requirements.txt            # Python dependencies
├── my_cases.csv                # Sample labeled benchmark dataset
├── data/
│   └── big.txt                 # Peter Norvig 1.1M-token English corpus (29,161 words)
├── web/
│   ├── index.html              # Studio interface markup (Pebble Silk / Plum / Amber)
│   ├── styles.css              # Custom styling, animations, and color system variables
│   └── app.js                  # Frontend state machine, visual diffing, and API client
└── tests/
    ├── test_app.py             # Server endpoints & REST API automated tests
    ├── test_spelling_correction.py  # NLP engine, Soundex, Damerau, and metrics unit tests
    └── test_run_comparison.py  # Benchmark runner and CSV parsing tests
```

---

## 🧪 Testing & Validation

LexiFix AI includes an automated test suite covering all modules:

```powershell
# Run the complete test suite
python -m pytest tests -v
```

**Test Coverage Highlights:**
* **24 Automated Test Cases:** All passing across Python 3.10 and Python 3.11.
* Evaluates token detection, case preservation, Soundex matching, Damerau transposition distance, Character Error Rate (CER), Word Error Rate (WER), REST endpoints, and online model training.

---

## 📚 Detailed Technical Documentation

For an exhaustive mathematical breakdown, algorithmic formulations, and component diagrams, please refer to:
* **[HOW_IT_WORKS.md](file:///d:/Programming/LexiFix%20AI/HOW_IT_WORKS.md)** — Architectural deep-dive, Bayesian equations, and system flowcharts.

---

<div align="center">
  <b>LexiFix AI — Precision Spelling Intelligence Studio</b><br>
  Built with Python, PyTorch, Transformers, and TailwindCSS.
</div>
