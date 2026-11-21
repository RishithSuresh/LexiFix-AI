# LexiFix AI — Spelling Intelligence Studio

LexiFix AI is an advanced, explainable spelling-correction and grammatical evaluation studio. It detects spelling errors, proposes context-aware corrections, and benchmarks an enhanced multi-signal Python NLP engine against state-of-the-art Transformer language models.

---

## 🎨 Proportional Design System
The studio is built around an editorial, tactile color hierarchy:
1. **Pebble Silk (`#D9D3CC`) — 60% of the project**: The dominant, warm silk canvas, main surfaces, workbench cards, and data views.
2. **Eclipse Plum (`#2B1C33`) — 20% of the project**: Structural anchoring, deep authoritative brand header, crisp high-contrast typography, and dark terminal panels.
3. **Amber Flare (`#FF8C12`) — 20% of the project**: Radiant high-energy CTA buttons, visual diff insertions, active indicators, and metric highlights.

---

## 🚀 Enhanced Multi-Signal NLP Architecture

LexiFix AI includes significant architectural improvements over simple unigram spellcheckers:

1. **Context-Aware Bigram Language Model:**
   - Evaluates consecutive word transitions $P(w_i \mid w_{i-1})$ and $P(w_{i+1} \mid w_i)$ to disambiguate candidates using sentence context.
2. **American Soundex Phonetic Matching:**
   - Detects sound-alike errors (e.g. `frend` $\rightarrow$ `friend`, `recieve` $\rightarrow$ `receive`, `definately` $\rightarrow$ `definitely`) and provides a calibrated phonetic confidence bonus.
3. **Damerau-Levenshtein Transposition Awareness:**
   - Recognizes adjacent letter swaps (like `ie` $\leftrightarrow$ `ei` in `recieve` vs `receive`) as a single transposition edit ($d=1$).
4. **Physical QWERTY Keyboard Adjacency:**
   - Calculates Euclidean distance between keys on a standard keyboard layout to prioritize physical slip-of-the-finger typos.
5. **Comprehensive English Vocabulary:**
   - Expanded zero-shot vocabulary of ~3,000+ words covering common misspellings, irregular verbs, contractions, and everyday terms.
6. **Advanced Evaluation Metrics:**
   - Positional **Word Accuracy**, Token-level **Precision / Recall / F1**, **Character Error Rate (CER)**, **Word Error Rate (WER)**, and **Exact Match (EM)**.

---

## 💻 Running the Application

### 1. Modern Desktop & Studio Interface

Launch the studio in native high-DPI desktop mode (WebView2) or web browser:

```powershell
# In Conda environment
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\app.py

# Or in standard Python environment
python app.py
```

**Launch Options:**
- `python app.py` — Launches native desktop window.
- `python app.py --browser` — Opens directly in default web browser.
- `python app.py --port 9000 --no-open` — Headless server mode.

### 2. Studio Modules
- **Live Studio (01):** Interactive text input, real-time token diffing, side-by-side model comparison (`vennify/t5-base-grammar-correction`, `oliverguhr/spelling-correction-english-base`, `google/flan-t5-small`, custom models).
- **CSV Benchmarks Explorer (02):** Browse, search, filter, and sort benchmark results with KPI cards and row inspection modals.
- **Batch Evaluator (03):** Execute batch evaluation over datasets (`my_cases.csv`) with animated progress tracking.
- **NLP Sandbox (04):** Inspect single-word diagnostics, Soundex codes, Damerau transpositions, and lexicon frequencies.
- **Environment & Settings (05):** Hardware acceleration status (PyTorch, CUDA GPU detection), Transformers runtime status, and theme preferences.

---

## 🔬 CLI Benchmark Evaluation

```powershell
# Basic evaluation
python run_comparison.py

# Multi-model comparative benchmark with summary report
python run_comparison.py `
  --cases .\my_cases.csv `
  --llm-models `
    "vennify/t5-base-grammar-correction" `
    "oliverguhr/spelling-correction-english-base" `
    "google/flan-t5-small" `
  --output .\multi_model_results.csv `
  --summary-output .\multi_model_summary.csv
```

---

## 🧪 Testing

Run unit and integration tests across both environments:

```powershell
python -m pytest
```
