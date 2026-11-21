# LexiFix AI

## What this project does

LexiFix AI is an explainable spelling-correction experiment. It detects
possible spelling errors, proposes corrections, and compares a deterministic
Python NLP method with an optional Transformer-based LLM.

The project is intentionally split into two layers:

- `spelling_correction.py` contains reusable correction and evaluation
  components.
- `run_comparison.py` is the command-line evaluation application.

The code comments explain the important decisions inside each component.

This project compares two spelling-correction approaches:

1. **Python NLP baseline:** a lightweight noisy-channel corrector detects
   out-of-vocabulary words and ranks edit-distance candidates by corpus
   frequency. It is deterministic and works offline.
2. **LLM-based correction:** an optional Hugging Face text-to-text model is
   prompted to correct spelling while preserving meaning and punctuation.

## Run

### Modern Desktop & Studio Interface

The easiest way to use LexiFix AI is the modern desktop application:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\app.py
```

LexiFix AI launches in a native, high-DPI desktop window powered by WebView2 (with instant fallback to your default browser via `--browser` or if headless).

**Core Studio Features:**
- **Live Studio (01):** Paste text or pick quick test samples. Compare the deterministic Python NLP baseline side-by-side with Transformer LLM models (`vennify/t5-base-grammar-correction`, `vennify/t5-small`, `google/flan-t5-small`, or custom Hugging Face checkpoints).
- **Interactive Visual Diff:** Real-time word-level diffing highlighting deleted misspellings in red strikethrough and corrected tokens in emerald green.
- **Explainability Inspector:** Inspect candidate ranking, Levenshtein edit distance ($d$), and corpus frequency ($f$) for each token.
- **Live Accuracy & Metrics:** Enter optional ground truth text to immediately calculate Word Accuracy %, Token Precision, Recall, and F1 score.
- **CSV Benchmarks Explorer (02):** Browse, filter, sort, and inspect comparison CSV files (`multi_model_results.csv`, `my_results.csv`, etc.) with KPI summary cards and detailed row inspection modals.
- **Batch Evaluator (03):** Run batch benchmark evaluations directly from the UI with animated progress tracking.
- **NLP Sandbox (04):** Test single words, view candidates, and explore the internal corpus frequency distribution.
- **Settings & Theme (05):** Toggle between dark glassmorphism and crisp light themes, and inspect PyTorch/device status.

Launch flags:
```powershell
python app.py            # Native desktop window
python app.py --browser  # Open directly in default web browser
python app.py --port 9000 --no-open  # Headless server mode
```

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py
```

This creates `spelling_correction_results.csv`. The baseline columns are
always populated; LLM columns are populated only when `--llm-model` is
provided:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py `
  --llm-model "vennify/t5-base-grammar-correction" `
  --summary-output .\llm_comparison_summary.csv
```

Use a model trained for correction rather than a general instruction model
such as `google/flan-t5-small`; general models often copy misspellings instead
of correcting them. The recommended checkpoint uses the `grammar:` task
prefix internally.

### Compare multiple models from PowerShell

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py `
  --cases .\my_cases.csv `
  --llm-models `
    "vennify/t5-base-grammar-correction" `
    "vennify/t5-small-grammar-correction" `
    "google/flan-t5-small" `
  --output .\multi_model_results.csv `
  --summary-output .\multi_model_summary.csv
```

The detailed CSV contains separate columns for each model, for example
`model_vennify_t5_base_grammar_correction_f1`. The summary CSV contains the mean baseline metrics plus one row per selected
model. Models may require a one-time download from
Hugging Face; a local model directory can be supplied instead.

For a larger labeled dataset, create a CSV with `input_text` and
`expected_text` columns and pass it to the runner:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py `
  --cases .\my_cases.csv `
  --output .\my_results.csv `
  --max-distance 2
```

The first LLM run may download model files. Use a local model path for
offline execution. `baseline_accuracy` and `llm_accuracy` are positional
word accuracies against the expected text. The CSV also reports token-level
correction precision, recall, and F1. These metrics measure whether the system
changed exactly the tokens that differ between the input and expected text;
they are more informative than accuracy when false corrections matter.
The optional summary CSV contains the mean metrics for each system.

## Understand the baseline

The baseline does not blindly replace every unknown word. For each unknown
token it:

1. Finds vocabulary words within `--max-distance` edits.
2. Prefers the smallest edit distance.
3. Uses word frequency to break ties.
4. Preserves capitalization and all non-word text.

If no candidate is close enough, the original word is retained and its
confidence is `0.0`. The confidence value is a ranking heuristic, not a
calibrated probability.

## Project structure

```text
LexiFix AI/
├── spelling_correction.py       # NLP and LLM correction components
├── run_comparison.py            # CLI runner and CSV evaluation
├── app.py                       # Modern desktop & studio application server
├── web/                         # Modern frontend application (HTML/CSS/JS)
│   ├── index.html               # Single-page studio interface
│   ├── styles.css               # Design system & visual diff styles
│   └── app.js                   # Reactive UI logic & analytics
├── my_cases.csv                 # Example labeled input
├── spelling_correction_results.csv
├── tests/test_spelling_correction.py
├── tests/test_run_comparison.py
├── tests/test_app.py
├── requirements.txt
└── README.md
```

## Recommended workflow

1. Start with the offline baseline and inspect its CSV.
2. Add representative examples to `my_cases.csv`.
3. Compare precision, recall, and F1, not only accuracy.
4. Run an LLM only when a model is configured and downloaded.
5. Review low-confidence or unchanged words before deploying corrections
   automatically.

## Run checks

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" -m compileall -q .
```

The focused regression tests can be run with pytest if it is installed:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" -m pytest .\tests
```

## API

```python
from spelling_correction import NoisyChannelSpellCorrector

checker = NoisyChannelSpellCorrector()
corrected_text, corrections = checker.correct("I recieve a mesage.")
```
