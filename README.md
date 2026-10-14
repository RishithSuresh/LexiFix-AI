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

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py
```

This creates `spelling_correction_results.csv`. The baseline columns are
always populated; LLM columns are populated only when `--llm-model` is
provided:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" .\run_comparison.py `
  --llm-model "google/flan-t5-small"
```

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
├── my_cases.csv                 # Example labeled input
├── spelling_correction_results.csv
├── tests/test_spelling_correction.py
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

The test file contains six focused checks and can be run with pytest if it is
installed:

```powershell
& "C:\Users\rishi\anaconda3\envs\nlp\python.exe" -m pytest .\tests
```

## API

```python
from spelling_correction import NoisyChannelSpellCorrector

checker = NoisyChannelSpellCorrector()
corrected_text, corrections = checker.correct("I recieve a mesage.")
```
