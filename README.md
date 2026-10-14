# LexiFix AI

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

The first LLM run may download model files. Use a local model path for
offline execution. `baseline_accuracy` and `llm_accuracy` are positional
word accuracies against the expected text; for a real evaluation, replace the
sample cases with a labeled benchmark and report correction precision,
recall, and F1 for misspelled tokens.

## API

```python
from spelling_correction import NoisyChannelSpellCorrector

checker = NoisyChannelSpellCorrector()
corrected_text, corrections = checker.correct("I recieve a mesage.")
```
