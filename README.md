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

## API

```python
from spelling_correction import NoisyChannelSpellCorrector

checker = NoisyChannelSpellCorrector()
corrected_text, corrections = checker.correct("I recieve a mesage.")
```
