"""Run baseline-vs-LLM spelling correction and write CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from spelling_correction import LLMCorrector, NoisyChannelSpellCorrector, word_accuracy


DEFAULT_CASES = [
    ("I recieve the mesage from my frend.", "I receive the message from my friend."),
    ("This is a beautifull and usefull example.", "This is a beautiful and useful example."),
    ("The quick brown fox jumpd over the lazy dog.", "The quick brown fox jumped over the lazy dog."),
]


def run(cases: list[tuple[str, str]], output: Path, llm_model: str | None) -> None:
    baseline = NoisyChannelSpellCorrector()
    llm = LLMCorrector(llm_model) if llm_model else None
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_text", "expected_text", "detected_errors", "baseline_text",
        "baseline_accuracy", "llm_text", "llm_accuracy", "llm_status",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for input_text, expected_text in cases:
            baseline_text, corrections = baseline.correct(input_text)
            row = {
                "input_text": input_text,
                "expected_text": expected_text,
                "detected_errors": "; ".join(item.original for item in corrections if item.distance != 0),
                "baseline_text": baseline_text,
                "baseline_accuracy": f"{word_accuracy(expected_text, baseline_text):.3f}",
                "llm_text": "",
                "llm_accuracy": "",
                "llm_status": "not configured",
            }
            if llm is not None:
                try:
                    llm_text = llm.correct(input_text)
                    row["llm_text"] = llm_text
                    row["llm_accuracy"] = f"{word_accuracy(expected_text, llm_text):.3f}"
                    row["llm_status"] = "ok"
                except (RuntimeError, OSError, ValueError) as exc:
                    row["llm_status"] = f"error: {exc}"
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("spelling_correction_results.csv"))
    parser.add_argument("--llm-model", help="Hugging Face text-to-text model name or local path.")
    args = parser.parse_args()
    run(DEFAULT_CASES, args.output, args.llm_model)
    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()
