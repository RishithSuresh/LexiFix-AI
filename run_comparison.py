"""Run the offline baseline, optionally compare an LLM, and write CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from spelling_correction import (
    LLMCorrector,
    NoisyChannelSpellCorrector,
    correction_metrics,
    word_accuracy,
)


DEFAULT_CASES = [
    ("I recieve the mesage from my frend.", "I receive the message from my friend."),
    ("This is a beautifull and usefull example.", "This is a beautiful and useful example."),
    ("The quick brown fox jumpd over the lazy dog.", "The quick brown fox jumped over the lazy dog."),
]


def load_cases(path: Path) -> list[tuple[str, str]]:
    """Load labeled examples and provide a useful error for malformed files."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Cases file not found: {path}. Create a CSV with "
            "input_text and expected_text columns, or omit --cases."
        )
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"input_text", "expected_text"} <= set(rows[0]):
        raise ValueError("Cases CSV must contain input_text and expected_text columns.")
    return [(row["input_text"], row["expected_text"]) for row in rows]


def run(
    cases: list[tuple[str, str]],
    output: Path,
    llm_model: str | None,
    max_distance: int = 2,
) -> None:
    """Evaluate every case and write one auditable row per example."""
    baseline = NoisyChannelSpellCorrector(max_distance=max_distance)
    llm = LLMCorrector(llm_model) if llm_model else None
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_text", "expected_text", "detected_errors", "baseline_text",
        "baseline_accuracy", "baseline_precision", "baseline_recall", "baseline_f1",
        "llm_text", "llm_accuracy", "llm_precision", "llm_recall", "llm_f1", "llm_status",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for input_text, expected_text in cases:
            # Keep the raw input and expected answer in the output so each
            # metric can be independently checked later.
            baseline_text, corrections = baseline.correct(input_text)
            row = {
                "input_text": input_text,
                "expected_text": expected_text,
                "detected_errors": "; ".join(
                    item.original for item in corrections if item.distance > 0
                ),
                "baseline_text": baseline_text,
                "baseline_accuracy": f"{word_accuracy(expected_text, baseline_text):.3f}",
                **{
                    f"baseline_{key}": f"{value:.3f}"
                    for key, value in correction_metrics(
                        input_text, expected_text, baseline_text
                    ).items()
                },
                "llm_text": "",
                "llm_accuracy": "",
                "llm_precision": "",
                "llm_recall": "",
                "llm_f1": "",
                "llm_status": "not configured",
            }
            if llm is not None:
                try:
                    llm_text = llm.correct(input_text)
                    row["llm_text"] = llm_text
                    row["llm_accuracy"] = f"{word_accuracy(expected_text, llm_text):.3f}"
                    row.update({
                        f"llm_{key}": f"{value:.3f}"
                        for key, value in correction_metrics(
                            input_text, expected_text, llm_text
                        ).items()
                    })
                    row["llm_status"] = "ok"
                except (RuntimeError, OSError, ValueError) as exc:
                    row["llm_status"] = f"error: {exc}"
            writer.writerow(row)


def main() -> None:
    """Parse command-line options and run the selected evaluation set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("spelling_correction_results.csv"))
    parser.add_argument("--cases", type=Path, help="CSV with input_text and expected_text columns.")
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--llm-model", help="Hugging Face text-to-text model name or local path.")
    args = parser.parse_args()
    cases = load_cases(args.cases) if args.cases else DEFAULT_CASES
    run(cases, args.output, args.llm_model, args.max_distance)
    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()
