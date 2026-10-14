"""Run the offline baseline, optionally compare an LLM, and write CSV results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

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
RECOMMENDED_LLM_MODEL = "vennify/t5-base-grammar-correction"
MODEL_PRESETS = {
    "vennify/t5-base-grammar-correction": "grammar:",
    "google/flan-t5-small": "",
    "vennify/t5-small-grammar-correction": "grammar:",
}


def model_key(model_name: str) -> str:
    """Create a stable CSV-safe column prefix from a model identifier."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


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
    summary_output: Path | None = None,
    llm_models: list[str] | None = None,
) -> None:
    """Evaluate every case and write one auditable row per example."""
    baseline = NoisyChannelSpellCorrector(max_distance=max_distance)
    selected_models = llm_models or ([llm_model] if llm_model else [])
    llms = {
        model: LLMCorrector(model, MODEL_PRESETS.get(model, "grammar:"))
        for model in selected_models
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "input_text", "expected_text", "detected_errors", "baseline_text",
        "baseline_accuracy", "baseline_precision", "baseline_recall", "baseline_f1",
        "llm_text", "llm_accuracy", "llm_precision", "llm_recall", "llm_f1", "llm_status",
    ]
    for model_name in selected_models:
        prefix = f"model_{model_key(model_name)}"
        fields.extend([
            f"{prefix}_text", f"{prefix}_accuracy", f"{prefix}_precision",
            f"{prefix}_recall", f"{prefix}_f1", f"{prefix}_status",
        ])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        metric_rows: list[dict[str, float]] = []
        model_metric_rows: dict[str, list[dict[str, float]]] = {
            model: [] for model in selected_models
        }
        for input_text, expected_text in cases:
            # Keep the raw input and expected answer in the output so each
            # metric can be independently checked later.
            baseline_text, corrections = baseline.correct(input_text)
            baseline_metrics = correction_metrics(input_text, expected_text, baseline_text)
            metric_rows.append({
                "baseline_accuracy": word_accuracy(expected_text, baseline_text),
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            })
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
                    for key, value in baseline_metrics.items()
                },
                "llm_text": "",
                "llm_accuracy": "",
                "llm_precision": "",
                "llm_recall": "",
                "llm_f1": "",
                "llm_status": "not configured",
            }
            model_outputs: list[str] = []
            model_statuses: list[str] = []
            for model_name, llm in llms.items():
                prefix = f"model_{model_key(model_name)}"
                try:
                    llm_text = llm.correct(input_text)
                    model_outputs.append(f"[{model_name}] {llm_text}")
                    model_statuses.append(f"{model_name}: ok")
                    metrics = correction_metrics(input_text, expected_text, llm_text)
                    row[f"{prefix}_text"] = llm_text
                    row[f"{prefix}_accuracy"] = f"{word_accuracy(expected_text, llm_text):.3f}"
                    row.update({
                        f"{prefix}_{key}": f"{value:.3f}"
                        for key, value in metrics.items()
                    })
                    row[f"{prefix}_status"] = "ok"
                    # Preserve the original single-model CSV columns for
                    # scripts that already consume the earlier format.
                    if len(selected_models) == 1:
                        row["llm_accuracy"] = row[f"{prefix}_accuracy"]
                        row["llm_precision"] = row[f"{prefix}_precision"]
                        row["llm_recall"] = row[f"{prefix}_recall"]
                        row["llm_f1"] = row[f"{prefix}_f1"]
                    model_metrics = {
                        "llm_accuracy": word_accuracy(expected_text, llm_text),
                        **{f"llm_{key}": value for key, value in metrics.items()},
                    }
                    metric_rows.append(model_metrics)
                    model_metric_rows[model_name].append({
                        "accuracy": model_metrics["llm_accuracy"],
                        "precision": model_metrics["llm_precision"],
                        "recall": model_metrics["llm_recall"],
                        "f1": model_metrics["llm_f1"],
                    })
                except (RuntimeError, OSError, ValueError) as exc:
                    model_statuses.append(f"{model_name}: error: {exc}")
                    row[f"{prefix}_status"] = f"error: {exc}"
            if model_outputs:
                row["llm_text"] = "\n".join(model_outputs)
                row["llm_status"] = "; ".join(model_statuses)
            elif model_statuses:
                row["llm_status"] = "; ".join(model_statuses)
            writer.writerow(row)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_fields = ["system", "accuracy", "precision", "recall", "f1"]
        with summary_output.open("w", newline="", encoding="utf-8") as stream:
            summary_writer = csv.DictWriter(stream, fieldnames=summary_fields)
            summary_writer.writeheader()
            summary_groups: dict[str, list[dict[str, float]]] = {
                "baseline": [
                    row for row in metric_rows if "baseline_accuracy" in row
                ],
            }
            summary_groups.update({
                model_name: values for model_name, values in model_metric_rows.items()
            })
            for system, values in summary_groups.items():
                if not values:
                    continue
                summary_writer.writerow({
                    "system": system,
                    **{
                        metric: f"{sum(row[metric] if system != 'baseline' else row[f'baseline_{metric}'] for row in values) / len(values):.3f}"
                        for metric in ("accuracy", "precision", "recall", "f1")
                    },
                })


def main() -> None:
    """Parse command-line options and run the selected evaluation set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("spelling_correction_results.csv"))
    parser.add_argument("--cases", type=Path, help="CSV with input_text and expected_text columns.")
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--llm-model", help="Hugging Face text-to-text model name or local path.")
    parser.add_argument(
        "--llm-models",
        nargs="+",
        help="Compare multiple Hugging Face models in one run.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional CSV containing mean baseline and LLM metrics.",
    )
    args = parser.parse_args()
    cases = load_cases(args.cases) if args.cases else DEFAULT_CASES
    run(
        cases, args.output, args.llm_model, args.max_distance, args.summary_output,
        args.llm_models,
    )
    print(f"Wrote comparison results to {args.output}")


if __name__ == "__main__":
    main()
