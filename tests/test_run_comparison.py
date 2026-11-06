import csv

import pytest

from run_comparison import load_cases, run


def test_load_cases_rejects_blank_values(tmp_path):
    cases_path = tmp_path / "cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["input_text", "expected_text"])
        writer.writeheader()
        writer.writerow({"input_text": "hello", "expected_text": ""})

    with pytest.raises(ValueError, match="row 2"):
        load_cases(cases_path)


def test_run_rejects_empty_cases(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        run([], tmp_path / "results.csv", None)


def test_run_deduplicates_model_identifiers(tmp_path):
    output = tmp_path / "results.csv"
    run([("I recieve text", "I receive text")], output, None, llm_models=["model", "model"])

    with output.open(newline="", encoding="utf-8") as stream:
        fieldnames = next(csv.reader(stream))
    assert fieldnames.count("model_model_text") == 1
