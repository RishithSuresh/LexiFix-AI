from spelling_correction import (
    NoisyChannelSpellCorrector,
    correction_metrics,
    word_accuracy,
)


def test_detects_unknown_words_and_preserves_known_words():
    checker = NoisyChannelSpellCorrector(vocabulary=["i", "receive", "message"])
    assert checker.detect("I recieve a message") == ["recieve", "a"]


def test_corrects_case_and_reports_distance():
    checker = NoisyChannelSpellCorrector(vocabulary=["receive"])
    correction = checker.correct_word("Recieve")
    assert correction.corrected == "Receive"
    assert correction.distance == 2


def test_correct_replaces_only_word_tokens():
    checker = NoisyChannelSpellCorrector(vocabulary=["hello", "world"])
    corrected, changes = checker.correct("Hello, worlld!")
    assert corrected == "Hello, world!"
    assert [change.original for change in changes] == ["Hello", "worlld"]


def test_word_accuracy_handles_different_lengths():
    assert word_accuracy("one two", "one two") == 1.0
    assert word_accuracy("one two", "one") == 0.5


def test_correction_metrics_measure_changed_tokens():
    metrics = correction_metrics(
        "I recieve text", "I receive text", "I receive text"
    )
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_correction_metrics_penalize_false_corrections():
    metrics = correction_metrics("I receive text", "I receive text", "I recieve text")
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.0
