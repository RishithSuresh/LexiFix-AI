from spelling_correction import (
    NoisyChannelSpellCorrector,
    _damerau_levenshtein_distance,
    _soundex,
    character_error_rate,
    correction_metrics,
    sentence_exact_match,
    word_accuracy,
    word_error_rate,
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
    assert word_accuracy("one", "one two") == 0.5


def test_word_accuracy_handles_empty_text():
    assert word_accuracy("", "") == 1.0
    assert word_accuracy("", "one") == 0.0


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


def test_invalid_configuration_is_rejected():
    try:
        NoisyChannelSpellCorrector(vocabulary=["word"], max_distance=-1)
    except ValueError as error:
        assert "max_distance" in str(error)
    else:
        raise AssertionError("Negative max_distance should raise ValueError")


def test_unmatched_word_is_left_unchanged():
    checker = NoisyChannelSpellCorrector(vocabulary=["hello"], max_distance=1)
    correction = checker.correct_word("zzzz")
    assert correction.corrected == "zzzz"
    assert correction.distance == -1
    assert correction.confidence == 0.0


def test_soundex_phonetic_matching():
    assert _soundex("frend") == _soundex("friend")
    assert _soundex("recieve") == _soundex("receive")
    assert _soundex("definitely") == _soundex("definately")


def test_damerau_transposition_distance():
    # Adjacent transposition is distance 1 in Damerau
    assert _damerau_levenshtein_distance("recieve", "receive") == 1
    assert _damerau_levenshtein_distance("cat", "act") == 1


def test_character_and_word_error_rate():
    assert character_error_rate("hello", "hello") == 0.0
    assert character_error_rate("hello", "helo") == 0.2
    assert word_error_rate("one two three", "one two three") == 0.0
    assert word_error_rate("one two three", "one three") == round(1 / 3, 4)
    assert sentence_exact_match("Hello World!", "hello world!") is True

