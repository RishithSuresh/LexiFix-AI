from spelling_correction import NoisyChannelSpellCorrector, word_accuracy


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
