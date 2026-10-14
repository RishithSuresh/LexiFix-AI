"""Spelling detection and correction with a lightweight noisy-channel model.

The baseline uses word frequencies and edit distance. It has no model download
requirement, making it suitable for repeatable offline experiments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib import import_module
import re
from typing import Iterable


DEFAULT_CORPUS = """
the of and to in a is that for it as was with be by on not he i this are or his
from at which but have an had they you were their one all we can her has there
been if more when will would who so no about what up said out its into than them
then she many some these would time very make like him two how our work first
well way even new want because any these give day most us good people think
also back after use now only see other get how our year your just know take
come could over much before should through still such where while never right
world life hand place great little old own same tell does set three high every
large another become much family point keep last long thing often together
school group important different home under leave put
receive message friend beautiful useful example quick brown fox jumped lazy my dog text
"""

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance using two rows of O(min(len(left), len(right)))."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    distance: int
    confidence: float


class NoisyChannelSpellCorrector:
    """Detect out-of-vocabulary words and choose likely corrections."""

    def __init__(
        self,
        corpus: str | Iterable[str] = DEFAULT_CORPUS,
        *,
        vocabulary: Iterable[str] | None = None,
        max_distance: int = 2,
    ) -> None:
        words = vocabulary if vocabulary is not None else _tokenize(corpus if isinstance(corpus, str) else " ".join(corpus))
        self.frequencies = Counter(word.lower() for word in words if word.isalpha())
        self.max_distance = max_distance
        self.vocabulary = frozenset(self.frequencies)

    def detect(self, text: str) -> list[str]:
        """Return unique alphabetic tokens not present in the vocabulary."""
        return list(dict.fromkeys(
            token.lower() for token in _tokenize(text)
            if token.isalpha() and token.lower() not in self.vocabulary
        ))

    def _candidates(self, word: str) -> list[tuple[str, int]]:
        return [
            (candidate, _edit_distance(word, candidate))
            for candidate in self.vocabulary
            if abs(len(candidate) - len(word)) <= self.max_distance
            and _edit_distance(word, candidate) <= self.max_distance
        ]

    def correct_word(self, word: str) -> Correction:
        normalized = word.lower()
        if normalized in self.vocabulary:
            return Correction(word, word, 0, 1.0)
        candidates = self._candidates(normalized)
        if not candidates:
            return Correction(word, word, -1, 0.0)
        candidates.sort(key=lambda item: (item[1], -self.frequencies[item[0]], item[0]))
        best, distance = candidates[0]
        total = sum(self.frequencies.values()) or 1
        confidence = (1 / (1 + distance)) * (self.frequencies[best] / total + 0.5)
        if word.istitle():
            best = best.title()
        elif word.isupper():
            best = best.upper()
        return Correction(word, best, distance, min(confidence, 1.0))

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        corrections: list[Correction] = []

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            correction = self.correct_word(token)
            corrections.append(correction)
            return correction.corrected

        return WORD_RE.sub(replace, text), corrections


class LLMCorrector:
    """Optional Hugging Face text-to-text corrector.

    The model is loaded lazily. Configure a model such as a fine-tuned
    spelling-correction checkpoint with ``model_name``.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def correct(self, text: str) -> str:
        if self._pipeline is None:
            try:
                pipeline = getattr(import_module("transformers"), "pipeline")
            except (ImportError, AttributeError) as exc:
                raise RuntimeError("Install transformers to use the LLM corrector.") from exc
            self._pipeline = pipeline("text2text-generation", model=self.model_name)
        result = self._pipeline(
            f"Correct the spelling only. Preserve meaning and punctuation: {text}",
            max_new_tokens=max(32, len(text.split()) * 3),
            do_sample=False,
        )
        if not result or "generated_text" not in result[0]:
            raise RuntimeError("The LLM returned no generated correction.")
        return str(result[0]["generated_text"]).strip()


def word_accuracy(reference: str, prediction: str) -> float:
    """Calculate case-insensitive positional word accuracy."""
    expected = [word.lower() for word in _tokenize(reference)]
    actual = [word.lower() for word in _tokenize(prediction)]
    if not expected:
        return 1.0 if not actual else 0.0
    return sum(a == b for a, b in zip(expected, actual)) / len(expected)


def correction_metrics(source: str, reference: str, prediction: str) -> dict[str, float]:
    """Return token-level precision, recall, and F1 for spelling corrections."""
    source_words = [word.lower() for word in _tokenize(source)]
    reference_words = [word.lower() for word in _tokenize(reference)]
    prediction_words = [word.lower() for word in _tokenize(prediction)]
    length = max(len(source_words), len(reference_words), len(prediction_words))
    source_words.extend([""] * (length - len(source_words)))
    reference_words.extend([""] * (length - len(reference_words)))
    prediction_words.extend([""] * (length - len(prediction_words)))
    expected_changes = {
        index for index in range(length) if source_words[index] != reference_words[index]
    }
    predicted_changes = {
        index for index in range(length) if source_words[index] != prediction_words[index]
    }
    true_positive = len(expected_changes & predicted_changes)
    precision = true_positive / len(predicted_changes) if predicted_changes else 1.0
    recall = true_positive / len(expected_changes) if expected_changes else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
