"""LexiFix AI — Spelling Intelligence and Grammar Correction Engine.

This module implements a hybrid spelling correction architecture:
1. Deterministic NLP Baseline:
   - Bayesian Noisy Channel Model: P(c|w) ~ P(w|c) * P(c)
   - Peter Norvig 1.1M-token corpus ('data/big.txt') with frequency tracking
   - Candidate generation via edits1 (deletions, transpositions, substitutions, insertions) and edits2
   - American Soundex phonetic encoding for phonetic slip matching (e.g. frend -> friend)
   - Damerau-Levenshtein distance calculation (handles character transpositions in 1 step)
   - Physical QWERTY keyboard Euclidean distance weighting for typo modeling
   - Bigram language model transition probabilities: P(w_i | w_{i-1}) and P(w_{i+1} | w_i)
   - Dynamic real-time corpus training and vocabulary expansion via train() / train_file()

2. Deep Learning Seq2Seq Transformers:
   - Hardware-accelerated Hugging Face Seq2Seq LM pipeline (PyTorch + CUDA/CPU)
   - Deterministic Beam Search decoding (num_beams=2) for high grammatical accuracy
   - Prompt-prefixed generation for instruction-following models (e.g., FLAN-T5, T5-GEC)

3. Comprehensive Evaluation Metrics:
   - Positional Word Accuracy
   - Character Error Rate (CER) via normalized string Levenshtein distance
   - Word Error Rate (WER) via token-level Levenshtein alignment
   - Sentence-level Exact Match (EM)
   - Token-level Precision, Recall, and F1 score
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib import import_module
import math
from pathlib import Path
import re
from typing import Iterable


# ==============================================================================
# SECTION 1: DEFAULT BASELINE VOCABULARY & CORPUS INITIALIZATION
# ==============================================================================
# Used as immediate offline fallback when external corpus files are unavailable.
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
grammar correction language spelling intelligence neural model baseline system
accuracy precision recall metric evaluation benchmark result dataset context
sentence token character algorithm artificial machine learning transform output
input human english word dictionary candidate ranking probability frequency
decision process document article verify explain analysis feature network
today tomorrow yesterday morning evening evening please thank welcome help
question answer problem solution succeed success successful successfully
believe belief receive receipt perceive perception achieve achievement
definite definitely separate separately separate occurrence occurred occurring
accommodate accommodation embarrassing embarrass recommend recommendation
necessary necessarily knowledge acknowledge privilege schedule colleague
foreign category rhythm parallel guarantee guarantee until forty twelfth
weather whether except accept affect effect principal principle lead led
lose loose choose chose though thought through throughout thoroughly
writing written piece peace quite quiet quit advice advise device devise
independent independence apparent appearance persistent persistence
noticeable visible eligible legible intelligent intelligence
software hardware computer python program develop developer test benchmark
interface studio application dataset table visual difference compare analysis
playing football play plays played sport sports soccer game games running
watch watching watched movie television study studying studied science
""".strip()

# ==============================================================================
# SECTION 2: TOKENIZATION & NORVIG CANDIDATE GENERATION
# ==============================================================================
# Uses Peter Norvig's classic spelling corrector formulation:
# Generates candidate sets at edit distances 1 and 2 using alphabet transformations.

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Path to the comprehensive 1.1M-token corpus if downloaded
CORPUS_FILE_PATH = Path(__file__).resolve().parent / "data" / "big.txt"


def _load_initial_corpus() -> str:
    """Load the comprehensive 1.1M token corpus if available, else default."""
    if CORPUS_FILE_PATH.is_file():
        try:
            return CORPUS_FILE_PATH.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return DEFAULT_CORPUS


# ==============================================================================
# SECTION 3: PHYSICAL KEYBOARD GEOMETRY & TYPO MODELING
# ==============================================================================
# Maps physical 2D coordinates of a QWERTY layout to calculate Euclidean distance
# between keys, heavily penalizing keys far apart on the keyboard.
QWERTY_COORDS: dict[str, tuple[int, int]] = {
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4), 'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    'a': (1, 0), 's': (1, 1), 'd': (1, 2), 'f': (1, 3), 'g': (1, 4), 'h': (1, 5), 'j': (1, 6), 'k': (1, 7), 'l': (1, 8),
    'z': (2, 0), 'x': (2, 1), 'c': (2, 2), 'v': (2, 3), 'b': (2, 4), 'n': (2, 5), 'm': (2, 6)
}


def _tokenize(text: str) -> list[str]:
    """Extract alphabetic words from raw text, retaining internal apostrophes."""
    return WORD_RE.findall(text)


def _edits1(word: str) -> set[str]:
    """Generate all candidate strings exactly 1 edit distance from input word.
    
    Covers four primary edit categories:
    1. Deletions: removing one character (e.g., 'word' -> 'wrd')
    2. Transpositions: swapping two adjacent characters (e.g., 'recieve' -> 'receive')
    3. Substitutions: replacing one character with any letter in the alphabet
    4. Insertions: adding any letter of the alphabet at any position
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in LETTERS]
    inserts = [L + c + R for L, R in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


def _edits2(word: str) -> set[str]:
    """Generate all candidate strings exactly 2 edits away by compounding edits1."""
    return set(e2 for e1 in _edits1(word) for e2 in _edits1(e1))


# ==============================================================================
# SECTION 4: PHONETIC ENCODING (AMERICAN SOUNDEX ALGORITHM)
# ==============================================================================
def _soundex(word: str) -> str:
    """Calculate the 4-character American Soundex phonetic hash of an English word.
    
    Soundex indexes words by how they sound rather than how they are spelled.
    Consonants that sound alike share the same digit code (e.g., B, F, P, V -> 1).
    Vowels and letters H/W are ignored. 
    Examples:
        'frend'   -> 'F653'
        'friend'  -> 'F653'  (Identical hash -> phonetic match!)
        'recieve' -> 'R210'
        'receive' -> 'R210'
    """
    if not word:
        return ""
    word_upper = word.upper()
    first_letter = word_upper[0]

    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }

    encoded = [first_letter]
    prev_code = mapping.get(first_letter, '')

    for char in word_upper[1:]:
        code = mapping.get(char, '')
        if code and code != prev_code:
            encoded.append(code)
            prev_code = code
        elif char not in ('H', 'W'):
            prev_code = ''

    digits = [c for c in encoded[1:] if c.isdigit()]
    res = first_letter + "".join(digits)
    return (res + "000")[:4]


def _keyboard_distance(char1: str, char2: str) -> float:
    """Euclidean distance between two keys on a QWERTY keyboard."""
    c1, c2 = char1.lower(), char2.lower()
    if c1 == c2:
        return 0.0
    if c1 in QWERTY_COORDS and c2 in QWERTY_COORDS:
        r1, col1 = QWERTY_COORDS[c1]
        r2, col2 = QWERTY_COORDS[c2]
        return math.sqrt((r1 - r2) ** 2 + (col1 - col2) ** 2)
    return 3.0


# ==============================================================================
# SECTION 5: STRING DISTANCE ALGORITHMS
# ==============================================================================
def _edit_distance(left: str, right: str) -> int:
    """Return standard Levenshtein distance using dynamic programming O(min(m, n)) space.
    
    Allowed operations: Insertion, Deletion, Substitution (cost = 1 each).
    """
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


def _damerau_levenshtein_distance(left: str, right: str) -> int:
    """Return true Damerau-Levenshtein distance.
    
    Allowed operations:
    1. Insertion
    2. Deletion
    3. Substitution
    4. Transposition of two adjacent characters (cost = 1).
    Crucial for typos like 'recieve' vs 'receive' where 'ie' -> 'ei' counts as 1 edit.
    """
    d: dict[tuple[int, int], int] = {}
    len_l, len_r = len(left), len(right)
    for i in range(-1, len_l + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, len_r + 1):
        d[(-1, j)] = j + 1

    for i, c_l in enumerate(left):
        for j, c_r in enumerate(right):
            cost = 0 if c_l == c_r else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,        # deletion
                d[(i, j - 1)] + 1,        # insertion
                d[(i - 1, j - 1)] + cost  # substitution
            )
            if i > 0 and j > 0 and c_l == right[j - 1] and left[i - 1] == c_r:
                d[(i, j)] = min(d[(i, j)], d[(i - 2, j - 2)] + 1)  # transposition

    return d[(len_l - 1, len_r - 1)]


@dataclass(frozen=True)
class Correction:
    """Represents the correction decision made for a single token."""

    original: str
    corrected: str
    distance: int
    confidence: float
    damerau_distance: int = 0
    phonetic_match: bool = False


# ==============================================================================
# SECTION 6: NOISY CHANNEL SPELL CORRECTOR (BAYESIAN NLP BASELINE)
# ==============================================================================
class NoisyChannelSpellCorrector:
    """Multi-signal spelling correction engine.

    Combines:
    - Bayesian prior word probabilities P(candidate) from token frequencies
    - Bigram context language model P(candidate | previous) and P(next | candidate)
    - Soundex phonetic matching bonus
    - Damerau transposition distance minimization
    - Physical QWERTY keyboard distance heuristics
    - Online learning / dynamic vocabulary ingestion
    """

    def __init__(
        self,
        corpus: str | Iterable[str] | None = None,
        *,
        vocabulary: Iterable[str] | None = None,
        max_distance: int = 2,
    ) -> None:
        if max_distance < 0:
            raise ValueError("max_distance must be zero or greater.")

        if vocabulary is not None:
            words = list(vocabulary)
        elif corpus is not None:
            raw_corpus = corpus if isinstance(corpus, str) else " ".join(corpus)
            words = _tokenize(raw_corpus)
        else:
            raw_corpus = _load_initial_corpus()
            words = _tokenize(raw_corpus)

        self.frequencies = Counter(word.lower() for word in words if word.isalpha())
        if not self.frequencies:
            raise ValueError("The corpus or vocabulary must contain at least one word.")

        # Build bigram transition model for contextual language modeling
        tokens_lower = [w.lower() for w in words if w.isalpha()]
        self.bigrams = Counter(
            (tokens_lower[i], tokens_lower[i + 1])
            for i in range(len(tokens_lower) - 1)
        )

        self.max_distance = max_distance
        self.vocabulary = frozenset(self.frequencies)
        self.total_tokens = sum(self.frequencies.values()) or 1

        # Precompute phonetic index for instant candidate retrieval
        self._soundex_index: dict[str, list[str]] = {}
        for w in self.vocabulary:
            sx = _soundex(w)
            self._soundex_index.setdefault(sx, []).append(w)

    def train(self, text_or_words: str | Iterable[str]) -> int:
        """Dynamically train the model on new text or vocabulary words.
        
        Extracts words, updates unigram frequencies, builds contextual bigram
        transition probabilities, and registers phonetic Soundex indices.
        """
        if isinstance(text_or_words, str):
            tokens = _tokenize(text_or_words)
        else:
            tokens = list(text_or_words)

        tokens_lower = [t.lower() for t in tokens if t.isalpha()]
        if not tokens_lower:
            return 0

        self.frequencies.update(tokens_lower)
        self.vocabulary = frozenset(self.frequencies)
        self.total_tokens = sum(self.frequencies.values())

        # Update bigram transitions
        for i in range(len(tokens_lower) - 1):
            self.bigrams[(tokens_lower[i], tokens_lower[i + 1])] += 1

        # Update Soundex index
        for w in set(tokens_lower):
            sx = _soundex(w)
            self._soundex_index.setdefault(sx, [])
            if w not in self._soundex_index[sx]:
                self._soundex_index[sx].append(w)

        return len(tokens_lower)

    def train_file(self, file_path: Path | str) -> int:
        """Train the model on an external text file."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        content = path.read_text(encoding="utf-8", errors="replace")
        return self.train(content)

    def detect(self, text: str) -> list[str]:
        """Return unique alphabetic tokens not present in the vocabulary."""
        return list(dict.fromkeys(
            token.lower() for token in _tokenize(text)
            if token.isalpha() and token.lower() not in self.vocabulary
        ))

    def _candidates(self, word: str) -> list[tuple[str, int]]:
        """Fast candidate extraction using edits1, Soundex index, and edits2."""
        candidates_map: dict[str, int] = {}

        # 1. Check edits1 (single deletion, transposition, substitution, insertion)
        for cand in _edits1(word):
            if cand in self.vocabulary:
                candidates_map[cand] = _edit_distance(word, cand)

        # 2. Check Soundex phonetic neighbors
        sx = _soundex(word)
        for cand in self._soundex_index.get(sx, []):
            if abs(len(cand) - len(word)) <= self.max_distance:
                if cand not in candidates_map:
                    dist = _edit_distance(word, cand)
                    if dist <= self.max_distance:
                        candidates_map[cand] = dist

        # 3. If few candidates found and max_distance >= 2, evaluate edits2
        if len(candidates_map) < 3 and self.max_distance >= 2:
            for cand in _edits2(word):
                if cand in self.vocabulary and cand not in candidates_map:
                    dist = _edit_distance(word, cand)
                    if dist <= self.max_distance:
                        candidates_map[cand] = dist

        # 4. Fallback for custom micro-vocabularies (e.g. unit tests with < 100 words)
        if not candidates_map and len(self.vocabulary) <= 200:
            for cand in self.vocabulary:
                if abs(len(cand) - len(word)) <= self.max_distance:
                    dist = _edit_distance(word, cand)
                    if dist <= self.max_distance:
                        candidates_map[cand] = dist

        return [(cand, dist) for cand, dist in candidates_map.items() if dist <= self.max_distance]

    def _score_candidate(
        self,
        candidate: str,
        word: str,
        distance: int,
        prev_word: str | None = None,
        next_word: str | None = None,
    ) -> float:
        """Calculate multi-signal heuristic score for candidate ranking."""
        freq = self.frequencies.get(candidate, 1)
        freq_score = math.log(1 + freq)

        # 1. Phonetic matching bonus
        sx_query = _soundex(word)
        sx_cand = _soundex(candidate)
        phonetic_bonus = 3.5 if sx_query == sx_cand else 0.0

        # 2. Damerau transposition bonus (e.g. 'recieve' -> 'receive')
        dam_dist = _damerau_levenshtein_distance(word, candidate)
        transposition_bonus = 2.0 if dam_dist < distance else 0.0

        # 3. Contextual Bigram Language Model P(cand | prev_word) and P(next_word | cand)
        bigram_bonus = 0.0
        if prev_word:
            p_prev = (prev_word.lower(), candidate)
            if p_prev in self.bigrams:
                bigram_bonus += 3.5 * math.log(1 + self.bigrams[p_prev])
        if next_word:
            p_next = (candidate, next_word.lower())
            if p_next in self.bigrams:
                bigram_bonus += 2.0 * math.log(1 + self.bigrams[p_next])

        # 4. Keyboard proximity bonus for single substitution edits
        kb_bonus = 0.0
        if len(word) == len(candidate) and distance == 1:
            for c_w, c_k in zip(word, candidate):
                if c_w != c_k:
                    kb_dist = _keyboard_distance(c_w, c_k)
                    if kb_dist <= 1.5:
                        kb_bonus += 1.5
                    break

        return freq_score + phonetic_bonus + transposition_bonus + bigram_bonus + kb_bonus

    def correct_word(
        self,
        word: str,
        prev_word: str | None = None,
        next_word: str | None = None,
    ) -> Correction:
        """Find the optimal spelling correction for a single token."""
        normalized = word.lower()
        if normalized in self.vocabulary:
            return Correction(word, word, 0, 1.0, 0, True)

        candidates = self._candidates(normalized)
        if not candidates:
            return Correction(word, word, -1, 0.0, -1, False)

        # Score candidates with contextual, phonetic, and language model signals
        scored_candidates = []
        for cand, dist in candidates:
            score = self._score_candidate(
                cand, normalized, dist, prev_word=prev_word, next_word=next_word
            )
            dam_dist = _damerau_levenshtein_distance(normalized, cand)
            # Effective distance: treats adjacent transpositions as 1 operation
            effective_dist = min(dist, dam_dist)
            scored_candidates.append((effective_dist, -score, cand, dist))

        # Primary sort: effective distance ascending
        # Secondary sort: multi-signal heuristic score descending (-score ascending)
        # Tertiary sort: alphabetical order for determinism
        scored_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        eff_dist, neg_score, best, best_dist = scored_candidates[0]

        # Calculate calibrated confidence estimate
        dam_dist = _damerau_levenshtein_distance(normalized, best)
        phonetic_match = (_soundex(normalized) == _soundex(best))
        freq_ratio = self.frequencies[best] / self.total_tokens

        base_conf = (1.0 / (1.0 + best_dist)) * (freq_ratio * 0.5 + 0.5)
        if phonetic_match:
            base_conf = min(1.0, base_conf + 0.15)
        if dam_dist < best_dist:
            base_conf = min(1.0, base_conf + 0.10)

        # Preserve original capitalization
        if word.istitle():
            best = best.title()
        elif word.isupper():
            best = best.upper()

        return Correction(
            original=word,
            corrected=best,
            distance=best_dist,
            confidence=min(round(base_conf, 4), 1.0),
            damerau_distance=dam_dist,
            phonetic_match=phonetic_match,
        )

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        """Correct an entire sentence, propagating contextual word tokens."""
        corrections: list[Correction] = []
        matches = list(WORD_RE.finditer(text))

        if not matches:
            return text, corrections

        token_list = [m.group(0) for m in matches]
        result_tokens: list[str] = []

        for idx, match in enumerate(matches):
            token = token_list[idx]
            prev_token = token_list[idx - 1] if idx > 0 else None
            next_token = token_list[idx + 1] if idx + 1 < len(token_list) else None

            corr = self.correct_word(token, prev_word=prev_token, next_word=next_token)
            corrections.append(corr)
            result_tokens.append(corr.corrected)

        # Reconstruct text preserving all non-word characters and whitespace
        parts: list[str] = []
        last_end = 0
        for match, repl in zip(matches, result_tokens):
            parts.append(text[last_end:match.start()])
            parts.append(repl)
            last_end = match.end()
        parts.append(text[last_end:])

        return "".join(parts), corrections


# ==============================================================================
# SECTION 7: DEEP LEARNING SEQ2SEQ TRANSFORMER BRIDGE (PYTORCH + HUGGING FACE)
# ==============================================================================
class LLMCorrector:
    """Hardware-accelerated Hugging Face Seq2Seq / text-to-text transformer bridge.

    Features:
    - Lazy loading with automatic CUDA GPU hardware acceleration detection
    - Deterministic beam search decoding (num_beams=2) for high grammatical consistency
    - Task prefix injection (e.g. 'grammar:', 'gec: ', 'Fix spelling: ') for instruction models
    - Robust fallback to standard transformers pipeline for legacy environments
    """

    def __init__(self, model_name: str, task_prefix: str = "grammar:") -> None:
        self.model_name = model_name
        self.task_prefix = task_prefix
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _load_model(self) -> None:
        transformers = import_module("transformers")

        # 1. Direct AutoModel Seq2Seq loader with CUDA acceleration
        try:
            auto_tok = getattr(transformers, "AutoTokenizer")
            auto_model = getattr(transformers, "AutoModelForSeq2SeqLM")
            self._tokenizer = auto_tok.from_pretrained(self.model_name)
            self._model = auto_model.from_pretrained(self.model_name)

            try:
                torch = import_module("torch")
                if torch.cuda.is_available():
                    self._device = "cuda"
                    self._model = self._model.to("cuda")
                else:
                    self._device = "cpu"
            except Exception:
                self._device = "cpu"

            self._model.eval()
            return
        except Exception:
            pass

        # 2. Fallback to pipeline for legacy environments
        try:
            pipeline_fn = getattr(transformers, "pipeline")
            try:
                self._pipeline = pipeline_fn("text2text-generation", model=self.model_name)
                return
            except (KeyError, ValueError):
                self._pipeline = pipeline_fn("text-generation", model=self.model_name)
                return
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face model '{self.model_name}': {exc}"
            ) from exc

    def correct(self, text: str) -> str:
        if self._model is None and self._pipeline is None:
            self._load_model()

        prompt = f"{self.task_prefix} {text}".strip()
        max_tokens = max(32, len(text.split()) * 3)

        # Direct Seq2Seq execution with Beam Search for highest grammatical fidelity
        if self._model is not None and self._tokenizer is not None:
            try:
                torch = import_module("torch")
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if self._device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                with torch.no_grad():
                    output_tokens = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        num_beams=2,
                        do_sample=False,
                    )
                decoded = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
                return self._clean_output(decoded)
            except Exception as exc:
                raise RuntimeError(f"Error during model generation: {exc}") from exc

        # Pipeline execution
        if self._pipeline is not None:
            result = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            if not result or "generated_text" not in result[0]:
                raise RuntimeError("The LLM returned no generated correction.")
            return self._clean_output(str(result[0]["generated_text"]))

        raise RuntimeError(f"Model '{self.model_name}' could not be initialized.")

    @staticmethod
    def _clean_output(text: str) -> str:
        """Normalize spacing around punctuation produced by some tokenizers."""
        text = text.strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text


# ==============================================================================
# SECTION 8: QUANTITATIVE EVALUATION & BENCHMARK METRICS
# ==============================================================================
def word_accuracy(reference: str, prediction: str) -> float:
    """Calculate case-insensitive positional word accuracy.
    
    Compares token-by-token alignment up to max length:
    accuracy = sum(match_i) / max(len(ref), len(pred))
    """
    expected = [word.lower() for word in _tokenize(reference)]
    actual = [word.lower() for word in _tokenize(prediction)]
    length = max(len(expected), len(actual))
    if not length:
        return 1.0
    expected.extend([""] * (length - len(expected)))
    actual.extend([""] * (length - len(actual)))
    return sum(a == b for a, b in zip(expected, actual)) / length


def character_error_rate(reference: str, prediction: str) -> float:
    """Compute normalized Levenshtein Character Error Rate (CER)."""
    if not reference and not prediction:
        return 0.0
    if not reference:
        return 1.0
    dist = _edit_distance(reference, prediction)
    return round(dist / max(len(reference), 1), 4)


def word_error_rate(reference: str, prediction: str) -> float:
    """Compute Word Error Rate (WER) using token-level Levenshtein alignment."""
    ref_words = _tokenize(reference.lower())
    pred_words = _tokenize(prediction.lower())
    if not ref_words and not pred_words:
        return 0.0
    if not ref_words:
        return 1.0

    # Token-level Levenshtein distance
    m, n = len(ref_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == pred_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return round(dp[m][n] / len(ref_words), 4)


def sentence_exact_match(reference: str, prediction: str) -> bool:
    """Return True if prediction exactly matches reference string ignoring case & edges."""
    return reference.strip().lower() == prediction.strip().lower()


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
