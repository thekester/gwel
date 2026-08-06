"""VQA-style answer normalization and accuracy metrics.

Normalization follows the official VQAv2 evaluation code: lowercase, strip
articles and most punctuation, digitise number words, expand contraction
typos. ``vqa_accuracy`` implements the standard min(matches / 3, 1) score
against the annotator answers; ``exact_match`` is the stricter single-answer
variant used for datasets with one gold answer. ``anls`` is the edit-distance
score used officially by DocVQA, where long transcribed strings should not be
penalised to zero by a single character.
"""

import re

_ARTICLES = {"a", "an", "the"}

_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# Missing-apostrophe fixes from the official VQA evaluation script.
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itll": "it'll",
    "lets": "let's", "maam": "ma'am", "mightve": "might've", "mustve": "must've",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "wholl": "who'll", "whos": "who's",
    "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "yall": "y'all",
    "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
}

_PUNCT_STRIP = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`?,!]")
_PERIOD_STRIP = re.compile(r"(?<!\d)\.(?!\d)")  # keep decimal points in numbers


def normalize_answer(answer: str) -> str:
    """Normalize a free-form answer the way the VQA evaluation does."""
    text = answer.lower().strip()
    text = _PERIOD_STRIP.sub("", text)
    text = _PUNCT_STRIP.sub(" ", text)
    text = text.replace(":", " ")

    words = []
    for word in text.split():
        word = _NUMBER_WORDS.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def exact_match(prediction: str, gold_answers: list[str] | tuple[str, ...]) -> bool:
    """True when the normalized prediction equals any normalized gold answer."""
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(gold) for gold in gold_answers)


def _levenshtein(left: str, right: str) -> int:
    """Edit distance between two strings (iterative, O(min) memory)."""
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,               # deletion
                    current[j - 1] + 1,            # insertion
                    previous[j - 1] + (left_char != right_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


def anls(
    prediction: str,
    gold_answers: list[str] | tuple[str, ...],
    *,
    threshold: float = 0.5,
) -> float:
    """Average Normalized Levenshtein Similarity, the official DocVQA metric.

    Returns the best similarity ``1 - dist / max(len)`` over the gold answers,
    zeroed when it falls below ``threshold`` so that near-misses earn partial
    credit but unrelated answers earn none.
    """
    if not gold_answers:
        return 0.0
    normalized = normalize_answer(prediction)
    best = 0.0
    for gold in gold_answers:
        gold_normalized = normalize_answer(gold)
        longest = max(len(normalized), len(gold_normalized))
        if longest == 0:
            similarity = 1.0
        else:
            similarity = 1.0 - _levenshtein(normalized, gold_normalized) / longest
        best = max(best, similarity)
    return best if best >= threshold else 0.0


def vqa_accuracy(prediction: str, gold_answers: list[str] | tuple[str, ...]) -> float:
    """Standard VQA accuracy: min(#matching annotators / 3, 1).

    With fewer than three gold answers this degrades gracefully to exact
    match (0.0 or 1.0), so it is safe for single-answer datasets too.
    """
    if not gold_answers:
        return 0.0
    normalized = normalize_answer(prediction)
    matches = sum(normalized == normalize_answer(gold) for gold in gold_answers)
    if len(gold_answers) < 3:
        return 1.0 if matches > 0 else 0.0
    return min(matches / 3.0, 1.0)


def _as_number(text: str) -> float | None:
    """Parse a numeric answer, tolerating the separators charts carry."""
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def relaxed_accuracy(
    prediction: str,
    gold_answers: list[str] | tuple[str, ...],
    tolerance: float = 0.05,
) -> float:
    """ChartQA's metric: numeric answers within ``tolerance``, else exact match.

    A predicted number counts when it lands within a relative tolerance of the
    gold number, which is what the benchmark specifies for chart reading, where
    the answer is often read off an axis. Everything else falls back to the
    normalised exact match used elsewhere in this module.
    """
    predicted = _as_number(prediction)
    for gold in gold_answers:
        target = _as_number(str(gold))
        if predicted is not None and target is not None:
            if target == 0.0:
                if predicted == 0.0:
                    return 1.0
            elif abs(predicted - target) / abs(target) <= tolerance:
                return 1.0
        elif normalize_answer(prediction) == normalize_answer(str(gold)):
            return 1.0
    return 0.0
