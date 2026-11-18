"""Text helpers shared by matching, de-duplication and classification."""
from __future__ import annotations

import re
import unicodedata

_OCR_FOLD = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "L": "I"})
_NON_ALNUM = re.compile(r"[^0-9A-Z]+")
_VOWELS = set("aeiou")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def compact(s) -> str:
    """Upper-case, accent-free, alphanumerics only (spaces and punctuation removed)."""
    if s is None:
        return ""
    return _NON_ALNUM.sub("", strip_accents(str(s)).upper())


def ocr_fold(s: str) -> str:
    """Map characters OCR confuses (0/O, 1/I/L, 5/S, 8/B) onto one symbol.

    Used only to *compare* strings: two strings that fold to the same value differ
    at most by those confusions.
    """
    return s.translate(_OCR_FOLD)


def text_key(s) -> str:
    """Comparison key for a description: decimal commas unified, compacted, OCR-folded."""
    if s is None:
        return ""
    s = re.sub(r"(?<=\d),(?=\d)", ".", str(s))
    s = re.sub(r"(?<=\d)\.(?=\d)", "D", s)  # keep the decimal point as a symbol
    return ocr_fold(compact(s))


def levenshtein(a: str, b: str, max_d: int) -> int:
    """Edit distance with early exit; returns max_d + 1 when the distance exceeds max_d."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_d:
        return max_d + 1
    if a == b:
        return 0
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        lo, hi = max(1, i - max_d), min(lb, i + max_d)
        if lo > 1:
            cur[lo - 1] = max_d + 1
        ca = a[i - 1]
        best = cur[0] if lo == 1 else max_d + 1
        for j in range(lo, hi + 1):
            v = prev[j - 1] + (ca != b[j - 1])
            if prev[j] + 1 < v:
                v = prev[j] + 1
            if cur[j - 1] + 1 < v:
                v = cur[j - 1] + 1
            cur[j] = v
            if v < best:
                best = v
        if hi < lb:
            cur[hi + 1 :] = [max_d + 1] * (lb - hi)
        if best > max_d:
            return max_d + 1
        prev = cur
    return prev[lb]


def digit_conflict(a: str, b: str) -> bool:
    """True when two compacted strings differ in a way that changes a number.

    "1800MM" vs "1500MM" is a different item; "R0OF" vs "ROOF" is OCR noise.
    Compared after removing the digits OCR can produce from letters.
    """
    da = re.findall(r"[234679]", a)
    db = re.findall(r"[234679]", b)
    if da != db:
        return True
    # ambiguous digits (0,1,5,8): conflict only if both sides have a digit and they differ
    na = re.findall(r"\d+", a)
    nb = re.findall(r"\d+", b)
    if len(na) == len(nb):
        for x, y in zip(na, nb):
            if len(x) == len(y) and x != y:
                return True
    return False


_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def stem(w: str) -> str:
    """Very light stemmer: plural and a few verbal endings. Language-agnostic enough."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def words(s) -> list[str]:
    """Lower-case alphabetic tokens (accents stripped), lightly stemmed."""
    if s is None:
        return []
    s = strip_accents(str(s))
    s = re.sub(r"(?<=[a-z])(?=[A-Z][a-z])", " ", s)  # split CamelCase
    return [stem(t) for t in _TOKEN.findall(s.lower())]


def skeleton(w: str) -> str:
    """First letter plus the following consonants: 'conveyor' -> 'cnvyr'."""
    if not w:
        return ""
    return w[0] + "".join(c for c in w[1:] if c not in _VOWELS)
