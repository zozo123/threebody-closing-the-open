"""Canonical symbolic topology utilities.

ATLAS deliberately distinguishes a raw word, its freely/cyclically reduced word,
and any later symmetry quotient.  This module never silently identifies a word
with its inverse or with body-label permutations.
"""
from __future__ import annotations

import hashlib

_INV = {"a": "A", "A": "a", "b": "B", "B": "b"}
_VALID = frozenset(_INV)


def free_reduce(word: str) -> str:
    stack: list[str] = []
    for symbol in word.replace(" ", ""):
        if symbol not in _VALID:
            raise ValueError(f"invalid F2 symbol: {symbol!r}")
        if stack and _INV[symbol] == stack[-1]:
            stack.pop()
        else:
            stack.append(symbol)
    return "".join(stack)


def cyclic_reduce(word: str) -> str:
    reduced = free_reduce(word)
    chars = list(reduced)
    while len(chars) > 1 and _INV[chars[0]] == chars[-1]:
        chars = chars[1:-1]
    return "".join(chars)


def canonical_conjugacy_word(word: str) -> str:
    """Lexicographically minimal cyclic rotation after cyclic reduction."""
    reduced = cyclic_reduce(word)
    if not reduced:
        return ""
    rotations = (reduced[i:] + reduced[:i] for i in range(len(reduced)))
    return min(rotations)


def topology_hash(word: str) -> str:
    canonical = canonical_conjugacy_word(word)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
