from __future__ import annotations

import re
from collections.abc import Iterator

# Python subset for flight scripts
_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "False",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "None",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "True",
        "try",
        "while",
        "with",
        "yield",
    }
)

_NAME_RE = re.compile(r"[A-Za-z_]\w*")
# Python numeric literals with optional underscores (e.g. 180_000.0)
_NUMBER_RE = re.compile(r"\d[\d_]*(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")


def iter_flight_program_tokens(line: str) -> Iterator[tuple[str, str]]:
    """
    Split a single source line into (text, kind) spans.
    kind: ws, comment, string, number, keyword, name, op
    """
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in " \t":
            j = i + 1
            while j < n and line[j] in " \t":
                j += 1
            yield line[i:j], "ws"
            i = j
            continue
        if ch == "#":
            yield line[i:], "comment"
            break
        if ch in "\"'":
            q = ch
            j = i + 1
            esc = False
            while j < n:
                if esc:
                    esc = False
                    j += 1
                    continue
                if line[j] == "\\":
                    esc = True
                    j += 1
                    continue
                if line[j] == q:
                    j += 1
                    break
                j += 1
            yield line[i:j], "string"
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and line[i + 1].isdigit()):
            nm = _NUMBER_RE.match(line, i)
            if nm:
                yield nm.group(0), "number"
                i = nm.end()
                continue
        if ch.isalpha() or ch == "_":
            m = _NAME_RE.match(line, i)
            if m:
                word = m.group(0)
                if word in ("sim", "ap"):
                    kind = "api_root"
                elif word in _KEYWORDS:
                    kind = "keyword"
                else:
                    kind = "name"
                yield word, kind
                i = m.end()
                continue
        yield ch, "op"
        i += 1
