"""Configuration loading with a small, dependency-free YAML-subset parser.

Supports the subset of YAML used by ``config.yaml``:
  * top-level and nested mappings via indentation
  * scalar values: int, float, bool, null, quoted strings, plain strings
  * inline lists of scalars: ``[a, b, c]``
  * comments beginning with ``#``

This removes a hard dependency on PyYAML so the pipeline runs in a bare venv.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_TRUE = {"true", "True", "TRUE", "yes", "on"}
_FALSE = {"false", "False", "FALSE", "no", "off"}
_NULL = {"null", "Null", "NULL", "none", "~", ""}

_INT_RE = re.compile(r"^[-+]?\d+$")
_FLOAT_RE = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?$")


def _split_top_level(s: str) -> list[str]:
    """Split a string on top-level commas (outside quotes/brackets)."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    cur: list[str] = []
    for ch in s:
        if quote is not None:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
            continue
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _strip_comment(val: str) -> str:
    quote: str | None = None
    for i, ch in enumerate(val):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch == "#":
            return val[:i].rstrip()
    return val.rstrip()


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s in _NULL:
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in _split_top_level(inner)]
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return s


def load_config(path: str | Path) -> dict[str, Any]:
    """Load ``config.yaml`` into a nested dict."""
    text = Path(path).read_text(encoding="utf-8")
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = _strip_comment(val).strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        _, parent = stack[-1]
        if val == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(val)
    return root


def resolve(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Resolve a dotted key path (``model.params.max_depth``) with a default."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
