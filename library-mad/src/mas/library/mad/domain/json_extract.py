"""Dependency-free extraction of a JSON object from LLM output.

Shared by the runtime engines (parsing structured votes) and the evaluation
package (parsing judge responses). Lives in :mod:`common` so neither of those
layers has to depend on the other's internals.

Behavior: try :func:`json.loads` first; otherwise scan for the first balanced
``{...}`` object while respecting strings and escapes; enforce a maximum size;
and raise :class:`ValueError` on any failure.
"""

from __future__ import annotations

import json
from typing import Any

MAX_JSON_CHARS = 100_000


def extract_json_object(text: str, *, max_chars: int = MAX_JSON_CHARS) -> dict[str, Any]:
    """Return the first balanced JSON object in ``text`` without evaluating it."""

    if len(text) > max_chars:
        raise ValueError("JSON response exceeded the configured size limit")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : index + 1])
                if not isinstance(parsed, dict):
                    raise ValueError("Response JSON must be an object")
                return parsed
    raise ValueError("Response contains an unbalanced JSON object")
