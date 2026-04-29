from collections.abc import Mapping
from typing import Any


def deep_get(obj: Any, *paths: str) -> Any:
    if not isinstance(obj, Mapping):
        return None
    for path in paths:
        current: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, "", []):
            return current
    return None


def first_fact_with(patterns: list[str], facts: list[str]) -> str | None:
    lowered_patterns = [pattern.lower() for pattern in patterns]
    for pattern in lowered_patterns:
        for fact in facts:
            if pattern in fact.lower():
                return fact
    return None
