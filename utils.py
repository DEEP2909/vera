from typing import Any


def deep_get(obj: dict[str, Any] | None, *paths: str) -> Any:
    if not obj:
        return None
    for path in paths:
        current: Any = obj
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, "", []):
            return current
    return None
