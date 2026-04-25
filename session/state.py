import json
import os
from pathlib import Path

_DEFAULT_PATH = "session_state.json"

_SERIALISABLE_TYPES = (str, int, float, bool, list, dict, type(None))


def _clean(obj):
    """Recursively drop non-JSON-serialisable values."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if isinstance(v, _SERIALISABLE_TYPES)}
    if isinstance(obj, list):
        return [_clean(i) for i in obj if isinstance(i, _SERIALISABLE_TYPES)]
    return obj


def save_session(state_dict: dict, path: str = _DEFAULT_PATH) -> None:
    """Persist a subset of Streamlit session state to disk as JSON."""
    Path(path).write_text(json.dumps(_clean(state_dict), indent=2))


def load_session(path: str = _DEFAULT_PATH) -> dict:
    """Load previously saved session state from JSON. Returns {} if file missing."""
    if not os.path.exists(path):
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return {}
