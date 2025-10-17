import traceback
import time
from typing import Any, Dict, List

def add_traceback(obj, step: str, info: str, *, with_stack: bool = False) -> None:
    """
    Append a trace event to `obj.traceback_info`.

    Parameters
    ----------
    obj        : any object that owns a `traceback_info` list.
    step, info : short label and free-form description.
    with_stack : include trimmed call-stack (default False).
    """
    if not hasattr(obj, "traceback_info"):
        raise AttributeError(f"{obj!r} has no attribute 'traceback_info'")

    event: Dict[str, Any] = {
        "step":       step,
        "info":       info,
        "timestamp":  time.time(),
    }
    if with_stack:
        # omit the last frame (this helper)
        event["stack"] = traceback.format_stack()[:-1]

    obj.traceback_info.append(event)