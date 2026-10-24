"""Runtime-intervention bootstrap, injected via PYTHONPATH.

Every sglang process (launcher + TP workers) imports sitecustomize during
interpreter startup, so putting this directory on PYTHONPATH is enough to
install the hook before any model code runs.

Dormant unless BLACKFROST_RT_INTERVENTION points at a config file, so simply
leaving this directory on PYTHONPATH cannot alter a normal serve.
"""
import os
import sys

_CFG = os.environ.get("BLACKFROST_RT_INTERVENTION")
if _CFG:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        import rt_intervention  # noqa: F401  (import installs the hooks)
    except Exception as exc:  # never break serving on hook failure
        import logging

        logging.getLogger("blackfrost.rt_intervention").warning(
            "sitecustomize: failed to install runtime intervention: %s", exc
        )