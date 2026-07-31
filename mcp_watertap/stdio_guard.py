"""Descriptor-level stdout guard, shared by the simulation modules.

Under MCP stdio transport fd 1 carries the JSON-RPC stream, so a single stray
byte written by a solver corrupts it.
"""
from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def stdout_to_stderr():
    """Point OS-level fd 1 at fd 2 for the duration.

    contextlib.redirect_stdout only swaps sys.stdout, which is not enough: three
    of the writers here bypass it. IDAES's logging handler binds the original
    stdout at import time, ipopt writes from C, and Reaktoro's solver trace does
    the same. Redirecting the descriptor catches all of them.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
