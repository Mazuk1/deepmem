"""DeepMemory — Mem0-compatible memory service with 1/10 token cost."""

import os
import sys

# Ensure the vendored core engine (_core package) is importable
_vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)

__version__ = "0.1.0"
