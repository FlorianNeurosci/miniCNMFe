#!/usr/bin/env python3
"""Top-level shim so users can launch the curation GUI like the other CLIs.

Equivalent to ``python -m cnmfe.gui``; mirrors ``run_extract.py`` /
``run_evaluate.py`` style.
"""

from __future__ import annotations

import sys

from cnmfe.gui.__main__ import main


if __name__ == "__main__":
    sys.exit(main())
