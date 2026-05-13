"""Run the code cells of tutorial_demo.ipynb in order to smoke-test it.

This is a lightweight stand-in for `jupyter nbconvert --execute` when
nbconvert isn't installed. Suppresses interactive plot windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI; plt.show() becomes a no-op
import matplotlib.pyplot as plt  # noqa: E402, F401

NB_PATH = Path(__file__).parent / "tutorial_demo.ipynb"
nb = json.load(open(NB_PATH, encoding="utf-8"))

# Shared namespace across cells, just like a real Jupyter kernel.
ns: dict = {"__name__": "__main__"}

# Ensure imports from the project root resolve (concat_avis_to_zarr is there).
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"\n========== cell {i} ==========\n{src}\n", flush=True)
    try:
        exec(compile(src, f"<nb cell {i}>", "exec"), ns)
    except Exception as exc:
        print(f"\n!!! cell {i} FAILED: {type(exc).__name__}: {exc}")
        raise

print("\n[smoke test] all code cells executed without exceptions.")
