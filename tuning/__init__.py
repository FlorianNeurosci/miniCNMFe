"""CNMFe parameter-tuning workflow.

Point the tuner at one recording (an AVI folder or an ``mc.zarr``) and it tests
the recording, suggests motion-correction + extraction parameters, and writes a
self-contained report folder (``recommended_params.json`` + ``report.md`` +
PNG figures) you can use to judge quality.

Two depth modes:

- ``heuristic`` — fast image-based suggestions (blob_log neuron radius,
  seed-count knee, shift histograms). No full extraction.
- ``sweep`` — actually run ``fit_extract`` across a small grid of the key
  extraction knobs and score each candidate with ground-truth-free quality
  proxies (cell count, accepted fraction, ``corr(C, C+YrA)``, footprint area,
  SNR). Slower, but lets you judge real output quality.

The heuristics are lifted from ``live_runs/estimate_params.ipynb``; the sweep +
quality metrics + report rendering are new. Nothing here changes ``minicnmfe``
defaults — the tuner only reads/runs the pipeline.

Public API::

    from tuning import TunerConfig, run_tuning, SweepSpec
    cfg = TunerConfig(input_path=..., output_dir=..., mode="both", ...)
    result = run_tuning(cfg)   # writes the report folder, returns the result dict

See ``tune.py`` (CLI) and ``live_runs/tune.ipynb`` (interactive viewer).
"""

from tuning.sweep import SweepSpec
from tuning.tuner import TunerConfig, run_tuning
from tuning.validate import (
    good_defaults,
    read_session_meta,
    resolve_session_paths,
    validate_session,
)

__all__ = ["TunerConfig", "run_tuning", "SweepSpec",
           "validate_session", "good_defaults", "read_session_meta",
           "resolve_session_paths"]
