"""CLI entrypoint for the curation GUI.

Usage::

    python -m cnmfe.gui --results <results_dir> --avi-folder <avi_dir>
                       [--ds-meta <ds_meta.json>] [--pattern '*.avi']

When ``--avi-folder`` is omitted we look in ``results_dir.parent`` for ``*.avi``
files. If none are found we exit with a clear message.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _autodetect_avi_folder(results_dir: Path) -> Path | None:
    for cand in (results_dir, results_dir.parent):
        avis = sorted(cand.glob("*.avi"))
        if avis:
            return cand
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cnmfe.gui",
        description="Curation GUI for CNMFe extraction results.",
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="Directory written by CNMFe.save (A.npz, C.npy, ...).")
    parser.add_argument("--avi-folder", type=Path, default=None,
                        help="Folder containing 0.avi, 1.avi, ... (auto-detected if omitted).")
    parser.add_argument("--ds-meta", type=Path, default=None,
                        help="Path to ds_meta.json (auto-detected next to --results if omitted).")
    parser.add_argument("--pattern", default="*.avi",
                        help="AVI glob pattern (default %(default)s).")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase log verbosity (-v info, -vv debug).")
    args = parser.parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(level=max(level, logging.DEBUG), format="%(levelname)s %(name)s: %(message)s")

    if not args.results.is_dir():
        print(f"error: --results {args.results} is not a directory", file=sys.stderr)
        return 2

    avi_folder = args.avi_folder or _autodetect_avi_folder(args.results)
    if avi_folder is None or not avi_folder.is_dir():
        print(
            "error: could not find an AVI folder. Pass --avi-folder explicitly.",
            file=sys.stderr,
        )
        return 2

    # Heavy imports happen here so --help is fast.
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print(
            "error: PyQt6 is required. Install with: pip install -e .[gui]",
            file=sys.stderr,
        )
        return 2
    from cnmfe.gui.curation_app import MainWindow, open_session
    from cnmfe.gui.curation_store import KMismatchError

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    try:
        session, reader, store = open_session(
            args.results,
            avi_folder,
            ds_meta_path=args.ds_meta,
            pattern=args.pattern,
        )
    except KMismatchError as e:
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"{e.path.name} expects K={e.file_K} but the current model has "
            f"K={e.model_K}. Move it aside (rename) or delete it, then relaunch."
        )
        box.exec()
        return 1

    win = MainWindow(session, reader, store)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
