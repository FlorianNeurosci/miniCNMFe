"""Headless smoke test for the curation GUI MainWindow.

We build a synthetic results dir + tiny AVI folder, construct MainWindow under
``QT_QPA_PLATFORM=offscreen``, and exercise the most important wiring:
component selection, frame change, accept toggle, merge, split.

Catches the bulk of "did we wire signal X to slot Y" mistakes without needing
a display.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import cv2-dependent modules BEFORE PyQt6 so cv2's bundled libgobject loads
# first; otherwise PyQt6's own glib wins and cv2 fails with
# "undefined symbol: g_string_copy".
import cv2  # noqa: F401, E402
import cnmfe.pipeline  # noqa: F401, E402
from tests.test_gui_session_data import _fake_results  # noqa: E402
from tests.test_avi_reader import _make_synthetic_avi  # noqa: E402

PyQt6 = pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from cnmfe.gui.curation_app import MainWindow, open_session  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def session_dirs(tmp_path: Path):
    """Create a matching results dir + 2-AVI folder.

    Movie shape: H=32, W=48, T=20. We pre-build the results to claim the
    same (H, W); ``SessionData.load`` enforces the match.
    """
    H, W = 32, 48
    avi_dir = tmp_path / "avis"
    avi_dir.mkdir()
    _make_synthetic_avi(avi_dir / "0.avi", n_frames=10, H=H, W=W)
    _make_synthetic_avi(avi_dir / "1.avi", n_frames=10, H=H, W=W)

    res_dir = tmp_path / "results"
    _fake_results(res_dir, H=H, W=W, K=3, T=20)
    return res_dir, avi_dir


def test_open_session(session_dirs):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    assert session.K == 3
    assert reader.n_frames == 20
    assert reader.dims == (32, 48)
    assert store.n_components == 3
    reader.close()


def test_mainwindow_construct_and_close(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)
    # The first component should be selected, the first frame loaded.
    assert win._current_k == 0
    assert win.transport.current_frame() >= 0
    # MovieView has its imshow artist.
    assert win.movie_view._im is not None
    # TraceView has its lines.
    assert win.trace_view._line_c is not None
    win.close()


def test_component_selection_drives_panes(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)

    # Pick k=2.
    win._on_component_selected(2)
    qapp.processEvents()
    assert win._current_k == 2
    # Curation panel's checkbox reflects the persisted accept state.
    expected = store.component(2).accepted
    assert win.curation_panel.accept_btn.isChecked() == expected
    win.close()


def test_accept_toggle_persists(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)

    win._on_component_selected(1)
    qapp.processEvents()
    initial = store.component(1).accepted

    win._on_accept(1, not initial)
    qapp.processEvents()
    assert store.component(1).accepted == (not initial)
    # File on disk reflects it too.
    import json
    raw = json.loads((res_dir / "curation.json").read_text())
    assert raw["components"][1]["accepted"] == (not initial)
    win.close()


def test_merge_two_components(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)

    win._on_merge_from_list([0, 2])
    assert store.state.merge_groups == [[0, 2]]
    win.close()


def test_split_request(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)

    win._on_split(1, "looks like two cells")
    assert len(store.state.split_requests) == 1
    assert store.state.split_requests[0].k == 1
    assert "two cells" in store.state.split_requests[0].note
    win.close()


def test_frame_change_updates_view(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)
    # Jump across the AVI boundary (10).
    win._on_frame_changed(11)
    qapp.processEvents()
    # No exception means we read across files. Frame label should mention 11.
    assert "frame 11" in win.transport.label.text()
    win.close()


def test_next_unreviewed_shortcut(session_dirs, qapp):
    res_dir, avi_dir = session_dirs
    session, reader, store = open_session(res_dir, avi_dir)
    win = MainWindow(session, reader, store)
    # Mark 0 reviewed so the search jumps past it.
    store.mark_reviewed(0, True)
    store.save()
    win._on_component_selected(0)
    qapp.processEvents()
    win._jump_next_unreviewed()
    qapp.processEvents()
    assert win._current_k != 0
    assert store.component(win._current_k).review_state == "unreviewed"
    win.close()
