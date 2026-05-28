"""Tests for cnmfe.gui.curation_store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cnmfe.gui.curation_store import (
    CurationStore,
    KMismatchError,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------- seed

def test_seed_from_accepted_mask(tmp_path: Path):
    mask = np.array([True, False, True, True, False], dtype=bool)
    store = CurationStore.load_or_seed(tmp_path, auto_accepted_mask=mask)
    assert store.path == tmp_path / "curation.json"
    assert store.path.exists()
    assert store.n_components == 5
    for k in range(5):
        c = store.component(k)
        assert c.k == k
        assert c.accepted == bool(mask[k])
        assert c.auto_accepted == bool(mask[k])
        assert c.review_state == "unreviewed"
        assert c.tags == []
        assert c.note == ""
    out = store.export_accepted_mask()
    np.testing.assert_array_equal(out, mask)


def test_seeded_file_has_expected_keys(tmp_path: Path):
    mask = np.array([True, True], dtype=bool)
    CurationStore.load_or_seed(tmp_path, auto_accepted_mask=mask)
    raw = json.loads((tmp_path / "curation.json").read_text())
    assert raw["version"] == SCHEMA_VERSION
    assert raw["n_components"] == 2
    assert raw["merge_groups"] == []
    assert raw["split_requests"] == []
    assert raw["reviewer"] is None
    assert len(raw["components"]) == 2
    assert set(raw["components"][0]) >= {
        "k", "accepted", "auto_accepted", "note", "tags", "review_state",
    }


# ---------------------------------------------------------------------- reuse

def test_load_existing_file(tmp_path: Path):
    mask = np.array([True, False, True], dtype=bool)
    store1 = CurationStore.load_or_seed(tmp_path, auto_accepted_mask=mask)
    store1.set_accepted(1, True)  # manual rescue
    store1.set_note(0, "hello world")
    store1.set_tags(2, ["neuron", "checked"])
    store1.mark_reviewed(0)
    store1.add_merge_group([0, 2])
    store1.add_split_request(1, "two cells overlapping")
    store1.set_reviewer("alice")
    store1.save()

    # Reload with same mask -> existing state preserved (no reseed).
    store2 = CurationStore.load_or_seed(tmp_path, auto_accepted_mask=mask)
    assert store2.component(0).note == "hello world"
    assert store2.component(0).review_state == "reviewed"
    assert store2.component(1).accepted is True  # manual rescue persisted
    assert store2.component(1).auto_accepted is False
    assert store2.component(2).tags == ["neuron", "checked"]
    assert store2.state.merge_groups == [[0, 2]]
    assert len(store2.state.split_requests) == 1
    assert store2.state.split_requests[0].k == 1
    assert store2.state.reviewer == "alice"


# ---------------------------------------------------------------------- K mismatch

def test_k_mismatch_raises(tmp_path: Path):
    mask = np.array([True, False, True], dtype=bool)
    CurationStore.load_or_seed(tmp_path, auto_accepted_mask=mask)
    # Now pretend K changed (re-extracted) -> refuse silent migration.
    with pytest.raises(KMismatchError) as ei:
        CurationStore.load_or_seed(
            tmp_path, auto_accepted_mask=np.array([True, False])
        )
    assert ei.value.file_K == 3
    assert ei.value.model_K == 2


def test_unknown_future_version_raises(tmp_path: Path):
    p = tmp_path / "curation.json"
    p.write_text(json.dumps({
        "version": 99,
        "n_components": 1,
        "components": [{"k": 0, "accepted": True, "auto_accepted": True}],
    }))
    with pytest.raises(RuntimeError):
        CurationStore.load_or_seed(
            tmp_path, auto_accepted_mask=np.array([True])
        )


# ---------------------------------------------------------------------- mutators

def test_merge_group_validation(tmp_path: Path):
    store = CurationStore.load_or_seed(
        tmp_path, auto_accepted_mask=np.array([True, True, True])
    )
    with pytest.raises(ValueError):
        store.add_merge_group([1])
    with pytest.raises(IndexError):
        store.add_merge_group([0, 99])
    # Deduplicated + sorted.
    store.add_merge_group([2, 0, 2])
    assert store.state.merge_groups[-1] == [0, 2]


def test_split_request_validation(tmp_path: Path):
    store = CurationStore.load_or_seed(
        tmp_path, auto_accepted_mask=np.array([True])
    )
    with pytest.raises(IndexError):
        store.add_split_request(5)
    store.add_split_request(0, "needs split")
    assert store.state.split_requests[0].k == 0


def test_updated_at_advances_on_mutation(tmp_path: Path):
    import time as time_

    store = CurationStore.load_or_seed(
        tmp_path, auto_accepted_mask=np.array([True, True])
    )
    t0 = store.state.updated_at
    time_.sleep(1.05)  # ISO format has second precision
    store.set_accepted(0, False)
    assert store.state.updated_at > t0


# ---------------------------------------------------------------------- IO

def test_atomic_save_no_tmp_left_behind(tmp_path: Path):
    store = CurationStore.load_or_seed(
        tmp_path, auto_accepted_mask=np.array([True])
    )
    store.set_note(0, "x")
    store.save()
    assert (tmp_path / "curation.json").exists()
    assert not (tmp_path / "curation.json.tmp").exists()


def test_export_accepted_mask_after_mutation(tmp_path: Path):
    store = CurationStore.load_or_seed(
        tmp_path, auto_accepted_mask=np.array([True, True, False])
    )
    store.set_accepted(0, False)
    store.set_accepted(2, True)
    np.testing.assert_array_equal(
        store.export_accepted_mask(),
        np.array([False, True, True]),
    )
