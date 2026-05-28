"""Persistent curation state for the GUI.

Writes ``curation.json`` next to ``A.npz`` / ``C.npy``. Schema is v1 and is
deliberately database-friendly so the eventual SQL migration is a flat
ingest of per-component records keyed by ``(results_dir, k)``.

Design notes
------------
* On ``load_or_seed`` we either reuse an existing file (if its
  ``n_components`` matches the model's ``K`` and its ``version`` is supported)
  or seed a fresh state from the model's ``accepted_mask`` (each component
  ``accepted = auto_accepted`` and ``review_state = "unreviewed"``).
* All mutations bump ``updated_at`` (UTC ISO-8601) and call ``save`` to write
  atomically (``write tmp -> os.replace``). The caller can choose between
  immediate save (accept/reject/merge/split) and a debounced save (note/tag
  edits) via ``save`` vs the GUI's own timer.
* No deletes: components are never removed, only flagged via ``accepted``.
* ``merge_groups`` / ``split_requests`` capture *intent*; the actual re-fit
  lives outside the GUI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------

@dataclass
class ComponentDecision:
    k: int
    accepted: bool
    auto_accepted: bool
    note: str = ""
    tags: list[str] = field(default_factory=list)
    review_state: str = "unreviewed"  # "unreviewed" | "reviewed"

    @classmethod
    def from_dict(cls, d: dict) -> "ComponentDecision":
        return cls(
            k=int(d["k"]),
            accepted=bool(d["accepted"]),
            auto_accepted=bool(d["auto_accepted"]),
            note=str(d.get("note", "")),
            tags=list(d.get("tags", [])),
            review_state=str(d.get("review_state", "unreviewed")),
        )


@dataclass
class SplitRequest:
    k: int
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SplitRequest":
        return cls(k=int(d["k"]), note=str(d.get("note", "")))


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KMismatchError(RuntimeError):
    """Raised when an existing curation.json's ``n_components`` doesn't match
    the model's current ``K``. Callers (the GUI) should present a dialog
    offering: read-only / rename-and-reseed / cancel.
    """

    def __init__(self, path: Path, file_K: int, model_K: int):
        super().__init__(
            f"{path} has n_components={file_K} but the model has K={model_K}. "
            "Refusing to silently migrate."
        )
        self.path = path
        self.file_K = file_K
        self.model_K = model_K


@dataclass
class CurationState:
    """In-memory mirror of ``curation.json``."""

    version: int = SCHEMA_VERSION
    results_dir: str = ""
    n_components: int = 0
    reviewer: str | None = None
    updated_at: str = field(default_factory=_utc_now_iso)
    components: list[ComponentDecision] = field(default_factory=list)
    merge_groups: list[list[int]] = field(default_factory=list)
    split_requests: list[SplitRequest] = field(default_factory=list)

    # ---- (de)serialisation -------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "results_dir": self.results_dir,
            "n_components": self.n_components,
            "reviewer": self.reviewer,
            "updated_at": self.updated_at,
            "components": [asdict(c) for c in self.components],
            "merge_groups": [list(g) for g in self.merge_groups],
            "split_requests": [asdict(s) for s in self.split_requests],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurationState":
        return cls(
            version=int(d.get("version", SCHEMA_VERSION)),
            results_dir=str(d.get("results_dir", "")),
            n_components=int(d.get("n_components", 0)),
            reviewer=d.get("reviewer"),
            updated_at=str(d.get("updated_at", _utc_now_iso())),
            components=[
                ComponentDecision.from_dict(c) for c in d.get("components", [])
            ],
            merge_groups=[
                [int(k) for k in g] for g in d.get("merge_groups", [])
            ],
            split_requests=[
                SplitRequest.from_dict(s) for s in d.get("split_requests", [])
            ],
        )


class CurationStore:
    """File-backed mutator for a single ``curation.json``."""

    FILENAME = "curation.json"

    def __init__(self, state: CurationState, path: Path):
        self.state = state
        self.path = Path(path)

    # ---- factories ---------------------------------------------------

    @classmethod
    def default_path(cls, results_dir: "str | Path") -> Path:
        return Path(results_dir) / cls.FILENAME

    @classmethod
    def load_or_seed(
        cls,
        results_dir: "str | Path",
        *,
        auto_accepted_mask: np.ndarray,
        path: "str | Path | None" = None,
    ) -> "CurationStore":
        """Load existing ``curation.json`` from ``results_dir`` if present and
        compatible; otherwise seed a fresh state from ``auto_accepted_mask``.
        """
        results_dir = Path(results_dir)
        path = Path(path) if path is not None else cls.default_path(results_dir)
        K = int(auto_accepted_mask.shape[0])

        if path.exists():
            d = json.loads(path.read_text())
            state = CurationState.from_dict(d)
            if state.version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{path} has schema version {state.version}; "
                    f"this GUI supports up to {SCHEMA_VERSION}."
                )
            if state.n_components != K:
                raise KMismatchError(path, state.n_components, K)
            return cls(state, path)

        # Fresh seed.
        state = cls._seed_from_mask(results_dir, auto_accepted_mask)
        store = cls(state, path)
        store.save()
        return store

    @classmethod
    def _seed_from_mask(
        cls, results_dir: Path, auto_accepted_mask: np.ndarray
    ) -> CurationState:
        K = int(auto_accepted_mask.shape[0])
        components = []
        for k in range(K):
            auto = bool(auto_accepted_mask[k])
            components.append(
                ComponentDecision(
                    k=k,
                    accepted=auto,
                    auto_accepted=auto,
                    note="",
                    tags=[],
                    review_state="unreviewed",
                )
            )
        return CurationState(
            version=SCHEMA_VERSION,
            results_dir=str(results_dir),
            n_components=K,
            reviewer=None,
            updated_at=_utc_now_iso(),
            components=components,
            merge_groups=[],
            split_requests=[],
        )

    # ---- accessors ---------------------------------------------------

    @property
    def n_components(self) -> int:
        return self.state.n_components

    def component(self, k: int) -> ComponentDecision:
        return self.state.components[int(k)]

    def export_accepted_mask(self) -> np.ndarray:
        return np.array(
            [c.accepted for c in self.state.components], dtype=bool
        )

    def reviewed_count(self) -> int:
        return sum(
            1 for c in self.state.components if c.review_state == "reviewed"
        )

    # ---- mutators ----------------------------------------------------

    def _touch(self) -> None:
        self.state.updated_at = _utc_now_iso()

    def set_accepted(self, k: int, value: bool) -> None:
        self.state.components[k].accepted = bool(value)
        self._touch()

    def set_note(self, k: int, note: str) -> None:
        self.state.components[k].note = str(note)
        self._touch()

    def set_tags(self, k: int, tags: Iterable[str]) -> None:
        self.state.components[k].tags = [str(t) for t in tags]
        self._touch()

    def mark_reviewed(self, k: int, reviewed: bool = True) -> None:
        self.state.components[k].review_state = (
            "reviewed" if reviewed else "unreviewed"
        )
        self._touch()

    def set_reviewer(self, reviewer: str | None) -> None:
        self.state.reviewer = reviewer
        self._touch()

    def add_merge_group(self, ks: Iterable[int]) -> None:
        ks_sorted = sorted({int(k) for k in ks})
        if len(ks_sorted) < 2:
            raise ValueError("a merge group needs at least 2 components")
        for k in ks_sorted:
            if not 0 <= k < self.state.n_components:
                raise IndexError(
                    f"merge component {k} out of range [0, {self.state.n_components})"
                )
        self.state.merge_groups.append(ks_sorted)
        self._touch()

    def remove_merge_group(self, idx: int) -> None:
        self.state.merge_groups.pop(idx)
        self._touch()

    def add_split_request(self, k: int, note: str = "") -> None:
        if not 0 <= k < self.state.n_components:
            raise IndexError(
                f"split component {k} out of range [0, {self.state.n_components})"
            )
        self.state.split_requests.append(SplitRequest(k=int(k), note=str(note)))
        self._touch()

    # ---- IO ----------------------------------------------------------

    def save(self) -> None:
        """Atomic write: ``foo.json.tmp`` -> ``os.replace`` -> ``foo.json``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2))
        os.replace(tmp, self.path)
