"""Minimal CSV logger (one file per stream: train / episode / eval)."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, Optional, TextIO


class CSVLogger:
    """Appends rows to ``<run_dir>/<stream>.csv``.

    The header is fixed on the first row of each stream; later rows may omit
    keys (written empty). Appends on resume instead of overwriting.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._files: Dict[str, TextIO] = {}
        self._writers: Dict[str, csv.DictWriter] = {}

    def log(self, stream: str, row: Dict[str, Any]) -> None:
        if stream not in self._writers:
            path = os.path.join(self.run_dir, f"{stream}.csv")
            exists = os.path.exists(path) and os.path.getsize(path) > 0
            fieldnames = list(row.keys())
            if exists:
                with open(path, newline="", encoding="utf-8") as f:
                    header = next(csv.reader(f), None)
                if header:
                    fieldnames = header + [k for k in row if k not in header]
            f = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not exists:
                w.writeheader()
            self._files[stream], self._writers[stream] = f, w
        self._writers[stream].writerow(row)
        self._files[stream].flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        self._writers.clear()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def fmt_metrics(d: Dict[str, Any], keys: Optional[list] = None) -> str:
    items = ((k, d[k]) for k in (keys or d.keys()) if k in d)
    return " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in items)
