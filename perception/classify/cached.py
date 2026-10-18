"""Demo safety nets (ARCHITECTURE §6.8): replay the last good result for a similar event; then fixed rules.

`cached`: every successful classification from a real provider is stored with a coarse signature (kind + size bucket
+ intensity/colour bucket). A later event of the same kind gets the nearest stored one, provider "cached".
`rules`: last resort so an event is never left unclassified when the ballroom Wi-Fi dies. It says only what the
geometry already knows (provider "rules", confidence 0.3), never a diagnosis. Added beyond the spec (DH-18).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import replace

from perception.classify.base import Classification, NotConfigured


def signature(kind: str, context: dict) -> dict:
    return {"kind": kind, "size_cm": round(float(context.get("geo_score") or 0) / 2) * 2,
            "intensity": round(float(context.get("mean_intensity") or 0), 1)}


class CachedClassifier:
    name = "cached"

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._entries = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._entries = json.load(f)
            except (OSError, ValueError):
                self._entries = []

    def remember(self, kind: str, context: dict, c: Classification) -> None:
        if c.provider in ("cached", "rules"):
            return
        with self._lock:
            self._entries.append({"sig": signature(kind, context), "classification": c.as_dict()})
            self._entries = self._entries[-200:]
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=1)
            os.replace(tmp, self.path)

    def classify(self, before, after, kind: str, context: dict) -> Classification:
        sig = signature(kind, context)
        with self._lock:
            same = [e for e in self._entries if e["sig"]["kind"] == kind]
        if not same:
            raise NotConfigured(f"no cached result for a '{kind}' event yet")
        best = min(same, key=lambda e: abs(e["sig"]["size_cm"] - sig["size_cm"]) + 10 * abs(e["sig"]["intensity"] - sig["intensity"]))
        return replace(Classification(**best["classification"]), provider="cached")


RULES = {
    "moved": Classification("object relocated", True, "none", "The same object appears at a new position; relocated inventory is routine.", 0.3, "rules"),
    "added": Classification("new object", False, "medium", "Something that was not in the reference scan appeared here.", 0.3, "rules"),
    "removed": Classification("object missing", False, "medium", "An object from the reference scan is no longer here.", 0.3, "rules"),
    "appearance": Classification("surface change", False, "medium", "A surface changed colour without changing shape (possible corrosion, stain or burn).", 0.3, "rules"),
}


class RulesClassifier:
    name = "rules"

    def classify(self, before, after, kind: str, context: dict) -> Classification:
        return RULES.get(kind, Classification("unknown change", False, "low", "Unrecognised change kind.", 0.1, "rules"))
