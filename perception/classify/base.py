"""Classifier interface (ARCHITECTURE §6.8): before/after crops + event kind -> a strict-JSON classification."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

import numpy as np

SEVERITIES = ("none", "low", "medium", "high")

CONTEXT = ("You are reviewing an industrial facility inspection. A robot compared tonight's scan of an area with the "
           "reference scan and found one change. You get a BEFORE crop and an AFTER crop of the same spot. "
           "Moved inventory (pallets, crates, boxes relocated) is routine. Corrosion, damage, spills, leaks, burns, "
           "and objects blocking aisles or walkways are issues.")

KIND_HINT = {
    "moved": "The geometry says an object was MOVED from one place to another (a strong prior toward routine).",
    "added": "The geometry says a NEW object appeared here.",
    "removed": "The geometry says an object that used to be here is GONE.",
    "appearance": "The shape did not change, only the colour/appearance of a surface (e.g. rust, stain, burn).",
}


class NotConfigured(RuntimeError):
    """The provider has no credentials / API details: skip to the next one, not an error in the event."""


class ClassifierError(RuntimeError):
    """The provider answered, but not with a usable classification."""


@dataclass
class Classification:
    label: str
    routine: bool
    severity: str
    rationale: str
    confidence: float
    provider: str

    def as_dict(self) -> dict:
        return asdict(self)


def build_prompt(kind: str, context: dict | None = None) -> str:
    extra = ""
    if context:
        extra = " Measurements: " + json.dumps(context, sort_keys=True) + "."
    return (CONTEXT + " " + KIND_HINT.get(kind, "") + extra +
            ' Reply with ONLY a JSON object: {"label": short noun phrase, "routine": true|false, '
            '"severity": "none"|"low"|"medium"|"high", "rationale": one sentence, "confidence": 0..1}.')


def parse_response(text: str, provider: str) -> Classification:
    """Strict: the first {...} in the text must hold every field with the right type. Anything else is an error."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ClassifierError(f"{provider}: no JSON object in the reply")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ClassifierError(f"{provider}: reply is not valid JSON ({e})") from None
    try:
        label, routine, sev = str(d["label"]).strip(), d["routine"], str(d["severity"]).lower()
        rationale, conf = str(d["rationale"]).strip(), float(d["confidence"])
    except (KeyError, TypeError, ValueError) as e:
        raise ClassifierError(f"{provider}: missing or bad field ({e})") from None
    if not isinstance(routine, bool) or sev not in SEVERITIES or not label or not 0.0 <= conf <= 1.0:
        raise ClassifierError(f"{provider}: field out of range: {d}")
    return Classification(label, routine, sev, rationale, conf, provider)


def encode_png(img: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ClassifierError("could not encode crop")
    return buf.tobytes()
