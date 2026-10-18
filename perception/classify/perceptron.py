"""Perceptron (Mk1.5 / Mk1 vision-language model): the PRIMARY classifier (ARCHITECTURE §6.8, DH-27).

One request per change: the BEFORE and AFTER crops (PNG, inline) + the geometry hint + a closed list of categories.
The reply is constrained by a JSON schema (category = one of the configured ids, severity, rationale, confidence) and
still validated here: anything off-schema is a ClassifierError and the chain falls through, never a guess.

Whether a category is ROUTINE comes from the config, not from the model, so "object moved" is always routine and
"rust" never is. API: https://api.perceptron.inc/v1/chat/completions (OpenAI chat-completions format), key in env
PERCEPTRON_API_KEY (repo-root .env, gitignored; never in config or git).
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from perception.classify.base import (CONTEXT, KIND_HINT, SEVERITIES, Classification, ClassifierError, NotConfigured,
                                      encode_png)

DEFAULT_MODEL = "perceptron-mk1.5"
DEFAULT_BASE_URL = "https://api.perceptron.inc/v1"

# id -> (label shown in the UI / text, routine). Overridden by config classifier.perceptron.categories.
DEFAULT_CATEGORIES = [
    {"id": "rust", "label": "rust", "routine": False, "hint": "orange-brown discolouration"},
    {"id": "corrosion", "label": "corrosion", "routine": False},
    {"id": "leak", "label": "leak or spill", "routine": False},
    {"id": "burnt", "label": "burnt", "routine": False, "min_severity": "high",   # Ron 2026-10-18: black = burnt, and
     "hint": "the surface turned black, charred or sooty; scorch marks"},           # "a burn should be pretty damn severe"
    {"id": "damage", "label": "damage", "routine": False},
    {"id": "new_object", "label": "new object", "routine": False},
    {"id": "object_removed", "label": "object removed", "routine": False},
    {"id": "object_moved", "label": "object moved", "routine": True},
    {"id": "no_change", "label": "no real change", "routine": True},
    {"id": "other", "label": "other change", "routine": False},
]


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class PerceptronClassifier:
    name = "perceptron"

    def __init__(self, cfg: dict | None = None):
        c = ((cfg or {}).get("classifier") or {}).get("perceptron") or {}
        self.model = c.get("model", DEFAULT_MODEL)
        self.base_url = c.get("base_url", DEFAULT_BASE_URL).rstrip("/")
        self.http_timeout_s = float(c.get("http_timeout_s", 30))
        self.categories = c.get("categories") or DEFAULT_CATEGORIES
        self.by_id = {k["id"]: k for k in self.categories}
        self.api_key = os.environ.get("PERCEPTRON_API_KEY")

    # -- request ------------------------------------------------------------------------------------------------------
    def prompt(self, kind: str, context: dict | None) -> str:
        # `hint` (optional) is for the model only: what the category looks like; `label` stays short for the UI / texts
        choices = "; ".join(f'"{k["id"]}" = {k["label"]}' + (f' ({k["hint"]})' if k.get("hint") else "")
                            for k in self.categories)
        extra = (" Measurements from the lidar: " + json.dumps(context, sort_keys=True) + ".") if context else ""
        return (CONTEXT + " " + KIND_HINT.get(kind, "") + extra +
                " Image 1 is BEFORE, image 2 is AFTER, same spot. Diagnose the change as exactly one category: "
                + choices + '. Use "no_change" if the two images show no real difference (lighting, noise). '
                'severity is "none" for routine changes, else "low", "medium" or "high". '
                "rationale: one sentence on what you see. confidence: 0 to 1.")

    def schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": [k["id"] for k in self.categories]},
                "severity": {"type": "string", "enum": list(SEVERITIES)},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["category", "severity", "rationale", "confidence"],
            "additionalProperties": False,
        }

    def request_body(self, kind: str, context: dict | None, before_png: bytes, after_png: bytes) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "BEFORE:"},
                {"type": "image_url", "image_url": {"url": _data_url(before_png)}},
                {"type": "text", "text": "AFTER:"},
                {"type": "image_url", "image_url": {"url": _data_url(after_png)}},
                {"type": "text", "text": self.prompt(kind, context)},
            ]}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "change_diagnosis", "strict": True, "schema": self.schema()}},
            "max_completion_tokens": 400,
            "temperature": 0,
        }

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": "Bearer " + self.api_key,
                                              "Content-Type": "application/json",
                                              # Cloudflare in front of the API answers 403 / 1010 to urllib's default UA
                                              "User-Agent": "noxen-dh26/1.0"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout_s) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise ClassifierError(f"perceptron HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise ClassifierError(f"perceptron unreachable: {e.reason}") from None

    # -- reply --------------------------------------------------------------------------------------------------------
    def parse(self, reply: dict) -> Classification:
        try:
            choice = reply["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ClassifierError(f"perceptron: unexpected reply shape: {str(reply)[:200]}") from None
        if choice.get("finish_reason") == "length":
            raise ClassifierError("perceptron: reply cut off (max tokens)")
        try:
            d = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            raise ClassifierError(f"perceptron: reply is not JSON: {str(text)[:200]}") from None
        cat = self.by_id.get(d.get("category"))
        sev = str(d.get("severity", "")).lower()
        try:
            conf = float(d.get("confidence"))
        except (TypeError, ValueError):
            conf = -1.0
        if cat is None or sev not in SEVERITIES or not 0.0 <= conf <= 1.0 or not str(d.get("rationale", "")).strip():
            raise ClassifierError(f"perceptron: field out of range: {d}")
        routine = bool(cat.get("routine"))
        if routine:
            sev = "none"
        elif sev == "none":
            sev = "low"                                  # an issue is never severity "none"
        floor = cat.get("min_severity")                  # config, not the model: e.g. a burn is always "high" (Ron)
        if not routine and floor in SEVERITIES and SEVERITIES.index(sev) < SEVERITIES.index(floor):
            sev = floor
        return Classification(cat["label"], routine, sev, str(d["rationale"]).strip(), conf, self.name)

    def classify(self, before, after, kind: str, context: dict) -> Classification:
        if not self.api_key:
            raise NotConfigured("PERCEPTRON_API_KEY not set")
        return self.parse(self._post(self.request_body(kind, context, encode_png(before), encode_png(after))))
