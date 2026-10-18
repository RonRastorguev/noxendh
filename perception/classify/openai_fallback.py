"""OpenAI vision fallback (ARCHITECTURE §6.8), plain HTTPS (no SDK dependency).

Needs OPENAI_API_KEY and a model name (config `classifier.openai_model` or env OPENAI_MODEL): no model is assumed.
Uses the public Chat Completions request shape with two base64 PNG crops as image parts; the reply must pass the same
strict JSON gate as every provider. Without a key or model it raises NotConfigured and the chain moves on.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request

from perception.classify.base import Classification, ClassifierError, NotConfigured, build_prompt, encode_png, parse_response

URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClassifier:
    name = "openai"

    def __init__(self, cfg: dict | None = None):
        c = (cfg or {}).get("classifier", {})
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.model = c.get("openai_model") or os.environ.get("OPENAI_MODEL")
        self.timeout_s = float(c.get("timeout_s", 8))

    def classify(self, before, after, kind: str, context: dict) -> Classification:
        if not self.api_key or not self.model:
            raise NotConfigured("OPENAI_API_KEY / model not set")
        img = lambda a: {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(encode_png(a)).decode()}}
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": build_prompt(kind, context) + " First image: BEFORE. Second image: AFTER."},
                    img(before), img(after)]}]}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                reply = json.load(r)
            text = reply["choices"][0]["message"]["content"]
        except (OSError, KeyError, IndexError, ValueError) as e:
            raise ClassifierError(f"openai: request failed ({type(e).__name__}: {e})") from None
        return parse_response(text, self.name)
