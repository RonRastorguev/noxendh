"""The provider chain (ARCHITECTURE §6.8): Perceptron -> OpenAI -> cached (-> rules), timeout per provider, parallel
per event. Every attempt is recorded on the event (`classification_attempts`) so the UI shows which provider answered
and why the earlier ones did not: fallbacks are never hidden.
"""
from __future__ import annotations

import concurrent.futures as cf
import time

import cv2
import numpy as np

from perception.classify.base import ClassifierError, NotConfigured
from perception.classify.cached import CachedClassifier, RulesClassifier

_BLANK = np.zeros((64, 64, 3), np.uint8)


def _read(path):
    img = cv2.imread(path) if path else None
    return img if img is not None else _BLANK


def context_of(ev: dict) -> dict:
    ctx = {"kind": ev["kind"], "geo_score": ev.get("geo_score")}
    if ev.get("bbox_map"):
        ctx["size_m"] = ev["bbox_map"]["size"]
    if ev.get("moved_from"):
        ctx["moved_cm"] = ev.get("geo_score")
    return ctx


class ClassifierChain:
    def __init__(self, providers: list, timeout_s: float = 8.0, cache: CachedClassifier | None = None, workers: int = 4):
        self.providers = list(providers)
        self.timeout_s = float(timeout_s)
        self.cache = cache
        self._pool = cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="classify_call")
        self.workers = workers

    def classify(self, before, after, kind: str, context: dict):
        attempts = []
        for p in self.providers:
            t0 = time.time()
            fut = self._pool.submit(p.classify, before, after, kind, context)
            try:
                c = fut.result(timeout=self.timeout_s)
            except cf.TimeoutError:
                attempts.append({"provider": p.name, "result": "timeout", "seconds": round(time.time() - t0, 2)})
                continue                    # the thread finishes in the background; its answer is ignored
            except NotConfigured as e:
                attempts.append({"provider": p.name, "result": "not configured", "detail": str(e)})
                continue
            except (ClassifierError, Exception) as e:   # any provider failure falls through to the next one
                attempts.append({"provider": p.name, "result": "error", "detail": f"{type(e).__name__}: {e}"})
                continue
            attempts.append({"provider": p.name, "result": "ok", "seconds": round(time.time() - t0, 2)})
            if self.cache is not None:
                self.cache.remember(kind, context, c)
            return c, attempts
        return None, attempts

    def classify_events(self, events: list, progress=None) -> list:
        """Classify every event in parallel (§6.8); fills event['classification'] and event['classification_attempts']."""
        done = [0]

        def one(ev):
            c, attempts = self.classify(_read(ev.get("crop_before")), _read(ev.get("crop_after")), ev["kind"], context_of(ev))
            ev["classification"] = c.as_dict() if c else None
            ev["classification_attempts"] = attempts
            done[0] += 1
            if progress:
                progress(done[0], len(events))
            return ev

        with cf.ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="classify_event") as ex:
            list(ex.map(one, events))
        return events


class _SwitchedOff:
    """Stands in for a provider the dashboard turned off: recorded as 'not configured' with the reason, never called."""

    def __init__(self, name: str, reason: str):
        self.name, self.reason = name, reason

    def classify(self, before, after, kind, context):
        raise NotConfigured(self.reason)


def with_provider_off(chain: ClassifierChain, name: str, reason: str) -> ClassifierChain:
    """The same chain (same cache, same timeout) with provider `name` switched off (DH-28: the Perceptron toggle)."""
    providers = [_SwitchedOff(name, reason) if p.name == name else p for p in chain.providers]
    return ClassifierChain(providers, chain.timeout_s, chain.cache, chain.workers)


def default_chain(cfg: dict, cache_path: str) -> ClassifierChain:
    """The configured order (config `classifier`), always ending in cached + rules so the demo never stalls."""
    from perception.classify.openai_fallback import OpenAIClassifier
    from perception.classify.perceptron import PerceptronClassifier
    c = cfg.get("classifier", {})
    cache = CachedClassifier(cache_path)
    by_name = {"perceptron": PerceptronClassifier(cfg), "openai": OpenAIClassifier(cfg), "cached": cache}
    order = [c.get("primary", "perceptron")] + list(c.get("fallbacks", ["openai", "cached"]))
    providers = [by_name[n] for n in order if n in by_name]
    if cache not in providers:
        providers.append(cache)
    providers.append(RulesClassifier())
    return ClassifierChain(providers, float(c.get("timeout_s", 8)), cache)
