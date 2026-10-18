"""Classifier chain (§6.8): strict JSON gate, per-provider timeout, fallbacks recorded, cache, Perceptron request/reply (DH-27).
No network. Runs under pytest or `python tests/test_classify.py`."""
from __future__ import annotations

import os
import sys
import tempfile
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception.classify.base import Classification, ClassifierError, NotConfigured, parse_response  # noqa: E402
from perception.classify.cached import CachedClassifier, RulesClassifier  # noqa: E402
from perception.classify.chain import ClassifierChain, default_chain  # noqa: E402
from perception.classify.perceptron import PerceptronClassifier  # noqa: E402

IMG = np.zeros((32, 32, 3), np.uint8)
GOOD = Classification("pallet relocated", True, "none", "Inventory moved.", 0.9, "fake")


class Fake:
    def __init__(self, name, behaviour):
        self.name, self.behaviour = name, behaviour

    def classify(self, before, after, kind, context):
        if self.behaviour == "slow":
            time.sleep(2.0)
            return GOOD
        if self.behaviour == "notcfg":
            raise NotConfigured("no key")
        if self.behaviour == "error":
            raise ClassifierError("garbage reply")
        return Classification(GOOD.label, GOOD.routine, GOOD.severity, GOOD.rationale, GOOD.confidence, self.name)


def test_parse_strict_json():
    c = parse_response('Sure! {"label": "rust on tank", "routine": false, "severity": "medium", '
                       '"rationale": "Orange-brown corrosion appeared.", "confidence": 0.8} done', "x")
    assert c.label == "rust on tank" and c.routine is False and c.severity == "medium"
    for bad in ("no json here", '{"label": "x"}', '{"label": "x", "routine": "no", "severity": "low", "rationale": "", "confidence": 0.5}',
                '{"label": "x", "routine": true, "severity": "critical", "rationale": "", "confidence": 0.5}',
                '{"label": "x", "routine": true, "severity": "low", "rationale": "", "confidence": 3}'):
        try:
            parse_response(bad, "x")
            raise AssertionError(f"accepted: {bad}")
        except ClassifierError:
            pass


def test_chain_falls_through_timeout_notconfigured_error_then_answers():
    chain = ClassifierChain([Fake("slow", "slow"), Fake("off", "notcfg"), Fake("bad", "error"), Fake("good", "ok")], timeout_s=0.3)
    t0 = time.time()
    c, attempts = chain.classify(IMG, IMG, "moved", {})
    assert c.provider == "good" and time.time() - t0 < 1.5
    assert [a["result"] for a in attempts] == ["timeout", "not configured", "error", "ok"], attempts


def test_cache_remembers_real_answers_and_replays_for_same_kind():
    path = os.path.join(tempfile.mkdtemp(), "cache.json")
    cache = ClassifierChain([Fake("good", "ok")], cache=CachedClassifier(path)).cache
    ClassifierChain([Fake("good", "ok")], cache=cache).classify(IMG, IMG, "moved", {"geo_score": 20})
    replay = CachedClassifier(path).classify(IMG, IMG, "moved", {"geo_score": 22})      # reloaded from disk
    assert replay.provider == "cached" and replay.label == GOOD.label
    try:
        CachedClassifier(path).classify(IMG, IMG, "appearance", {})
        raise AssertionError("cached answered for a kind it never saw")
    except NotConfigured:
        pass


def test_rules_never_claim_more_than_geometry():
    r = RulesClassifier()
    assert r.classify(IMG, IMG, "moved", {}).routine is True
    assert all(r.classify(IMG, IMG, k, {}).routine is False for k in ("added", "removed", "appearance"))
    assert all(r.classify(IMG, IMG, k, {}).confidence <= 0.3 for k in ("moved", "added", "removed", "appearance"))


def _reply(d, finish="stop"):
    import json
    return {"choices": [{"message": {"content": json.dumps(d) if isinstance(d, dict) else d}, "finish_reason": finish}]}


def test_perceptron_request_sends_both_crops_and_a_closed_category_list():
    """DH-27: no network. The body carries BEFORE then AFTER as inline PNGs and a JSON schema whose category enum is
    exactly the configured ids."""
    cats = [{"id": "rust", "label": "rust", "routine": False}, {"id": "object_moved", "label": "object moved", "routine": True}]
    p = PerceptronClassifier({"classifier": {"perceptron": {"model": "perceptron-mk1", "categories": cats}}})
    body = p.request_body("appearance", {"kind": "appearance"}, b"BEFORE-PNG", b"AFTER-PNG")
    parts = body["messages"][0]["content"]
    assert body["model"] == "perceptron-mk1"
    imgs = [x["image_url"]["url"] for x in parts if x["type"] == "image_url"]
    assert len(imgs) == 2 and all(u.startswith("data:image/png;base64,") for u in imgs)
    import base64
    assert [base64.b64decode(u.split(",", 1)[1]) for u in imgs] == [b"BEFORE-PNG", b"AFTER-PNG"]
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["category"]["enum"] == ["rust", "object_moved"] and schema["additionalProperties"] is False
    assert '"rust" = rust' in parts[-1]["text"] and "shape did not change" in parts[-1]["text"]


def test_perceptron_reply_routine_comes_from_config_and_off_schema_is_an_error():
    p = PerceptronClassifier({})
    c = p.parse(_reply({"category": "rust", "severity": "high", "rationale": "Orange flaking.", "confidence": 0.8}))
    assert (c.label, c.routine, c.severity, c.provider) == ("rust", False, "high", "perceptron")
    moved = p.parse(_reply({"category": "object_moved", "severity": "medium", "rationale": "Crate shifted.", "confidence": 0.7}))
    assert moved.routine is True and moved.severity == "none"                   # routine is the config's call
    assert p.parse(_reply({"category": "leak", "severity": "none", "rationale": "Puddle.", "confidence": 0.6})).severity == "low"
    for bad in ({"category": "alien", "severity": "low", "rationale": "x", "confidence": 0.5},
                {"category": "rust", "severity": "low", "rationale": "x", "confidence": 1.5},
                {"category": "rust", "severity": "low", "rationale": "", "confidence": 0.5}, "not json"):
        try:
            p.parse(_reply(bad))
            raise AssertionError(f"accepted {bad}")
        except ClassifierError:
            pass
    try:
        p.parse(_reply({"category": "rust", "severity": "low", "rationale": "x", "confidence": 0.5}, finish="length"))
        raise AssertionError("accepted a truncated reply")
    except ClassifierError:
        pass


def test_burnt_is_always_high_severity_whatever_the_model_says():
    """Ron 2026-10-18: a burn is severe. The config's min_severity raises the model's pick, never lowers it."""
    p = PerceptronClassifier({})                                              # the default categories include burnt
    for model_sev in ("none", "low", "medium", "high"):
        c = p.parse(_reply({"category": "burnt", "severity": model_sev, "rationale": "Charred patch.", "confidence": 0.85}))
        assert (c.label, c.routine, c.severity) == ("burnt", False, "high"), (model_sev, c)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "table_demo.yaml"), encoding="utf-8"))
    pc = PerceptronClassifier(cfg)
    assert pc.parse(_reply({"category": "burnt", "severity": "low", "rationale": "Soot.", "confidence": 0.9})).severity == "high"
    assert pc.parse(_reply({"category": "rust", "severity": "low", "rationale": "Spots.", "confidence": 0.9})).severity == "low"


def test_perceptron_without_key_is_not_configured():
    os.environ.pop("PERCEPTRON_API_KEY", None)
    try:
        PerceptronClassifier().classify(IMG, IMG, "added", {})
        raise AssertionError("answered without a key")
    except NotConfigured as e:
        assert "PERCEPTRON_API_KEY" in str(e)


def test_default_chain_without_network_ends_in_rules_and_records_why():
    for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "PERCEPTRON_API_KEY"):
        os.environ.pop(k, None)
    chain = default_chain({"classifier": {"primary": "perceptron", "fallbacks": ["openai", "cached"], "timeout_s": 2}},
                          os.path.join(tempfile.mkdtemp(), "cache.json"))
    assert [p.name for p in chain.providers] == ["perceptron", "openai", "cached", "rules"]
    events = [{"kind": "moved", "geo_score": 20, "crop_before": None, "crop_after": None},
              {"kind": "appearance", "geo_score": 30, "crop_before": None, "crop_after": None}]
    chain.classify_events(events)
    for ev in events:
        assert ev["classification"]["provider"] == "rules"
        assert [a["provider"] for a in ev["classification_attempts"]] == ["perceptron", "openai", "cached", "rules"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [ok] {name}")
    print(f"{len(tests)}/{len(tests)} passed")
