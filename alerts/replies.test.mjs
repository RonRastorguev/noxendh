import test from "node:test";
import assert from "node:assert/strict";
import { alertText, confirmText, isAlertable, listText, openFlags, parseReply, pickFlag } from "./replies.mjs";

const ev = {
  id: "evt_1", kind: "appearance", bbox_map: { center: [0.4213, -0.12, 0.05], size: [0.1, 0.1, 0.1] },
  classification: { label: "corrosion on tank", routine: false, severity: "high", rationale: "Orange-brown patch on the tank wall." },
};

test("routine events never alert; unclassified ones do", () => {
  assert.equal(isAlertable(ev), true);
  assert.equal(isAlertable({ ...ev, classification: { ...ev.classification, routine: true } }), false);
  assert.equal(isAlertable({ id: "x", kind: "added" }), true);
});

test("alert text names the flag, severity, place and the three options", () => {
  const t = alertText(ev, 3, true);
  assert.match(t, /^Noxen flag #3: corrosion on tank \(high\)/);
  assert.match(t, /x 0\.42 m, y -0\.12 m/);
  assert.match(t, /Before and after below/);
  assert.match(t, /1 = send the Scout · 2 = real issue · 3 = false alarm/);
  assert.match(alertText({ id: "e", kind: "added" }, 1, false), /added change[\s\S]*No camera crop/);
});

test("replies parse to actions, with an optional #N", () => {
  assert.deepEqual(parseReply("1"), { action: "scout", n: null });
  assert.deepEqual(parseReply(" Scout! "), { action: "scout", n: null });
  assert.deepEqual(parseReply("2 #4"), { action: "confirmed", n: 4 });
  assert.deepEqual(parseReply("#2 3"), { action: "false_alarm", n: 2 });
  assert.deepEqual(parseReply("false alarm #5"), { action: "false_alarm", n: 5 });
  assert.deepEqual(parseReply("list"), { action: "list", n: null });
  assert.deepEqual(parseReply("what is this"), { action: null, n: null });
  assert.deepEqual(parseReply("12"), { action: null, n: null });     // a bare number is an option, not a flag
});

test("a bare reply answers the most recent open flag", () => {
  const s = { flags: { 1: { status: "open", label: "a" }, 2: { status: "confirmed", label: "b" }, 3: { status: "open", label: "c" } } };
  assert.deepEqual(openFlags(s), [1, 3]);
  assert.equal(pickFlag(s, null), 3);
  assert.equal(pickFlag(s, 2), 2);                                   // explicit #N may relabel a closed flag
  assert.equal(pickFlag(s, 9), null);
  assert.equal(pickFlag({ flags: {} }, null), null);
  assert.equal(listText(s), "#1: a\n#3: c");
});

test("confirmations echo the flag; the Scout reply carries no internal status", () => {
  const f = { label: "corrosion on tank" };
  const scout = confirmText("scout", 3, f);
  assert.match(scout, /Scout requested for flag #3 \(corrosion on tank\)/);
  assert.doesNotMatch(scout, /queued|not connected|Server/);
  assert.match(confirmText("confirmed", 3, f), /#3 .* real issue/);
  assert.match(confirmText("false_alarm", 3, f), /#3 .* false alarm/);
});
