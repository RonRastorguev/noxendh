// Pure logic for the iMessage alert loop (DH-22): what a flag alert says, and what a reply means.
// No network, no Spectrum: bridge.mjs does the I/O, this file is unit-tested (replies.test.mjs).

export const ACTIONS = {
  scout: { words: ["1", "scout", "look", "send", "send scout"], feedback: null },
  confirmed: { words: ["2", "issue", "real", "yes", "confirm", "confirmed", "flag"], feedback: "confirmed" },
  false_alarm: { words: ["3", "false", "false alarm", "no", "ignore", "dismiss", "fine"], feedback: "false_alarm" },
  help: { words: ["help", "?", "options"], feedback: null },
  list: { words: ["list", "status", "open"], feedback: null },
};

const PROMPT = "Reply 1 = send the Scout · 2 = real issue · 3 = false alarm";

export function isAlertable(ev) {
  // ARCHITECTURE §5: routine events never alert.
  return !(ev && ev.classification && ev.classification.routine === true);
}

function fmtM(v) {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(2) : "?";
}

export function where(ev) {
  const c = ev && ev.bbox_map && ev.bbox_map.center;
  return Array.isArray(c) ? `x ${fmtM(c[0])} m, y ${fmtM(c[1])} m on the map` : "location unknown";
}

export function describe(ev) {
  const cl = (ev && ev.classification) || {};
  return cl.label || (ev && ev.kind ? `${ev.kind} change` : "change");
}

export function alertText(ev, n, hasImages) {
  const cl = (ev && ev.classification) || {};
  const sev = cl.severity && cl.severity !== "none" ? ` (${cl.severity})` : "";
  const lines = [`Noxen flag #${n}: ${describe(ev)}${sev}`];
  if (cl.rationale) lines.push(cl.rationale);
  lines.push(`Where: ${where(ev)}`);
  lines.push(hasImages ? "Before and after below." : "No camera crop for this one.");
  lines.push(PROMPT);
  return lines.join("\n");
}

// "1", "scout", "#4 2", "3 #2", "false alarm #5" -> { action, n } (n = null: the most recent open flag).
export function parseReply(raw) {
  const text = String(raw || "").trim().toLowerCase().replace(/[.!]+$/, "");
  let n = null;
  const m = text.match(/#\s*(\d+)/);
  if (m) n = Number(m[1]);
  const rest = text.replace(/#\s*\d+/, " ").replace(/\s+/g, " ").trim();
  for (const [action, def] of Object.entries(ACTIONS)) {
    if (def.words.includes(rest)) return { action, n };
  }
  return { action: null, n };
}

export function pickFlag(state, n) {
  if (n != null) return state.flags[String(n)] ? n : null;
  const open = openFlags(state);
  return open.length ? open[open.length - 1] : null;
}

export function openFlags(state) {
  return Object.keys(state.flags).map(Number).filter((k) => state.flags[String(k)].status === "open").sort((a, b) => a - b);
}

export function helpText(state) {
  const open = openFlags(state);
  const which = open.length ? `Open flags: ${open.map((k) => "#" + k).join(", ")}. A bare reply answers #${open[open.length - 1]}.` : "No open flags.";
  return `${PROMPT}\nAdd #N to answer an older flag (e.g. "2 #3"). "list" shows open flags.\n${which}`;
}

export function listText(state) {
  const open = openFlags(state);
  if (!open.length) return "No open flags.";
  return open.map((k) => `#${k}: ${state.flags[String(k)].label}${state.flags[String(k)].scout ? " (Scout requested)" : ""}`).join("\n");
}

// The confirmation sent back after a reply was acted on. The dispatch status stays in the server log and the bridge
// log, not in the text: the Scout does not fly in the live demo (it is shown on video), DH-22.
export function confirmText(action, n, flag) {
  if (action === "scout") {
    return `OK, Scout requested for flag #${n} (${flag.label}).\nReply 2 or 3 once you've seen the close-up.`;
  }
  if (action === "confirmed") return `OK, flag #${n} (${flag.label}) marked as a real issue.`;
  if (action === "false_alarm") return `OK, flag #${n} (${flag.label}) marked as a false alarm and dismissed. That label goes into the training data.`;
  return "";
}
