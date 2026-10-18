// iMessage alerts for flagged changes (ARCHITECTURE §9, DH-22), on Photon Spectrum (stable SDK, `spectrum-ts`).
//
//   cd alerts && npm start            (reads alerts/.env; the Demo A server must be running)
//   npm start -- --hello              send one test message and exit
//
// Listens to the server's WS /live. Every new non-routine event from a scan run with the dashboard's text toggle on
// (`text_alert` on the event_created message) is texted to NOXEN_ALERT_TO with its before/after
// crops and a numbered prompt. Replies from that number act on the flag: 1 -> POST /scout/dispatch,
// 2 -> feedback "confirmed", 3 -> feedback "false_alarm"; each gets a confirmation back.
// Flag numbers and their status live in data/alerts_state.json, so a restart keeps the numbering.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Spectrum, attachment } from "spectrum-ts";
import { imessage } from "spectrum-ts/providers/imessage";
import { ACTIONS, alertText, confirmText, describe, helpText, isAlertable, listText, parseReply, pickFlag } from "./replies.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SERVER = (process.env.NOXEN_SERVER || "http://localhost:8000").replace(/\/$/, "");
const TO = normPhone(process.env.NOXEN_ALERT_TO || "");
const STATE_PATH = process.env.NOXEN_ALERT_STATE || path.join(HERE, "..", "data", "alerts_state.json");

function log(...a) {
  console.log(new Date().toISOString().slice(11, 19), ...a);
}

function normPhone(p) {
  const d = String(p).replace(/[^\d+]/g, "");
  if (!d) return "";
  if (d.startsWith("+")) return d;
  return d.length === 10 ? "+1" + d : "+" + d;          // 10 digits = US
}

function loadState() {
  try {
    return JSON.parse(fs.readFileSync(STATE_PATH, "utf8"));
  } catch {
    return { next: 1, flags: {}, alerted: {} };
  }
}

function saveState(s) {
  fs.mkdirSync(path.dirname(STATE_PATH), { recursive: true });
  fs.writeFileSync(STATE_PATH + ".tmp", JSON.stringify(s, null, 2));
  fs.renameSync(STATE_PATH + ".tmp", STATE_PATH);
}

async function api(method, p, body) {
  const r = await fetch(SERVER + p, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${method} ${p} -> ${r.status} ${await r.text()}`);
  return r.json();
}

async function crop(url, name) {
  if (!url) return null;
  const r = await fetch(SERVER + url);
  if (!r.ok) return null;
  return attachment(Buffer.from(await r.arrayBuffer()), { name, mimeType: "image/png" });
}

if (!TO) {
  console.error("NOXEN_ALERT_TO is not set (alerts/.env). Nothing to send to.");
  process.exit(2);
}
if (!process.env.SPECTRUM_PROJECT_ID || !process.env.SPECTRUM_PROJECT_SECRET) {
  console.error("SPECTRUM_PROJECT_ID / SPECTRUM_PROJECT_SECRET missing (alerts/.env).");
  process.exit(2);
}

const app = await Spectrum({ providers: [imessage.config()] });   // credentials from SPECTRUM_PROJECT_ID / _SECRET
const im = imessage(app);
const dm = await im.space.create(TO);
const state = loadState();

if (process.argv.includes("--hello")) {
  await dm.send("Noxen alerts are connected. Flagged changes will arrive here with before/after images.");
  log("sent hello to", TO);
  await app.stop();
  process.exit(0);
}

// ---- outbound: a flagged event -> one iMessage thread entry ---------------------------------------------------------
let sending = Promise.resolve();                         // alerts go out one at a time, in event order

function onEvent(ev) {
  if (!ev || !ev.id || state.alerted[ev.id] || !isAlertable(ev)) return;
  state.alerted[ev.id] = true;
  const n = state.next++;
  state.flags[String(n)] = { event_id: ev.id, scan_id: ev.scan_id, label: describe(ev), status: "open", scout: false,
                             sent_at: new Date().toISOString() };
  saveState(state);
  sending = sending.then(async () => {
    try {
      const full = await api("GET", `/events/${ev.id}`);         // adds crop_before_url / crop_after_url
      const imgs = (await Promise.all([crop(full.crop_before_url, `flag${n}_before.png`),
                                       crop(full.crop_after_url, `flag${n}_after.png`)])).filter(Boolean);
      await dm.send(alertText(full, n, imgs.length > 0), ...imgs);
      log(`flag #${n} sent: ${describe(full)} (${imgs.length} image(s))`);
    } catch (e) {
      log(`flag #${n} FAILED to send:`, e.message);
    }
  });
}

function onEventUpdated(ev) {                            // answered on the dashboard: close the flag quietly
  if (!ev || !ev.feedback || ev.feedback === "none") return;
  for (const f of Object.values(state.flags)) {
    if (f.event_id === ev.id && f.status === "open") f.status = ev.feedback;
  }
  saveState(state);
}

function connectLive(delay = 1000) {
  const url = SERVER.replace(/^http/, "ws") + "/live";
  const ws = new WebSocket(url);
  let ping;
  ws.onopen = () => {
    log("connected to", url);
    delay = 1000;
    ws.send("bridge");                                   // the dashboard shows the text toggle as live only while we're here
    ping = setInterval(() => ws.readyState === 1 && ws.send("bridge"), 20000);
  };
  ws.onmessage = (m) => {
    let msg;
    try { msg = JSON.parse(m.data); } catch { return; }
    if (msg.type === "event_created" && msg.text_alert === true) onEvent(msg.event);   // dashboard text toggle was on
    else if (msg.type === "event_updated") onEventUpdated(msg.event);
  };
  ws.onclose = () => {
    clearInterval(ping);
    log(`server not reachable at ${url}; retrying in ${Math.round(delay / 1000)} s`);
    setTimeout(() => connectLive(Math.min(delay * 2, 30000)), delay);
  };
  ws.onerror = () => {};                                 // onclose follows and reconnects
}

connectLive();

// ---- inbound: replies from the alert number ----------------------------------------------------------------------
for await (const [space, message] of app.messages) {
  if (message.platform !== "imessage" || message.direction !== "inbound") continue;
  const from = normPhone(message.sender?.id || "");
  if (from !== TO) {
    log("ignored a message from an unknown sender");
    continue;
  }
  if (message.content.type !== "text") {
    await space.send(helpText(state));
    continue;
  }
  const { action, n } = parseReply(message.content.text);
  log(`reply: "${message.content.text}" -> ${action || "?"}${n != null ? " #" + n : ""}`);
  if (action === "help" || action === null) {
    await space.send((action === null ? "Didn't catch that.\n" : "") + helpText(state));
    continue;
  }
  if (action === "list") {
    await space.send(listText(state));
    continue;
  }
  const k = pickFlag(state, n);
  if (k == null) {
    await space.send(n != null ? `There is no flag #${n}.` : "There are no open flags right now.");
    continue;
  }
  const flag = state.flags[String(k)];
  try {
    if (action === "scout") {
      const dispatch = await api("POST", "/scout/dispatch", { event_id: flag.event_id });
      log(`flag #${k} scout dispatch: ${dispatch.status}`);
      flag.scout = true;
    } else {
      await api("POST", `/events/${flag.event_id}/feedback`, { feedback: ACTIONS[action].feedback });
      flag.status = ACTIONS[action].feedback;
    }
    saveState(state);
    await space.send(confirmText(action, k, flag));
  } catch (e) {
    log("action failed:", e.message);
    await space.send(`Couldn't do that for flag #${k}: the Noxen server didn't accept it (${e.message.slice(0, 120)}).`);
  }
}
