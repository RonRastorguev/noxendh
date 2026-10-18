# Scout first flights (2026-10-16): log `sess009/log103.ulg` (the last of 4 arm cycles)

Pulled read-only over MAVLink FTP (bridge tx log: only FTP reads + heartbeats, audit clean). `analysis_log103.txt` is
`tools/scout_fc/analyze_flight.py` on it. Ron took off in Position and reported it "pretty stable", but it "rocks a bit,
like in a circle".

- 111 s airborne: ~95 s Stabilized (ch 5 low), ~12 s Position (ch 5 high). ch 5 is a 2-position toggle, so Altitude
  (slot 4) is unreachable with the current mapping.
- Flow fused 92 % of airborne samples; range height 100 %. `SENS_FLOW_ROT` 2 and `SDLOG_PROFILE` 3 active.
- **Position hold works**: within 7–17 cm mean (22–44 cm max) of the hold point, height ±1 cm at 1.79 m. In Stabilized the
  position wanders ~0.6–1.1 m (no hold, expected).
- **The rocking = an inner-loop resonance at ~1.5 Hz, mostly pitch, worst in Position**: pitch ±4.7–5.9°; at 1.5 Hz the actual
  pitch is 6.3° vs 3.2° commanded and lags it by ~130°; the actual pitch rate lags its setpoint by ~70°. A healthy rate loop
  tracks 1.5 Hz with a few degrees of lag. Present but weaker in Stabilized (8–27 % of the band's power vs 52–62 %).
- Gains (set by whoever configured the airframe): `MC_ROLL/PITCHRATE_P` **0.08** (PX4 default 0.15), rate D 0.001
  (default 0.003), `MC_ROLL/PITCH_P` **8.0** (default 6.5). A weak rate loop under a high attitude P rings at a low
  frequency: consistent with this log.
- Vibration low (0.6 m/s²). Battery 14.49 → **13.05 V under load** (9.6 A mean); resting 14.37 V afterwards: charge before flying.
- Caveat: one log, 12 s of Position.

## Follow-ups (2026-10-16)

- Ron: took off with ch 5 LOW (= Stabilized), flipped HIGH (= Position) → the wobble started and the throttle "locked". The log
  confirms: in every Position segment the throttle stick sat inside the centre dead zone (−0.09 / +0.07 / 0.00 on −1…+1), which in
  Position means "hold height". In Stabilized his hover stick is also ~centre (median −0.01), so the same stick meant hover in one
  mode and hold-height in the other. To climb in Position / Altitude: stick clearly above centre (max 1.0 m/s, S2).
- Correction to the gain table above: PX4 scales rate P/I/D by `MC_*RATE_K` (roll 0.35, pitch 0.40), so the effective rate P is
  0.028 / 0.032; the values look like the 250-racer airframe preset (`SYS_AUTOSTART` 4050), not comparable to a 500 mm
  airframe. PX4 1.18's default attitude P is 4.0 (not 6.5).
- S4 prepared (Ron's values) and dry-run clean: `hardware/scout/PARAM_CHANGESET_S4.json`.
