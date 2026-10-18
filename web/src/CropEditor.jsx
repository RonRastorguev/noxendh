import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'

// Hand crop: the newest scan seen from above, uncropped. Drag anywhere to draw a new box, drag its centre square to
// move it, drag an edge to resize it, drag the round knob above it to rotate it (or type the angle); scroll zooms,
// right-drag pans. Points inside the height band are bright, the rest dim. The box is saved as a rotated box
// {center, size, yaw_deg, z} in config/crop.yaml (perception/preprocess.roi_mask, DH-33) and used by every scan.
const EDGE_PX = 8
const KNOB_PX = 26

function pct(sorted, q) { return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] : 0 }

// box: { cx, cy, w, l, yaw } (metres, radians, yaw counter-clockwise from +x; w along the box's x, l along its y)
const axes = (b) => [[Math.cos(b.yaw), Math.sin(b.yaw)], [-Math.sin(b.yaw), Math.cos(b.yaw)]]
const local = (b, x, y) => { const [ex, ey] = axes(b); const dx = x - b.cx, dy = y - b.cy; return [dx * ex[0] + dy * ex[1], dx * ey[0] + dy * ey[1]] }
const world = (b, u, v) => { const [ex, ey] = axes(b); return [b.cx + u * ex[0] + v * ey[0], b.cy + u * ex[1] + v * ey[1]] }

function boxFromRoi(roi) {
  if (!roi) return null
  if (roi.center && roi.size) return { cx: roi.center[0], cy: roi.center[1], w: roi.size[0], l: roi.size[1], yaw: ((roi.yaw_deg || 0) * Math.PI) / 180 }
  if (roi.min && roi.max) return { cx: (roi.min[0] + roi.max[0]) / 2, cy: (roi.min[1] + roi.max[1]) / 2, w: roi.max[0] - roi.min[0], l: roi.max[1] - roi.min[1], yaw: 0 }
  return null
}
function bandFromRoi(roi) {
  if (!roi) return null
  if (roi.z) return { lo: roi.z[0], hi: roi.z[1] }
  if (roi.min && roi.max) return { lo: roi.min[2], hi: roi.max[2] }
  return null
}

export default function CropEditor({ onClose, onSaved }) {
  const wrap = useRef(null)
  const canvas = useRef(null)
  const layer = useRef(null)
  const [cloud, setCloud] = useState(null)
  const [state, setState] = useState(null)
  const [box, setBox] = useState(null)
  const [band, setBand] = useState({ lo: -0.05, hi: 0.5 })
  const [view, setView] = useState(null)
  const [size, setSize] = useState({ w: 800, h: 520 })
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)
  const lastYaw = useRef(0)
  const drag = useRef(null)

  useEffect(() => {
    Promise.all([api.cropState(), api.cropPreview()]).then(([st, pv]) => {
      setState(st)
      setCloud(pv)
      const b = boxFromRoi(st.roi)
      if (b) { setBox(b); lastYaw.current = b.yaw }
      const z = bandFromRoi(st.roi)
      if (z) setBand(z)
    }).catch((e) => setError(e.message))
  }, [])
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  useEffect(() => {
    const ro = new ResizeObserver(() => { const w = wrap.current.clientWidth; setSize({ w, h: Math.round(Math.min(560, w * 0.68)) }) })
    ro.observe(wrap.current)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!cloud || view) return
    const xs = [], ys = []
    for (let i = 0; i < cloud.n; i += 7) { xs.push(cloud.f[i * 4]); ys.push(cloud.f[i * 4 + 1]) }
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b)
    const x0 = pct(xs, 0.01), x1 = pct(xs, 0.99), y0 = pct(ys, 0.01), y1 = pct(ys, 0.99)
    const s = Math.min(size.w / Math.max(x1 - x0, 0.5), size.h / Math.max(y1 - y0, 0.5)) * 0.9
    setView({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s })
  }, [cloud, size, view])

  const toPx = useCallback((x, y) => [size.w / 2 + (x - view.cx) * view.s, size.h / 2 - (y - view.cy) * view.s], [size, view])
  const toW = useCallback((px, py) => [view.cx + (px - size.w / 2) / view.s, view.cy - (py - size.h / 2) / view.s], [size, view])

  // points layer
  useEffect(() => {
    if (!cloud || !view) return
    const c = layer.current || (layer.current = document.createElement('canvas'))
    c.width = size.w; c.height = size.h
    const ctx = c.getContext('2d')
    const img = ctx.createImageData(size.w, size.h)
    const d = img.data
    const span = Math.max(band.hi - band.lo, 0.05)
    for (let i = 0; i < cloud.n; i++) {
      const x = cloud.f[i * 4], y = cloud.f[i * 4 + 1], z = cloud.f[i * 4 + 2]
      const px = Math.round(size.w / 2 + (x - view.cx) * view.s), py = Math.round(size.h / 2 - (y - view.cy) * view.s)
      if (px < 0 || py < 0 || px >= size.w || py >= size.h) continue
      let r, g, b
      if (z >= band.lo && z <= band.hi) {
        const t = Math.min(1, Math.max(0, (z - band.lo) / span))
        r = 90 + 143 * t; g = 97 + 138 * t; b = 107 + 131 * t
      } else { r = 38; g = 43; b = 51 }
      const o = (py * size.w + px) * 4
      if (d[o + 3] && d[o] > r) continue
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }, [cloud, view, size, band])

  const knobWorld = (b) => world(b, 0, b.l / 2 + KNOB_PX / view.s)

  // compose: points + box + handles + lidar
  useEffect(() => {
    const cv = canvas.current
    if (!cv || !view || !layer.current) return
    cv.width = size.w; cv.height = size.h
    const ctx = cv.getContext('2d')
    ctx.fillStyle = '#0A0C0F'; ctx.fillRect(0, 0, size.w, size.h)
    ctx.drawImage(layer.current, 0, 0)
    if (cloud && cloud.lidar) {
      const [lx, ly] = toPx(cloud.lidar[0], cloud.lidar[1])
      ctx.fillStyle = '#3DDC84'; ctx.beginPath(); ctx.arc(lx, ly, 6, 0, Math.PI * 2); ctx.fill()     // the lidar (Ron: green)
      ctx.strokeStyle = 'rgba(61, 220, 132, .35)'; ctx.lineWidth = 4; ctx.beginPath(); ctx.arc(lx, ly, 9, 0, Math.PI * 2); ctx.stroke()
      ctx.font = '600 11px "Geist Mono", monospace'; ctx.fillText('LIDAR', lx + 13, ly + 4)
    }
    if (box) {
      const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([su, sv]) => toPx(...world(box, (su * box.w) / 2, (sv * box.l) / 2)))
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath()
      ctx.fillStyle = 'rgba(255, 90, 31, 0.08)'; ctx.fill()
      ctx.strokeStyle = '#FF5A1F'; ctx.lineWidth = 1.5; ctx.stroke()
      const [mx, my] = toPx(box.cx, box.cy)
      ctx.fillStyle = '#FF5A1F'; ctx.fillRect(mx - 6, my - 6, 12, 12)                                 // move handle
      const [tx, ty] = toPx(...world(box, 0, box.l / 2))
      const [kx, ky] = toPx(...knobWorld(box))
      ctx.strokeStyle = '#FF5A1F'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(kx, ky); ctx.stroke()
      ctx.fillStyle = '#0A0C0F'; ctx.beginPath(); ctx.arc(kx, ky, 7, 0, Math.PI * 2); ctx.fill()     // rotate knob
      ctx.strokeStyle = '#FF5A1F'; ctx.lineWidth = 2; ctx.stroke()
    }
  }, [view, size, box, cloud, band, toPx])

  const autoBand = (b) => {
    if (!cloud) return
    const zs = []
    for (let i = 0; i < cloud.n; i++) {
      const [u, v] = local(b, cloud.f[i * 4], cloud.f[i * 4 + 1])
      if (Math.abs(u) <= b.w / 2 && Math.abs(v) <= b.l / 2) zs.push(cloud.f[i * 4 + 2])
    }
    if (zs.length < 20) return
    zs.sort((a, c) => a - c)
    const lo = pct(zs, 0.02) - 0.03
    setBand({ lo: +lo.toFixed(3), hi: +Math.max(pct(zs, 0.995) + 0.05, lo + 0.2).toFixed(3) })
  }

  const hit = (px, py) => {
    if (!box) return null
    const [kx, ky] = toPx(...knobWorld(box))
    if (Math.hypot(px - kx, py - ky) < 11) return { kind: 'rotate' }
    const [mx, my] = toPx(box.cx, box.cy)
    if (Math.abs(px - mx) < 10 && Math.abs(py - my) < 10) return { kind: 'move' }
    const [u, v] = local(box, ...toW(px, py))
    const tol = EDGE_PX / view.s
    const inU = Math.abs(u) <= box.w / 2 + tol, inV = Math.abs(v) <= box.l / 2 + tol
    const edges = { l: inV && Math.abs(u + box.w / 2) < tol, r: inV && Math.abs(u - box.w / 2) < tol,
                    b: inU && Math.abs(v + box.l / 2) < tol, t: inU && Math.abs(v - box.l / 2) < tol }
    if (edges.l || edges.r || edges.t || edges.b) return { kind: 'edge', edges }
    return null
  }
  const pos = (e) => { const r = canvas.current.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] }

  const onDown = (e) => {
    if (!view) return
    e.preventDefault()
    const [px, py] = pos(e)
    const [wx, wy] = toW(px, py)
    if (e.button === 2 || e.button === 1) { drag.current = { kind: 'pan', px, py, v: view }; return }
    const h = hit(px, py)
    if (h && h.kind === 'rotate') drag.current = { kind: 'rotate' }
    else if (h && h.kind === 'move') drag.current = { kind: 'move', wx, wy, b: box }
    else if (h) drag.current = { kind: 'edge', edges: h.edges, b: box }
    else { drag.current = { kind: 'draw', wx, wy, yaw: lastYaw.current }; setBox({ cx: wx, cy: wy, w: 0, l: 0, yaw: lastYaw.current }) }
  }
  const onMove = (e) => {
    if (!view) return
    const [px, py] = pos(e)
    const [wx, wy] = toW(px, py)
    const g = drag.current
    if (!g) {
      const h = hit(px, py)
      canvas.current.style.cursor = !h ? 'crosshair' : h.kind === 'move' ? 'move' : h.kind === 'rotate' ? 'grab' : 'pointer'
      return
    }
    if (g.kind === 'pan') setView({ ...g.v, cx: g.v.cx - (px - g.px) / g.v.s, cy: g.v.cy + (py - g.py) / g.v.s })
    else if (g.kind === 'draw') {
      // a box at the current angle, from the press point to here
      const b0 = { cx: g.wx, cy: g.wy, w: 0, l: 0, yaw: g.yaw }
      const [u, v] = local(b0, wx, wy)
      const [cx, cy] = world(b0, u / 2, v / 2)
      setBox({ cx, cy, w: Math.abs(u), l: Math.abs(v), yaw: g.yaw })
    } else if (g.kind === 'move') setBox({ ...g.b, cx: g.b.cx + wx - g.wx, cy: g.b.cy + wy - g.wy })
    else if (g.kind === 'rotate') {
      const yaw = Math.atan2(wy - box.cy, wx - box.cx) - Math.PI / 2                // the knob sits on the box's +y side
      lastYaw.current = yaw
      setBox({ ...box, yaw })
    } else if (g.kind === 'edge') {
      const b = g.b
      const [u, v] = local(b, wx, wy)
      let u0 = -b.w / 2, u1 = b.w / 2, v0 = -b.l / 2, v1 = b.l / 2
      if (g.edges.l) u0 = Math.min(u, u1 - 0.02)
      if (g.edges.r) u1 = Math.max(u, u0 + 0.02)
      if (g.edges.b) v0 = Math.min(v, v1 - 0.02)
      if (g.edges.t) v1 = Math.max(v, v0 + 0.02)
      const [cx, cy] = world(b, (u0 + u1) / 2, (v0 + v1) / 2)
      setBox({ cx, cy, w: u1 - u0, l: v1 - v0, yaw: b.yaw })
    }
  }
  const onUp = () => {
    const g = drag.current
    drag.current = null
    if (!g || g.kind === 'pan' || !box) return
    if (box.w < 0.03 || box.l < 0.03) { if (g.kind === 'draw') setBox(null); return }
    if (g.kind === 'draw') autoBand(box)
  }
  const onWheel = (e) => {
    if (!view) return
    const [px, py] = pos(e)
    const [wx, wy] = toW(px, py)
    const s = Math.min(4000, Math.max(20, view.s * Math.exp(-e.deltaY * 0.0015)))
    setView({ s, cx: wx - (px - size.w / 2) / s, cy: wy + (py - size.h / 2) / s })
  }
  useEffect(() => {
    const cv = canvas.current
    const h = (e) => e.preventDefault()
    cv.addEventListener('wheel', h, { passive: false })
    return () => cv.removeEventListener('wheel', h)
  }, [])

  const angleDeg = box ? +((((box.yaw * 180) / Math.PI + 540) % 360) - 180).toFixed(1) : 0
  const setAngle = (deg) => {
    const yaw = (deg * Math.PI) / 180
    lastYaw.current = yaw
    setBox((b) => (b ? { ...b, yaw } : b))
  }

  const save = async () => {
    setSaving(true); setError(null)
    try {
      await api.setCrop({ center: [box.cx, box.cy], size: [box.w, box.l], yaw_deg: angleDeg, z: [band.lo, band.hi], scan_id: cloud.scanId })
      onSaved('Crop saved. Rescan to compare inside it.')
    } catch (e) { setError(e.message) } finally { setSaving(false) }
  }
  const reset = async () => {
    setSaving(true); setError(null)
    try { await api.resetCrop(); onSaved('Crop reset to the automatic table rectangle.') } catch (e) { setError(e.message) } finally { setSaving(false) }
  }

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Crop" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">Crop · seen from above</div>
            <div className="muted small">Drag a box around the diorama. Drag the square to move it, an edge to resize, the round knob to rotate. Scroll to zoom, right-drag to pan.</div>
          </div>
          <button className="btn ghost" onClick={onClose}>Close</button>
        </div>
        {state && state.stale_crop && <div className="note">Your last crop was drawn before the table was recalibrated, so it is not in use.</div>}
        <div ref={wrap} className="crop-canvas">
          <canvas ref={canvas} style={{ width: size.w, height: size.h }} onMouseDown={onDown} onMouseMove={onMove}
                  onMouseUp={onUp} onMouseLeave={onUp} onWheel={onWheel} onContextMenu={(e) => e.preventDefault()} />
          {!cloud && !error && <div className="viewer-loading eyebrow">Loading scan</div>}
        </div>
        <div className="crop-bar">
          <label className="crop-field"><span className="eyebrow">Angle (°)</span>
            <input type="number" step="0.5" value={angleDeg} disabled={!box} onChange={(e) => setAngle(+e.target.value)} /></label>
          <label className="crop-field"><span className="eyebrow">Bottom (m)</span>
            <input type="number" step="0.01" value={band.lo} onChange={(e) => setBand((b) => ({ ...b, lo: +e.target.value }))} /></label>
          <label className="crop-field"><span className="eyebrow">Top (m)</span>
            <input type="number" step="0.01" value={band.hi} onChange={(e) => setBand((b) => ({ ...b, hi: +e.target.value }))} /></label>
          <span className="eyebrow">{box ? `${box.w.toFixed(2)} × ${box.l.toFixed(2)} m` : 'No box yet'}{state ? ` · now: ${state.source} crop` : ''}</span>
        </div>
        {error && <div className="error modal-err" role="alert">{error}</div>}
        <div className="actions">
          <button className="btn ghost" disabled={saving || !state || state.source !== 'hand'} onClick={reset}>Reset to auto</button>
          <button className="btn primary" disabled={saving || !box || box.w < 0.03 || box.l < 0.03 || band.hi <= band.lo} onClick={save}>{saving ? 'Saving…' : 'Save crop'}</button>
        </div>
      </div>
    </div>
  )
}
