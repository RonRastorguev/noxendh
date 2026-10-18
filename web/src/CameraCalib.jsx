import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import LidarPicker3D from './LidarPicker3D.jsx'

// Camera calibration. Two ways to tell the server where the camera is relative to the lidar:
// - Point pairs (DH-26 / DH-29, any setup): click a spot in the camera picture, then the same spot in the 3D lidar
//   view (it opens from the lidar's own position, so it looks like the photo; a click picks the point you are
//   looking at). From 6 pairs on, the fit is previewed: the lidar points are drawn on the photo and each pair shows
//   its error, so a bad pair can be redone before saving. 8 or more pairs, spread out, near and far.
// - Table corners (DH-24): click the tabletop's four corners, when the table calibration found the diorama.
const PAIR_COLORS = ['#FF5A1F', '#3DDC84', '#4EA8FF', '#F5C542', '#C77DFF', '#FF6FAE', '#5EC2B0', '#FFFFFF']
const color = (i) => PAIR_COLORS[i % PAIR_COLORS.length]
const BAD_PX = 25

function roiCentre(roi) {
  if (!roi) return null
  if (roi.center) return [roi.center[0], roi.center[1], roi.z ? roi.z[0] + 0.1 : 0]
  if (roi.min && roi.max) return [(roi.min[0] + roi.max[0]) / 2, (roi.min[1] + roi.max[1]) / 2, roi.min[2] + 0.1]
  return null
}

function PointPairs({ target, onDone, setError, saving, setSaving }) {
  const [pairs, setPairs] = useState([])          // [{ px: [u, v], xyz: [x, y, z] | null }]
  const [cloud, setCloud] = useState(null)
  const [look, setLook] = useState(null)
  const [fit, setFit] = useState(null)            // preview result, or null
  const [overlay, setOverlay] = useState(true)
  const img = useRef(null)
  const over = useRef(null)

  useEffect(() => {
    Promise.all([api.cropPreview(target.scan_id), api.cropState().catch(() => null)]).then(([c, st]) => {
      setCloud(c)
      setLook(roiCentre(st && st.roi))
    }).catch((e) => setError(e.message))
  }, [target.scan_id, setError])

  const complete = pairs.filter((p) => p.xyz)
  const waitingFor = !pairs.length || pairs[pairs.length - 1].xyz ? 'image' : 'lidar'

  // preview the fit whenever there are enough complete pairs
  const key = JSON.stringify(complete)
  useEffect(() => {
    if (complete.length < target.min_pairs) { setFit(null); return }
    let stale = false
    const t = setTimeout(() => {
      api.previewCamera(target.scan_id, { pairs: complete }).then((r) => { if (!stale) setFit(r) }).catch(() => {})
    }, 150)
    return () => { stale = true; clearTimeout(t) }
  }, [key])                                       // eslint-disable-line react-hooks/exhaustive-deps

  // draw the lidar points on the photo with the previewed fit, and each pair's miss
  const draw = useCallback(() => {
    const cv = over.current, im = img.current
    if (!cv || !im) return
    const w = im.clientWidth, h = im.clientHeight
    cv.width = w; cv.height = h
    const g = cv.getContext('2d')
    g.clearRect(0, 0, w, h)
    if (!fit || !fit.ok || !cloud) return
    const [iw, ih] = target.image_size
    const sx = w / iw, sy = h / ih
    const K = fit.K, T = fit.T_cam_from_frame
    const proj = (x, y, z) => {
      const cx = T[0][0] * x + T[0][1] * y + T[0][2] * z + T[0][3]
      const cy = T[1][0] * x + T[1][1] * y + T[1][2] * z + T[1][3]
      const cz = T[2][0] * x + T[2][1] * y + T[2][2] * z + T[2][3]
      if (cz <= 0.02) return null
      return [(K[0][0] * cx / cz + K[0][2]) * sx, (K[1][1] * cy / cz + K[1][2]) * sy, cz]
    }
    if (overlay) {
      const step = Math.max(1, Math.floor(cloud.n / 30000))
      for (let i = 0; i < cloud.n; i += step) {
        const p = proj(cloud.f[i * 4], cloud.f[i * 4 + 1], cloud.f[i * 4 + 2])
        if (!p || p[0] < 0 || p[1] < 0 || p[0] >= w || p[1] >= h) continue
        const t = Math.min(1, Math.max(0, (p[2] - 0.3) / 3))          // near = orange, far = blue
        g.fillStyle = `rgba(${Math.round(255 - 180 * t)}, ${Math.round(110 + 60 * t)}, ${Math.round(40 + 215 * t)}, 0.75)`
        g.fillRect(p[0] - 1, p[1] - 1, 2, 2)
      }
    }
    complete.forEach((pr, i) => {
      const p = proj(...pr.xyz)
      if (!p) return
      const [cx, cy] = [pr.px[0] * sx, pr.px[1] * sy]
      g.strokeStyle = '#ffffff'; g.lineWidth = 1.5
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(p[0], p[1]); g.stroke()
      g.strokeStyle = color(pairs.indexOf(pr)); g.lineWidth = 2
      g.beginPath(); g.arc(p[0], p[1], 5, 0, Math.PI * 2); g.stroke()
    })
  }, [fit, cloud, overlay, key, target.image_size])      // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { draw() }, [draw])
  useEffect(() => {
    const ro = new ResizeObserver(draw)
    if (img.current) ro.observe(img.current)
    return () => ro.disconnect()
  }, [draw])

  const clickImage = (e) => {
    if (waitingFor !== 'image') return
    const r = img.current.getBoundingClientRect()
    const [w, h] = target.image_size
    setPairs((ps) => [...ps, { px: [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h], xyz: null }])
  }
  const pick = (xyz) => setPairs((ps) => ps.map((p, i) => (i === ps.length - 1 ? { ...p, xyz } : p)))
  const remove = (i) => setPairs((ps) => ps.filter((_, j) => j !== i))
  const errOf = (i) => {
    if (!fit || !fit.ok || !fit.pair_err_px) return null
    const k = complete.indexOf(pairs[i])
    return k < 0 ? null : fit.pair_err_px[k]
  }

  const save = async () => {
    setSaving(true); setError(null)
    try { onDone(await api.calibrateCamera(target.scan_id, { pairs: complete })) } catch (e) { setError(e.message) } finally { setSaving(false) }
  }

  return (
    <>
      <div className="muted small">
        {waitingFor === 'image' ? <>Pair {pairs.length + 1}: click a spot in the <b>camera picture</b>.</>
          : <>Pair {pairs.length}: click the <b>same spot</b> in the 3D lidar view. Drag to look around, scroll to zoom.</>}
        {' '}Corners and edges of things work best, spread across the picture, near and far. {complete.length}/{target.min_pairs}+ pairs (8 or more is best).
      </div>
      <div className="pair-panes">
        <div className="calib-img">
          <img ref={img} src={target.image_url} alt="Camera frame" onClick={clickImage} onLoad={draw} draggable={false}
               style={{ cursor: waitingFor === 'image' ? 'crosshair' : 'not-allowed' }} />
          <canvas ref={over} className="calib-overlay" />
          {pairs.map((p, i) => (
            <span key={i} className="calib-dot" style={{ left: `${(p.px[0] / target.image_size[0]) * 100}%`, top: `${(p.px[1] / target.image_size[1]) * 100}%`, background: color(i) }}>{i + 1}</span>
          ))}
        </div>
        <LidarPicker3D cloud={cloud} target={look} pairs={pairs} colorFor={color} active={waitingFor === 'lidar'} onPick={pick}
                       height={img.current ? img.current.clientHeight || 330 : 330} />
      </div>
      <div className="pair-bar">
        {pairs.map((p, i) => {
          const e = errOf(i)
          return (
            <span key={i} className={`pair-chip ${e != null && e > BAD_PX ? 'bad' : ''}`} style={{ borderColor: color(i) }}>
              <b style={{ color: color(i) }}>{i + 1}</b>{p.xyz ? (e != null ? ` ${Math.round(e)} px` : '') : ' · lidar?'}
              <button aria-label={`Remove pair ${i + 1}`} onClick={() => remove(i)} disabled={saving}>×</button>
            </span>
          )
        })}
      </div>
      {fit && (fit.ok
        ? <div className={`note ${fit.reproj_px > BAD_PX ? 'bad' : ''}`}>
            Fit: the clicks miss by {fit.reproj_px} px on average, camera {Math.round(fit.camera_to_lidar_m * 100)} cm from the lidar.
            {fit.reproj_px > BAD_PX ? ' Remove or redo the pairs marked red (white lines show each miss).' : ' The dots on the photo should sit on the objects.'}
            {fit.warnings.length ? ` ${fit.warnings.join(' ')}` : ''}
          </div>
        : <div className="note bad">{fit.reason}</div>)}
      <div className="actions">
        <label className="check"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Lidar on photo</label>
        <button className="btn ghost" disabled={!pairs.length || saving}
                onClick={() => setPairs((ps) => { const l = ps[ps.length - 1]; return l && l.xyz ? ps.map((p, i) => (i === ps.length - 1 ? { ...p, xyz: null } : p)) : ps.slice(0, -1) })}>Undo</button>
        <button className="btn ghost" disabled={!pairs.length || saving} onClick={() => setPairs([])}>Clear</button>
        <button className="btn primary" disabled={complete.length < target.min_pairs || waitingFor === 'lidar' || saving} onClick={save}>{saving ? 'Solving…' : 'Save'}</button>
      </div>
    </>
  )
}

function TableCorners({ target, onDone, setError, saving, setSaving }) {
  const [points, setPoints] = useState([])
  const img = useRef(null)
  const click = (e) => {
    if (points.length >= 4) return
    const r = img.current.getBoundingClientRect()
    const [w, h] = target.image_size
    setPoints((p) => [...p, [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h]])
  }
  const save = async () => {
    setSaving(true); setError(null)
    try { onDone(await api.calibrateCamera(target.scan_id, { corners: points })) } catch (e) { setError(e.message) } finally { setSaving(false) }
  }
  return (
    <>
      <div className="muted small">Click the four corners of the tabletop, in any order. {4 - points.length > 0 ? `${4 - points.length} to go.` : 'Ready to save.'}</div>
      <div className="calib-img">
        <img ref={img} src={target.image_url} alt="Camera frame" onClick={click} draggable={false} />
        {points.map(([x, y], i) => (
          <span key={i} className="calib-dot" style={{ left: `${(x / target.image_size[0]) * 100}%`, top: `${(y / target.image_size[1]) * 100}%` }}>{i + 1}</span>
        ))}
      </div>
      <div className="actions">
        <button className="btn ghost" disabled={!points.length || saving} onClick={() => setPoints((p) => p.slice(0, -1))}>Undo</button>
        <button className="btn ghost" disabled={!points.length || saving} onClick={() => setPoints([])}>Clear</button>
        <button className="btn primary" disabled={points.length !== 4 || saving} onClick={save}>{saving ? 'Solving…' : 'Save'}</button>
      </div>
    </>
  )
}

export default function CameraCalib({ onClose, onDone, calibrated = false }) {
  const [target, setTarget] = useState(null)
  const [mode, setMode] = useState(null)
  const [error, setError] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.cameraTarget().then((t) => { setTarget(t); setMode(t.corners_ok ? 'corners' : 'pairs') }).catch((e) => setError(e.message))
  }, [])
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const props = { target, onDone, setError, saving, setSaving }
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal modal-wide" role="dialog" aria-label="Calibrate camera" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">Calibrate camera</div>
            {target && (
              <div className="seg" role="tablist" aria-label="Method" style={{ marginTop: 8 }}>
                <button className={mode === 'pairs' ? 'on' : ''} onClick={() => setMode('pairs')}>Point pairs</button>
                <button className={mode === 'corners' ? 'on' : ''} disabled={!target.corners_ok} onClick={() => setMode('corners')}
                        title={target.corners_ok ? '' : 'The table calibration did not find the diorama (it found a larger surface): use point pairs'}>Table corners</button>
              </div>
            )}
          </div>
          <div className="actions">
            {calibrated && (
              <button className="btn ghost" disabled={saving} title="Line the lidar's object outlines up with the photo's edges, starting from the current calibration. No clicks."
                      onClick={async () => {
                        setSaving(true); setError(null)
                        try { onDone(await api.autoAlignCamera()) } catch (e) { setError(e.message) } finally { setSaving(false) }
                      }}>{saving ? 'Aligning…' : 'Auto-align'}</button>
            )}
            <button className="btn ghost" onClick={onClose}>Close</button>
          </div>
        </div>
        {target && mode === 'pairs' && <PointPairs key="pairs" {...props} />}
        {target && mode === 'corners' && <TableCorners key="corners" {...props} />}
        {error && <div className="error modal-err" role="alert">{error}</div>}
      </div>
    </div>
  )
}
