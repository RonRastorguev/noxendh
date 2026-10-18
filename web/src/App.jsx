import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, eventStatus } from './api.js'
import { useLive } from './useLive.js'
import Viewer3D from './Viewer3D.jsx'
import EventDetail from './EventDetail.jsx'
import CameraCalib from './CameraCalib.jsx'
import CropEditor from './CropEditor.jsx'

const STAGES = {
  accumulating: 'Scanning with lidar', camera_open: 'Opening camera', aligning: 'Aligning to baseline',
  comparing: 'Comparing geometry', 'comparing colours': 'Comparing colour', classifying: 'Classifying',
  written: 'Saving scan', done: 'Done', 'fitting table': 'Finding the table',
}
const SCENARIO_LABEL = { no_change: 'No change', crate_moved: 'Crate moved', rust_tank: 'Rust on tank',
                         dropped_in_aisle: 'Dropped in aisle', everything: 'All three' }

function fmtTime(iso) {
  const d = new Date(iso)
  return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 472 257" aria-hidden="true">
      <polygon points="0,0 78,0 286,257 210,257 56,66.7 56,257 0,257" />
      <polygon points="135,0 215,0 472,257 392,257" />
      <polygon points="392,0 472,0 365,107 325,67" />
    </svg>
  )
}

export default function App() {
  const [status, setStatus] = useState(null)
  const [scans, setScans] = useState([])
  const [scanId, setScanId] = useState(null)
  const [events, setEvents] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [camCalib, setCamCalib] = useState(false)
  const [cropping, setCropping] = useState(false)
  const [cloudVersion, setCloudVersion] = useState(0)
  const [scenario, setScenario] = useState('everything')

  const refresh = useCallback(async (selectLatest = false) => {
    const [st, sc] = await Promise.all([api.status(), api.scans()])
    setStatus(st)
    setScans(sc)
    if (selectLatest) {
      // only a scan compared against the CURRENT baseline; with none yet, the view shows the baseline itself
      const latest = sc.find((s) => !s.is_baseline && s.status === 'done' && s.baseline_id === st.baseline_id)
      if (latest) setScanId(latest.id)
    }
    return sc
  }, [])

  useEffect(() => { refresh(true).catch((e) => setError(e.message)) }, [refresh])

  useEffect(() => {
    if (!scanId) { setEvents([]); return }
    api.events(scanId).then((evs) => {
      setEvents(evs)
      const firstFlag = evs.find((e) => eventStatus(e) === 'flagged') || evs[0]
      setSelectedId(firstFlag ? firstFlag.id : null)
    }).catch((e) => setError(e.message))
  }, [scanId])

  const connected = useLive((m) => {
    if (m.type === 'scan_started') { setError(null); setNotice(null); setProgress({ kind: m.kind, stage: 'starting' }) }
    else if (m.type === 'scan_progress') setProgress((p) => ({ ...(p || {}), stage: m.stage, detail: m.detail, remaining: m.remaining_s }))
    else if (m.type === 'scan_done') { setProgress(null); refresh().then(() => setScanId(m.scan_id)) }
    else if (m.type === 'baseline_set') { setProgress(null); setScanId(null); refresh() }
    else if (m.type === 'scan_error') { setProgress(null); setError(m.error); refresh() }
    else if (m.type === 'camera_calibrated') { setCloudVersion((v) => v + 1); refresh() }
    else if (m.type === 'crop_changed') setCloudVersion((v) => v + 1)
    else if (m.type === 'table_calibrated') {
      setProgress(null)
      const t = m.table
      setNotice(`Table found: ${t.measured_size_m.map((v) => v.toFixed(2)).join(' × ')} m, lidar ${t.lidar_height_m.toFixed(2)} m above it. `
        + 'Set a new baseline.' + (t.crop ? ` ${t.crop}` : '')
        + (t.warnings.length ? ` Note: ${t.warnings.join(' ')}` : ''))
      refresh()
    }
    else if (m.type === 'event_updated') setEvents((evs) => evs.map((e) => (e.id === m.event.id ? { ...e, ...m.event } : e)))
    else if (m.type === 'alerts_changed') setStatus((s) => (s ? { ...s, text_alerts: m.text_alerts, text_bridge: m.text_bridge } : s))
    else if (m.type === 'perceptron_changed') setStatus((s) => (s ? { ...s, perceptron: m.perceptron } : s))
  })

  // Perceptron toggle (DH-28): while on, each rescan's changes are diagnosed by Perceptron; off = offline fallbacks.
  const perc = (status && status.perceptron) || {}
  const percOn = !!perc.enabled
  const togglePerceptron = () => run(async () => {
    const a = await api.setPerceptron(!percOn)
    setStatus((s) => ({ ...s, ...a }))
  })

  // Text toggle (DH-22): while on, each rescan's flagged changes go to Ron's phone through the alerts/ bridge.
  const textOn = !!(status && status.text_alerts)
  const bridgeUp = !!(status && status.text_bridge)
  const toggleText = () => run(async () => {
    const a = await api.setTextAlerts(!textOn)
    setStatus((s) => ({ ...s, ...a }))
  })

  const synthetic = status && status.source === 'synthetic'
  const busy = !!progress || (status && status.busy)
  const run = async (fn) => { setError(null); try { await fn() } catch (e) { setError(e.message) } }

  const counts = useMemo(() => {
    const c = { flagged: 0, routine: 0, dismissed: 0 }
    events.forEach((e) => { c[eventStatus(e)] += 1 })
    return c
  }, [events])
  const selected = events.find((e) => e.id === selectedId) || null
  const selectedIndex = events.findIndex((e) => e.id === selectedId)
  const scan = scans.find((s) => s.id === scanId)

  const onFeedback = async (id, fb) => {
    const ev = await api.feedback(id, fb)
    setEvents((evs) => evs.map((e) => (e.id === id ? { ...e, ...ev } : e)))
    setScans(await api.scans())
  }

  const recording = progress && progress.stage === 'accumulating' && progress.remaining != null
  const stageText = progress && (progress.stage === 'starting'
    ? ({ baseline: 'Capturing baseline', calibration: 'Calibrating table' }[progress.kind] || 'Starting scan')
    : recording ? `Recording lidar · ${Math.ceil(progress.remaining)} s left`
    : (STAGES[progress.stage] || progress.stage) + (progress.stage === 'classifying' && progress.detail ? ` ${progress.detail}` : ''))

  return (
    <div className="app">
      <header className="top">
        <div className="brand">
          <Mark />
          <span className="brand-name">Noxen <span className="brand-sys">Systems</span></span>
          <span className="brand-div" />
          <span className="eyebrow">Inspection console</span>
        </div>
        <div className="top-right">
          {synthetic && <span className="eyebrow chip">Synthetic</span>}
          <span className={`live ${connected ? 'on' : ''}`}><i />{connected ? 'Live' : 'Offline'}</span>
          {synthetic && (
            <select value={scenario} onChange={(e) => setScenario(e.target.value)} disabled={busy} aria-label="Scenario">
              {(status.scenarios || []).map((s) => <option key={s} value={s}>{SCENARIO_LABEL[s] || s}</option>)}
            </select>
          )}
          {status && !synthetic && (
            <button className="btn ghost" disabled={busy} title={status.table
              ? `Calibrated ${fmtTime(status.table.fitted_at)}: ${status.table.size_m.join(' × ')} m crop`
              : 'Find the diorama table and crop every scan to it'}
                    onClick={() => run(() => api.calibrateTable())}>{status.table ? 'Recalibrate table' : 'Calibrate table'}</button>
          )}
          {status && (
            <button className={`btn ghost text-toggle ${textOn ? 'active' : ''} ${textOn && !bridgeUp ? 'warn' : ''}`}
                    role="switch" aria-checked={textOn} onClick={toggleText}
                    title={textOn ? (bridgeUp ? 'Flagged changes from each rescan are texted to your phone'
                                              : 'Text alerts are on, but the alert bridge is not running (cd alerts && npm start)')
                                  : 'Turn on to text flagged changes from each rescan to your phone'}>
              <span className="track"><i /></span>Text alerts{textOn && !bridgeUp ? ' · bridge off' : ''}
            </button>
          )}
          {status && (
            <button className={`btn ghost text-toggle ${percOn ? 'active' : ''} ${percOn && !perc.configured ? 'warn' : ''}`}
                    role="switch" aria-checked={percOn} onClick={togglePerceptron}
                    title={percOn ? (perc.configured ? `Rescans are diagnosed by Perceptron (${perc.model})`
                                                      : 'Perceptron is on, but PERCEPTRON_API_KEY is not set: the fallbacks answer')
                                  : 'Turn on to diagnose each rescan\'s changes with Perceptron (uses API credit)'}>
              <span className="track"><i /></span>Perceptron{percOn && !perc.configured ? ' · no key' : ''}
            </button>
          )}
          {status && !synthetic && (
            <button className="btn ghost" disabled={busy} title="Draw the area every scan is cropped to, seen from above"
                    onClick={() => setCropping(true)}>Crop</button>
          )}
          {status && !synthetic && (
            <button className="btn ghost" disabled={busy} title="Show the system where the camera is, so the points get the camera's colour"
                    onClick={() => setCamCalib(true)}>{status.calibrated ? 'Recalibrate camera' : 'Calibrate camera'}</button>
          )}
          <button className="btn ghost" disabled={busy} onClick={() => run(() => api.baseline())}>Set baseline</button>
          <button className="btn primary" disabled={busy || !(status && status.baseline_id)}
                  onClick={() => run(() => api.rescan(synthetic ? scenario : undefined))}>Rescan</button>
        </div>
      </header>

      <div className={`progress ${progress ? 'on' : ''}`}>
        <div className={`bar ${recording ? 'det' : ''}`} style={recording ? { '--p': progress.detail || 0 } : undefined} />
        <span className="eyebrow">{stageText || ''}</span>
      </div>
      {error && <div className="error" role="alert">{error}</div>}
      {notice && <div className="notice" role="status">{notice}</div>}
      {cropping && (
        <CropEditor onClose={() => setCropping(false)} onSaved={(msg) => {
          setCropping(false); setNotice(msg); setCloudVersion((v) => v + 1); refresh()
        }} />
      )}
      {camCalib && (
        <CameraCalib calibrated={!!(status && status.calibrated)} onClose={() => setCamCalib(false)} onDone={(r) => {
          setCamCalib(false)
          setNotice((r.auto
            ? `Camera auto-aligned: lidar outlines now sit ${r.auto.cost_end} px from the photo's edges on average (was ${r.auto.cost_start}); `
              + `moved ${Math.round(r.auto.moved_m * 1000)} mm, turned ${r.auto.turned_deg}°. `
            : `Camera calibrated: the clicks fit to ${r.reproj_px} px, camera ${Math.round(r.camera_to_lidar_m * 100)} cm from the lidar. `)
            + 'The 3D view is now in colour.' + (r.warnings.length ? ` Note: ${r.warnings.join(' ')}` : ''))
          setCloudVersion((v) => v + 1)
          refresh()
        }} />
      )}

      <main className="grid">
        <section className="panel view-panel">
          <div className="summary">
            <div className="eyebrow">{scan ? `Tonight · ${fmtTime(scan.created_at)}${scan.stats && scan.stats.seconds ? ` · ${scan.stats.seconds}s` : ''}` : 'Tonight'}</div>
            {scan ? (
              <div className="headline">
                <span className="num-big flagged">{counts.flagged}</span><span>{counts.flagged === 1 ? 'issue' : 'issues'}</span>
                <span className="num-big routine">{counts.routine}</span><span>routine</span>
                {counts.dismissed > 0 && <><span className="num-big dismissed">{counts.dismissed}</span><span>dismissed</span></>}
              </div>
            ) : (
              <div className="headline muted">{status && status.baseline_id ? 'Baseline captured. Press Rescan.' : 'Set a baseline to begin.'}</div>
            )}
          </div>
          <Viewer3D scanId={scanId || (status && status.baseline_id)} baselineId={status && status.baseline_id}
                    events={scanId ? events : []} selectedId={selectedId} onSelect={setSelectedId} version={cloudVersion} />
          {events.length > 0 && (
            <ol className="event-list">
              {events.map((e, i) => (
                <li key={e.id} className={e.id === selectedId ? 'selected' : ''} onClick={() => setSelectedId(e.id)}>
                  <span className={`num ${eventStatus(e)}`}>{i + 1}</span>
                  <span className="ev-label">{e.classification ? e.classification.label : e.kind}</span>
                  {!e.bbox_map && <span className="eyebrow">not located</span>}
                  <span className="eyebrow">{eventStatus(e) === 'routine' ? 'routine' : e.classification ? e.classification.severity : ''}</span>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section className="panel">
          <EventDetail event={selected} index={selectedIndex} total={events.length} onFeedback={onFeedback} onDispatch={api.dispatch} />
        </section>

        <section className="panel history">
          <div className="eyebrow">Scan history</div>
          <ul>
            {scans.map((s) => (
              <li key={s.id} className={s.id === scanId ? 'selected' : ''}
                  onClick={() => !s.is_baseline && s.status === 'done' && setScanId(s.id)}>
                <span className="when">{fmtTime(s.created_at)}</span>
                {s.is_baseline ? <span className="eyebrow">Baseline</span>
                  : s.status === 'error' ? <span className="eyebrow err" title={s.error}>Failed</span>
                  : <span className="counts"><b className="flagged">{s.n_flagged}</b> issues · <b className="routine">{s.n_routine}</b> routine</span>}
              </li>
            ))}
            {scans.length === 0 && <li className="muted">No scans yet.</li>}
          </ul>
        </section>
      </main>
    </div>
  )
}
