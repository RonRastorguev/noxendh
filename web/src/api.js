// Thin client for the FastAPI server (ARCHITECTURE §7.1). Relative URLs: Vite proxies them in dev, FastAPI serves
// the built app in production.
async function req(method, url, body) {
  const r = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail || detail } catch { /* not JSON */ }
    const err = new Error(detail)
    err.status = r.status
    throw err
  }
  return r.status === 204 ? null : r.json()
}

export const api = {
  status: () => req('GET', '/status'),
  scans: () => req('GET', '/scans'),
  events: (scanId) => req('GET', `/scans/${scanId}/events`),
  map: () => req('GET', '/map'),
  baseline: (scenario) => req('POST', '/baseline', scenario ? { scenario } : {}),
  rescan: (scenario) => req('POST', '/scans/trigger', scenario ? { scenario } : {}),
  calibrateTable: () => req('POST', '/calibrate/table'),
  cameraTarget: () => req('GET', '/calibrate/camera/target'),
  cropState: () => req('GET', '/crop'),
  cropPreview: async (scanId) => {
    const r = await fetch(scanId ? `/crop/preview?scan_id=${encodeURIComponent(scanId)}` : '/crop/preview')
    if (!r.ok) {
      let detail = r.statusText
      try { detail = (await r.json()).detail || detail } catch { /* not JSON */ }
      throw new Error(detail)
    }
    const f = new Float32Array(await r.arrayBuffer())
    const lidar = (r.headers.get('X-Lidar') || '').split(',').map(Number)
    return { f, n: f.length / 4, lidar: lidar.length === 3 ? lidar : null, scanId: r.headers.get('X-Scan-Id') }
  },
  setCrop: (box) => req('POST', '/crop', box),
  resetCrop: () => req('DELETE', '/crop'),
  calibrateCamera: (scanId, clicks) => req('POST', '/calibrate/camera', { scan_id: scanId, ...clicks }),   // {corners} or {pairs}
  previewCamera: (scanId, clicks) => req('POST', '/calibrate/camera/preview', { scan_id: scanId, ...clicks }),
  autoAlignCamera: () => req('POST', '/calibrate/camera/auto'),
  feedback: (id, feedback) => req('POST', `/events/${id}/feedback`, { feedback }),
  dispatch: (id) => req('POST', '/scout/dispatch', { event_id: id }),
  setTextAlerts: (text) => req('POST', '/alerts', { text }),
  setPerceptron: (enabled) => req('POST', '/perceptron', { enabled }),
}

// Status of an event for colour and counts: flagged (coral), routine (teal), dismissed (grey).
export function eventStatus(ev) {
  if (ev.feedback === 'false_alarm') return 'dismissed'
  if (ev.classification && ev.classification.routine) return 'routine'
  return 'flagged'
}
