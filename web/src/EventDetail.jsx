import { useState } from 'react'
import { eventStatus } from './api.js'

const KIND = { moved: 'Moved', added: 'New object', removed: 'Missing object', appearance: 'Surface change' }
const SCORE = { moved: (v) => `moved ${v} cm`, added: (v) => `${v} cm across`, removed: (v) => `${v} cm across`,
                appearance: (v) => `ΔE ${v}` }
const STATUS_TEXT = { flagged: 'Issue', routine: 'Routine', dismissed: 'Dismissed' }

function Crop({ url, label }) {
  return (
    <figure className="crop">
      {url ? <img src={url} alt={label} /> : <div className="crop-missing">No image</div>}
      <figcaption className="eyebrow">{label}</figcaption>
    </figure>
  )
}

export default function EventDetail({ event, index, total, onFeedback, onDispatch }) {
  const [note, setNote] = useState(null)
  if (!event) {
    return <div className="detail empty"><div className="eyebrow">No change selected</div><div className="muted">Pick a change in the 3D view or the list.</div></div>
  }
  const c = event.classification
  const st = eventStatus(event)
  const attempts = event.classification_attempts || []
  const skipped = attempts.filter((a) => a.result !== 'ok')
  return (
    <div className="detail">
      <div className="eyebrow">Change {index + 1} / {total} · {KIND[event.kind] || event.kind}{event.geo_score != null && SCORE[event.kind] ? ` · ${SCORE[event.kind](event.geo_score)}` : ''}</div>
      <div className="title-row">
        <h2 className="label">{c ? c.label : 'Unclassified'}</h2>
        <span className={`status-chip ${st}`}>{STATUS_TEXT[st]}</span>
      </div>
      {c && (
        <div className="meta-row">
          <span className={`sev sev-${c.severity}`}>{c.severity === 'none' ? 'No action' : `${c.severity} severity`}</span>
          <span className="eyebrow provider" title={skipped.map((a) => `${a.provider}: ${a.result}${a.detail ? ' (' + a.detail + ')' : ''}`).join('\n')}>
            {c.provider}{skipped.length ? ` · ${skipped.length} fallback${skipped.length > 1 ? 's' : ''}` : ''}
          </span>
          <span className="eyebrow">{Math.round(c.confidence * 100)}% conf.</span>
        </div>
      )}
      {c && <p className="rationale">{c.rationale}</p>}
      <div className="crops">
        <Crop url={event.crop_before_url} label="Baseline" />
        <Crop url={event.crop_after_url} label="Tonight" />
      </div>
      {!event.crop_located && <div className="muted small">Camera not calibrated yet: crops show the whole frame.</div>}
      <div className="actions">
        <button className={`btn ghost ${event.feedback === 'false_alarm' ? 'active' : ''}`} onClick={() => onFeedback(event.id, event.feedback === 'false_alarm' ? 'none' : 'false_alarm')}>False alarm</button>
        <button className={`btn ghost ${event.feedback === 'confirmed' ? 'active' : ''}`} onClick={() => onFeedback(event.id, event.feedback === 'confirmed' ? 'none' : 'confirmed')}>Confirm issue</button>
        <button className="btn primary" disabled={!event.bbox_map}
                onClick={async () => { const r = await onDispatch(event.id); setNote(r && r.status) }}>Send Scout</button>
      </div>
      {note && <div className="note">{note}</div>}
      <div className="muted tiny">Every verdict is stored with its scan. It trains the next model.</div>
    </div>
  )
}
