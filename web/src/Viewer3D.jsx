import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { eventStatus } from './api.js'

// 3D lidar view of a scan (GET /scans/{id}/cloud: float32 rows x, y, z, intensity, label, ghost, rgb).
// Map frame is z-up; the camera keeps z as "up". Unchanged points take the camera's colour (Colour mode, when the
// camera is calibrated: rgb >= 0) or a quiet monochrome height gradient; points of a change take its status colour;
// "ghost" points (what disappeared: a moved object's old place) are drawn dim.
const STATUS = { flagged: new THREE.Color('#FF5A1F'), routine: new THREE.Color('#5EC2B0'), dismissed: new THREE.Color('#5A616B') }
const LOW = new THREE.Color('#252b34')
const HIGH = new THREE.Color('#d9dee5')
const GHOST = new THREE.Color('#3a414b')
const COLS = 7
const LIDAR = new THREE.Color('#3DDC84')       // the lidar's own position (Ron: a green dot)

function parseCloud(buf) {
  const f = new Float32Array(buf)
  const n = f.length / COLS
  return { n, f }
}

export default function Viewer3D({ scanId, baselineId, events, selectedId, onSelect, version = 0 }) {
  const mount = useRef(null)
  const ctx = useRef(null)                 // three.js objects that live as long as the component
  const [layer, setLayer] = useState('tonight')
  const [view, setView] = useState('perspective')
  const [colorMode, setColorMode] = useState('colour')
  const [hasColour, setHasColour] = useState(false)
  const [loading, setLoading] = useState(false)
  const [labels, setLabels] = useState([])  // [{id, n, x, y, status, visible}]

  // one-time scene setup
  useEffect(() => {
    const el = mount.current
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200)
    camera.up.set(0, 0, 1)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.5
    controls.maxPolarAngle = Math.PI * 0.49
    controls.minDistance = 0.2
    controls.maxDistance = 4
    const stopSpin = () => { controls.autoRotate = false }
    renderer.domElement.addEventListener('pointerdown', stopSpin)
    renderer.domElement.addEventListener('wheel', stopSpin, { passive: true })


    const group = new THREE.Group()            // cloud + boxes, rebuilt on data change
    scene.add(group)
    const focus = { from: null, to: null, t: 1 }
    let raf
    const tmp = new THREE.Vector3()
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false)
      camera.aspect = w / Math.max(h, 1)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    const tick = () => {
      raf = requestAnimationFrame(tick)
      if (focus.t < 1 && focus.to) {
        focus.t = Math.min(1, focus.t + 0.06)
        const e = 1 - Math.pow(1 - focus.t, 3)
        controls.target.lerpVectors(focus.from, focus.to, e)
      }
      controls.update()
      renderer.render(scene, camera)
      // project label anchors to screen space
      const w = el.clientWidth, h = el.clientHeight
      const anchors = ctx.current.anchors
      if (anchors.length) {
        const out = anchors.map((a) => {
          tmp.copy(a.pos).project(camera)
          return { ...a.meta, x: (tmp.x * 0.5 + 0.5) * w, y: (-tmp.y * 0.5 + 0.5) * h, visible: tmp.z < 1 }
        })
        ctx.current.setLabelsThrottled(out)
      }
    }
    let last = 0
    ctx.current = {
      renderer, scene, camera, controls, group, focus, anchors: [], grid: null, k: 1,
      setLabelsThrottled: (out) => { const now = performance.now(); if (now - last > 30) { last = now; setLabels(out) } },
    }
    tick()
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [])

  // camera presets (any deliberate choice ends the idle spin)
  useEffect(() => {
    const c = ctx.current
    if (!c) return
    if (c.touched) c.controls.autoRotate = false
    c.touched = true
    const target = c.controls.target.clone()
    const k = c.k                                // view scale: 1 = the 1 m diorama, larger for a room
    if (view === 'top') c.camera.position.set(target.x, target.y - 0.001 * k, target.z + 1.45 * k)
    else c.camera.position.set(target.x + 0.25 * k, target.y - 1.0 * k, target.z + 0.75 * k)
    c.controls.update()
  }, [view])

  // (re)build the cloud and the change boxes
  const cloudId = layer === 'baseline' ? baselineId : scanId
  useEffect(() => {
    const c = ctx.current
    if (!c || !cloudId) return
    let cancelled = false
    setLoading(true)
    let lidarPos = null
    fetch(`/scans/${cloudId}/cloud?v=${version}`).then((r) => {
      if (!r.ok) return Promise.reject(r.status)
      const l = (r.headers.get('X-Lidar') || '').split(',').map(Number)
      if (l.length === 3 && l.every(Number.isFinite)) lidarPos = l
      return r.arrayBuffer()
    }).then((buf) => {
      if (cancelled) return
      const { n, f } = parseCloud(buf)
      // clear previous
      c.group.traverse((o) => { o.geometry && o.geometry.dispose(); o.material && o.material.dispose() })
      c.group.clear()
      let zmin = Infinity, zmax = -Infinity
      let coloured = 0
      for (let i = 0; i < n; i++) {
        const z = f[i * COLS + 2]; if (z < zmin) zmin = z; if (z > zmax) zmax = z
        if (f[i * COLS + 6] >= 0) coloured++
      }
      setHasColour(coloured > 0)
      const useColour = colorMode === 'colour' && coloured > 0
      const pos = new Float32Array(n * 3)
      const col = new Float32Array(n * 3)
      const k = new THREE.Color()
      const showChanges = layer === 'tonight'
      for (let i = 0; i < n; i++) {
        const o = i * COLS
        pos[i * 3] = f[o]; pos[i * 3 + 1] = f[o + 1]; pos[i * 3 + 2] = f[o + 2]
        const label = f[o + 4], ghost = f[o + 5] > 0.5
        const ev = showChanges && label >= 0 ? events[label] : null
        if (ghost) {
          if (!showChanges) continue
          k.copy(GHOST)
        } else if (ev) {
          k.copy(STATUS[eventStatus(ev)])
        } else if (useColour && f[o + 6] >= 0) {
          const c = f[o + 6]
          k.setRGB(Math.floor(c / 65536) / 255, (Math.floor(c / 256) % 256) / 255, (c % 256) / 255, THREE.SRGBColorSpace)
        } else {
          const t = Math.pow((f[o + 2] - zmin) / Math.max(zmax - zmin, 1e-3), 0.8)
          k.copy(LOW).lerp(HIGH, t)
          k.multiplyScalar(0.85 + 0.3 * f[o + 3])
        }
        col[i * 3] = k.r; col[i * 3 + 1] = k.g; col[i * 3 + 2] = k.b
      }
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
      geo.computeBoundingBox()
      // everything sized for the 1 m diorama scales with the cloud, so a whole-room scan frames just as well
      const bb = geo.boundingBox
      const span = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, 0.5)
      const ks = Math.max(1, span / 1.06)
      const reframe = !c.framed || Math.abs(Math.log(ks / c.k)) > Math.log(1.5)
      c.k = ks
      const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.0075 * Math.min(ks, 6), vertexColors: true, sizeAttenuation: true }))
      c.group.add(points)
      if (lidarPos) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(0.018 * Math.min(ks, 6), 20, 14), new THREE.MeshBasicMaterial({ color: LIDAR }))
        dot.position.set(...lidarPos)
        c.group.add(dot)
        const halo = new THREE.Mesh(new THREE.SphereGeometry(0.032 * Math.min(ks, 6), 20, 14),
          new THREE.MeshBasicMaterial({ color: LIDAR, transparent: true, opacity: 0.18, depthWrite: false }))
        halo.position.copy(dot.position)
        c.group.add(halo)
      }

      const anchors = []
      if (showChanges) {
        events.forEach((ev, i) => {
          if (!ev.bbox_map) return
          const st = eventStatus(ev)
          const [cx, cy, cz] = ev.bbox_map.center
          const [sx, sy, sz] = ev.bbox_map.size.map((v) => Math.max(v, 0.02) + 0.016)
          const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, sz)),
            new THREE.LineBasicMaterial({ color: STATUS[st], transparent: true, opacity: 0.9 }))
          box.position.set(cx, cy, cz)
          box.userData = { id: ev.id }
          c.group.add(box)
          const fill = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz),
            new THREE.MeshBasicMaterial({ color: STATUS[st], transparent: true, opacity: 0.08, depthWrite: false }))
          fill.position.copy(box.position)
          fill.userData = { id: ev.id, fill: true }
          c.group.add(fill)
          if (ev.moved_from) {
            const [ox, oy] = ev.moved_from.center
            const ghost = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, sz)),
              new THREE.LineDashedMaterial({ color: 0x5a616b, dashSize: 0.012, gapSize: 0.008 }))
            ghost.position.set(ox, oy, cz)
            ghost.computeLineDistances()
            c.group.add(ghost)
            const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
              new THREE.Vector3(ox, oy, cz + sz / 2 + 0.01), new THREE.Vector3(cx, cy, cz + sz / 2 + 0.01)]),
              new THREE.LineDashedMaterial({ color: STATUS[st], dashSize: 0.015, gapSize: 0.01 }))
            line.computeLineDistances()
            c.group.add(line)
          }
          anchors.push({ pos: new THREE.Vector3(cx, cy, cz + sz / 2 + 0.035), meta: { id: ev.id, n: i + 1, status: st } })
        })
      }
      c.anchors = anchors
      if (!anchors.length) setLabels([])
      // frame the cloud on first load, and again when its scale changes (diorama <-> room)
      if (reframe) {
        const mid = new THREE.Vector3().addVectors(bb.min, bb.max).multiplyScalar(0.5)
        mid.z = bb.min.z + 0.1 * (bb.max.z - bb.min.z)
        c.controls.target.copy(mid)
        c.camera.position.set(mid.x + 0.25 * ks, mid.y - 1.0 * ks, mid.z + 0.77 * ks)
        c.controls.maxDistance = 4 * ks
        // floor grid for scale: 10 cm cells on the diorama, 50 cm on a room
        if (c.grid) { c.scene.remove(c.grid); c.grid.geometry.dispose(); c.grid.material.dispose() }
        const cell = ks < 2 ? 0.1 : 0.5
        const size = Math.ceil((2 * span) / cell) * cell
        c.grid = new THREE.GridHelper(size, Math.round(size / cell), 0x2a3039, 0x161a20)
        c.grid.rotation.x = Math.PI / 2
        c.grid.position.set(mid.x, mid.y, bb.min.z - 0.002)
        c.scene.add(c.grid)
        c.framed = true
      }
      setLoading(false)
    }).catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [cloudId, layer, events, colorMode, version])

  // emphasise the selected change and glide the orbit target to it
  useEffect(() => {
    const c = ctx.current
    if (!c) return
    c.group.children.forEach((o) => {
      if (!o.userData || !o.userData.id) return
      const on = o.userData.id === selectedId
      if (o.userData.fill) o.material.opacity = on ? 0.2 : 0.06
      else o.material.opacity = on ? 1 : 0.55
    })
    const ev = events.find((e) => e.id === selectedId)
    if (c.userSelected) c.controls.autoRotate = false        // the first, automatic selection does not count
    c.userSelected = true
    if (ev && ev.bbox_map && layer === 'tonight') {
      c.focus.from = c.controls.target.clone()
      c.focus.to = new THREE.Vector3(...ev.bbox_map.center)
      c.focus.t = 0
    }
  }, [selectedId, events, layer, loading])

  const stopSpin = () => { if (ctx.current) ctx.current.controls.autoRotate = false }

  return (
    <div className="viewer">
      <div className="viewer-bar">
        <div className="seg" role="tablist" aria-label="Cloud">
          <button className={layer === 'tonight' ? 'on' : ''} onClick={() => { stopSpin(); setLayer('tonight') }} disabled={!scanId}>Tonight</button>
          <button className={layer === 'baseline' ? 'on' : ''} onClick={() => { stopSpin(); setLayer('baseline') }} disabled={!baselineId}>Baseline</button>
        </div>
        <div className="seg" role="tablist" aria-label="Colour">
          <button className={colorMode === 'colour' && hasColour ? 'on' : ''} disabled={!hasColour}
                  title={hasColour ? 'Points take the camera\'s colour' : 'Calibrate the camera to colour the points'}
                  onClick={() => setColorMode('colour')}>Colour</button>
          <button className={colorMode === 'height' || !hasColour ? 'on' : ''} onClick={() => setColorMode('height')}>Height</button>
        </div>
        <div className="seg" role="tablist" aria-label="View">
          <button className={view === 'perspective' ? 'on' : ''} onClick={() => { stopSpin(); setView('perspective') }}>3D</button>
          <button className={view === 'top' ? 'on' : ''} onClick={() => { stopSpin(); setView('top') }}>Top</button>
        </div>
      </div>
      <div className="viewer-canvas" ref={mount}>
        {labels.filter((l) => l.visible).map((l) => (
          <button key={l.id} className={`tag3d ${l.status} ${l.id === selectedId ? 'selected' : ''}`}
                  style={{ transform: `translate(${l.x}px, ${l.y}px) translate(-50%, -100%)` }}
                  onClick={() => { stopSpin(); onSelect(l.id) }}>{l.n}</button>
        ))}
        {!cloudId && <div className="viewer-empty"><div className="eyebrow">No scan yet</div><div className="muted">Set a baseline to begin.</div></div>}
        {loading && <div className="viewer-loading eyebrow">Loading point cloud</div>}
      </div>
      <div className="viewer-hint">Drag to orbit · scroll to zoom · right-drag to pan</div>
    </div>
  )
}
