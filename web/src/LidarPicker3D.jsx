import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// 3D point picker for the camera calibration's point pairs (DH-29). It opens from the lidar's own position looking
// at `target` (the crop's centre), so it looks like the camera's photo; drag orbits, scroll zooms, right-drag pans.
// A click (not a drag) picks the point you are LOOKING AT: the nearest point along the ray, not the top of a stack.
const LIDAR = new THREE.Color('#3DDC84')
const LOW = new THREE.Color('#3a414b')
const HIGH = new THREE.Color('#eef1f4')

export default function LidarPicker3D({ cloud, target, pairs, colorFor, active, onPick, height }) {
  const mount = useRef(null)
  const ctx = useRef(null)
  const [labels, setLabels] = useState([])

  useEffect(() => {
    const el = mount.current
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(72, 1, 0.01, 50)     // ~ the DECXIN's vertical field of view
    camera.up.set(0, 0, 1)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.1
    const markers = new THREE.Group()
    scene.add(markers)
    const hover = new THREE.Mesh(new THREE.SphereGeometry(0.012, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7, depthTest: false }))
    hover.visible = false
    hover.renderOrder = 2
    scene.add(hover)
    const raycaster = new THREE.Raycaster()
    const tmp = new THREE.Vector3()
    let raf
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h, false)
      camera.aspect = w / Math.max(h, 1)
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()
    let lastLabels = 0
    const tick = () => {
      raf = requestAnimationFrame(tick)
      controls.update()
      const ld = ctx.current.lidarDot
      if (ld) ld.visible = camera.position.distanceTo(ld.position) > 0.4      // hidden while looking from the lidar
      renderer.render(scene, camera)
      const now = performance.now()
      if (now - lastLabels > 40) {
        lastLabels = now
        const w = el.clientWidth, h = el.clientHeight
        setLabels(ctx.current.anchors.map((a) => {
          tmp.copy(a.pos).project(camera)
          return { ...a, x: (tmp.x * 0.5 + 0.5) * w, y: (-tmp.y * 0.5 + 0.5) * h, visible: tmp.z < 1 }
        }))
      }
    }
    ctx.current = { renderer, scene, camera, controls, markers, hover, raycaster, points: null, anchors: [], lidarDot: null }
    tick()
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      el.removeChild(renderer.domElement)
    }
  }, [])

  // the cloud, and the starting view: from the lidar, looking at the target
  useEffect(() => {
    const c = ctx.current
    if (!c || !cloud) return
    if (c.points) { c.scene.remove(c.points); c.points.geometry.dispose(); c.points.material.dispose() }
    const n = cloud.n
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3)
    let zlo = Infinity, zhi = -Infinity
    for (let i = 0; i < n; i++) { const z = cloud.f[i * 4 + 2]; if (z < zlo) zlo = z; if (z > zhi) zhi = z }
    const k = new THREE.Color()
    for (let i = 0; i < n; i++) {
      pos[i * 3] = cloud.f[i * 4]; pos[i * 3 + 1] = cloud.f[i * 4 + 1]; pos[i * 3 + 2] = cloud.f[i * 4 + 2]
      const t = Math.pow((cloud.f[i * 4 + 2] - zlo) / Math.max(zhi - zlo, 1e-3), 0.7)
      k.copy(LOW).lerp(HIGH, t).multiplyScalar(0.75 + 0.35 * cloud.f[i * 4 + 3])
      col[i * 3] = k.r; col[i * 3 + 1] = k.g; col[i * 3 + 2] = k.b
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3))
    c.points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.012, vertexColors: true, sizeAttenuation: true }))
    c.scene.add(c.points)
    if (c.lidarDot) { c.scene.remove(c.lidarDot); c.lidarDot = null }
    if (cloud.lidar) {
      c.lidarDot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12), new THREE.MeshBasicMaterial({ color: LIDAR }))
      c.lidarDot.position.set(...cloud.lidar)
      c.scene.add(c.lidarDot)
    }
    const look = new THREE.Vector3(...(target || [0, 0, 0]))
    const from = cloud.lidar ? new THREE.Vector3(...cloud.lidar) : look.clone().add(new THREE.Vector3(0, -1, 0.6))
    const back = from.clone().sub(look).setZ(0)
    if (back.lengthSq() < 1e-6) back.set(0, -1, 0)
    back.normalize()
    c.camera.position.copy(from).addScaledVector(back, 0.05).add(new THREE.Vector3(0, 0, 0.03))
    c.controls.target.copy(look)
    c.controls.update()
  }, [cloud, target])

  // pair markers
  useEffect(() => {
    const c = ctx.current
    if (!c) return
    c.markers.children.forEach((m) => { m.geometry.dispose(); m.material.dispose() })
    c.markers.clear()
    const anchors = []
    pairs.forEach((p, i) => {
      if (!p.xyz) return
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.014, 16, 12),
        new THREE.MeshBasicMaterial({ color: colorFor(i), depthTest: false }))
      m.renderOrder = 3
      m.position.set(...p.xyz)
      c.markers.add(m)
      anchors.push({ n: i + 1, color: colorFor(i), pos: m.position.clone() })
    })
    c.anchors = anchors
  }, [pairs, colorFor])

  // picking: the nearest point along the ray (what you see), among those close to the ray
  const pickAt = (e) => {
    const c = ctx.current
    if (!c || !c.points) return null
    const r = c.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    c.raycaster.setFromCamera(ndc, c.camera)
    const dist = c.camera.position.distanceTo(c.controls.target)
    c.raycaster.params.Points.threshold = Math.max(0.004, dist * 0.012)
    const hits = c.raycaster.intersectObject(c.points)
    if (!hits.length) return null
    const d0 = hits[0].distance
    const near = hits.filter((h) => h.distance < d0 + 0.03)
    const best = near.reduce((a, b) => (b.distanceToRay < a.distanceToRay ? b : a))
    return [best.point.x, best.point.y, best.point.z].map((v, j) => cloud.f[best.index * 4 + j] ?? v)
  }
  const down = useRef(null)
  const onPointerDown = (e) => { down.current = { x: e.clientX, y: e.clientY, b: e.button } }
  const onPointerUp = (e) => {
    const d = down.current
    down.current = null
    if (!d || d.b !== 0 || !active) return
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return              // that was an orbit drag
    const p = pickAt(e)
    if (p) onPick(p)
  }
  const lastMove = useRef(0)
  const onPointerMove = (e) => {
    const c = ctx.current
    if (!c || down.current) return
    const now = performance.now()
    if (now - lastMove.current < 50) return
    lastMove.current = now
    const p = active ? pickAt(e) : null
    c.hover.visible = !!p
    if (p) c.hover.position.set(...p)
  }

  return (
    <div className="picker3d" style={{ height }}>
      <div ref={mount} className="picker3d-canvas" onPointerDown={onPointerDown} onPointerUp={onPointerUp}
           onPointerMove={onPointerMove} onPointerLeave={() => { if (ctx.current) ctx.current.hover.visible = false }}
           style={{ cursor: active ? 'crosshair' : 'grab' }} />
      {labels.filter((l) => l.visible).map((l) => (
        <span key={l.n} className="picker3d-tag" style={{ transform: `translate(${l.x}px, ${l.y}px) translate(-50%, -140%)`, background: l.color }}>{l.n}</span>
      ))}
      {!cloud && <div className="viewer-loading eyebrow">Loading lidar</div>}
    </div>
  )
}
