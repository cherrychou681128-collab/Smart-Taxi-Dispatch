import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  useMap,
  useMapEvents,
  CircleMarker,
  Popup,
  Tooltip,
} from 'react-leaflet'
import L from 'leaflet'
import { useMemo, useEffect, useRef, useState, useCallback } from 'react'
import { taxiIcon, passengerIcon, dropoffIcon } from '../mapIcons'
import { t } from '../i18n'
import { apiFetch as defaultApiFetch } from '../apiBase.js'

// ====== 設定 ======
const DEFAULT_CENTER = [40.758, -73.9855]
const DEFAULT_ZOOM = 11
const STEP_MS = 60
const HEADER_OFFSET_PX = 64

const FALLBACK_SPEED_KPH = 30
const ICON_HEADING_OFFSET_DEG = 0

const ORDER_START_PREFIX = 'orderStart:'
const ORDER_ROUTE_PREFIX = 'orderRoute:'
const SUMO_TRACE_URL = import.meta.env?.VITE_SUMO_TRACE_URL || '/sumo_traces/demo.json'

// ====== 物理參數（fallback physics sim 用） ======
const CAR_ACCEL = 3.5
const CAR_DECEL = 4.0
const INITIAL_V = 2.0
const ARRIVAL_THRESHOLD_METERS = 15.0

const HEADING_LOOKAHEAD_METERS = 18
const HEADING_SMOOTH_TAU_MS = 120

const ACTIVE_STATUS_SET = new Set([
  'assigned', 'accepted', 'en_route', 'enroute',
  'picked_up', 'in_progress', 'on_trip', 'ongoing',
])

// ====== ✅ Playback Factor Shared Sync (driver/passenger 即時同步) ======
const PLAYBACK_LS_KEY = 'simPlaybackFactor'
const PLAYBACK_EVT = 'simPlaybackFactorChanged'

function readPlaybackFactor() {
  try {
    const v = Number(localStorage.getItem(PLAYBACK_LS_KEY) || '1')
    return Number.isFinite(v) && v > 0 ? v : 1
  } catch {
    return 1
  }
}

function writePlaybackFactor(v) {
  const x = Number(v)
  const val = Number.isFinite(x) && x > 0 ? x : 1
  try {
    localStorage.setItem(PLAYBACK_LS_KEY, String(val))
  } catch {}
  try {
    window.dispatchEvent(new CustomEvent(PLAYBACK_EVT, { detail: { value: val } }))
  } catch {}
  return val
}

function usePlaybackFactorSync() {
  const [factor, setFactor] = useState(() => readPlaybackFactor())

  useEffect(() => {
    const onStorage = (e) => {
      if (e?.key !== PLAYBACK_LS_KEY) return
      const v = readPlaybackFactor()
      setFactor(v)
    }
    const onCustom = (e) => {
      const v = Number(e?.detail?.value)
      if (Number.isFinite(v) && v > 0) setFactor(v)
      else setFactor(readPlaybackFactor())
    }
    window.addEventListener('storage', onStorage)
    window.addEventListener(PLAYBACK_EVT, onCustom)
    return () => {
      window.removeEventListener('storage', onStorage)
      window.removeEventListener(PLAYBACK_EVT, onCustom)
    }
  }, [])

  const update = useCallback((v) => {
    const val = writePlaybackFactor(v)
    setFactor(val)
  }, [])

  return [factor, update]
}

// ====== Utils ======
function sameId(a, b) {
  const A = Number(a)
  const B = Number(b)
  return Number.isFinite(A) && Number.isFinite(B) && A === B
}
function isActiveStatus(status) {
  return ACTIVE_STATUS_SET.has(String(status || '').toLowerCase())
}
function getOrderDriverId(order) {
  return order?.driverId ?? order?.assignedDriverId ?? order?.driver_id ?? null
}
function getOrderKey(order) {
  const id = Number(order?.id)
  if (!Number.isFinite(id)) return null
  const createdAt = order?.createdAt || order?.created_at || ''
  return `sim_order_${id}_${createdAt}`
}
function canShowPassengerRoute(order) {
  const driverId = getOrderDriverId(order)
  if (driverId == null) return false

  const s = String(order?.status || '').toLowerCase()
  return [
    'assigned',
    'accepted',
    'en_route',
    'enroute',
    'picked_up',
    'in_progress',
    'on_trip',
    'ongoing',
  ].includes(s)
}
function canShowPassengerPlannedRoute(order) {
  const pickup = order?.pickupLocation
  const dropoff = order?.dropoffLocation

  return Boolean(
    pickup &&
    dropoff &&
    Number.isFinite(Number(pickup.lat)) &&
    Number.isFinite(Number(pickup.lng)) &&
    Number.isFinite(Number(dropoff.lat)) &&
    Number.isFinite(Number(dropoff.lng))
  )
}

// ====== Order Start LocalStorage ======
function readOrderStart(orderKey) {
  try {
    if (!orderKey) return null
    const raw = localStorage.getItem(`${ORDER_START_PREFIX}${orderKey}`)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}
function writeOrderStart(orderKey, latlng) {
  if (!orderKey || !latlng) return
  try {
    const k = `${ORDER_START_PREFIX}${orderKey}`
    localStorage.setItem(k, JSON.stringify(latlng))
  } catch {}
}
function writeOrderStartOnce(orderKey, latlng) {
  if (!orderKey || !latlng) return
  try {
    const k = `${ORDER_START_PREFIX}${orderKey}`
    if (localStorage.getItem(k)) return
    localStorage.setItem(k, JSON.stringify(latlng))
  } catch {}
}

function readOrderRoute(orderKey) {
  try {
    if (!orderKey) return null
    const raw = localStorage.getItem(`${ORDER_ROUTE_PREFIX}${orderKey}`)
    const parsed = raw ? JSON.parse(raw) : null
    if (!Array.isArray(parsed) || parsed.length < 2) return null
    const coords = parsed
      .map(p => {
        if (Array.isArray(p) && p.length >= 2) return [Number(p[0]), Number(p[1])]
        if (p && Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lng))) {
          return [Number(p.lat), Number(p.lng)]
        }
        return null
      })
      .filter(Boolean)
    return coords.length >= 2 ? coords : null
  } catch {
    return null
  }
}
function writeOrderRoute(orderKey, coords) {
  try {
    if (!orderKey || !Array.isArray(coords) || coords.length < 2) return
    const normalized = coords
      .map(p => {
        if (Array.isArray(p) && p.length >= 2) return [Number(p[0]), Number(p[1])]
        if (p && Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lng))) {
          return [Number(p.lat), Number(p.lng)]
        }
        return null
      })
      .filter(Boolean)
    if (normalized.length < 2) return
    localStorage.setItem(`${ORDER_ROUTE_PREFIX}${orderKey}`, JSON.stringify(normalized))
  } catch {}
}

// ====== Driver Click Handler ======
function DriverClickHandler({ enabled, driverId, onLocationChange, apiFetch }) {
  useMapEvents({
    click(e) {
      if (!enabled || driverId == null || typeof onLocationChange !== 'function') return
      const { lat, lng } = e.latlng

      onLocationChange({ id: driverId, lat, lng })

      const f = apiFetch || defaultApiFetch
      f(`/api/drivers/${driverId}/location`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat, lng }),
      }).catch(() => {})

      try { localStorage.setItem(`driverLoc:${driverId}`, JSON.stringify({ lat, lng })) } catch {}
    },
  })
  return null
}

// ====== Resize Fixer ======
function MapSizeFixer({ deps = [] }) {
  const map = useMap()
  useEffect(() => {
    const run = () => { try { map.invalidateSize(true) } catch {} }
    const timer = requestAnimationFrame(() => {
      run()
      requestAnimationFrame(run)
    })
    return () => cancelAnimationFrame(timer)
  }, deps)
  return null
}

// ====== Map State Persistence ======
function mapStateKey({ mode, driverId, previewEnabled }) {
  return `mapState:${mode}:${driverId ?? 'na'}:${previewEnabled ? 'p1' : 'p0'}`
}
function writeMapState(key, val) {
  try { localStorage.setItem(key, JSON.stringify(val)) } catch {}
}
function readMapState(key) {
  try {
    const raw = localStorage.getItem(key)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}
function MapStateTracker({ storageKey, disabled = false, minIntervalMs = 800 }) {
  const map = useMap()
  const lastWriteRef = useRef(0)

  const tryWrite = useCallback(() => {
    if (disabled) return
    const now = Date.now()
    if (now - lastWriteRef.current < minIntervalMs) return
    const c = map.getCenter()
    writeMapState(storageKey, { lat: c.lat, lng: c.lng, zoom: map.getZoom() })
    lastWriteRef.current = now
  }, [disabled, map, minIntervalMs, storageKey])

  useMapEvents({ moveend: tryWrite, zoomend: tryWrite })
  return null
}

function MapViewInitializer({ storageKey, streetViewMode, getInitialTarget }) {
  const map = useMap()

  useEffect(() => {
    try {
      if (streetViewMode) {
        const target = getInitialTarget?.()
        if (
          target &&
          Number.isFinite(Number(target.lat)) &&
          Number.isFinite(Number(target.lng))
        ) {
          map.setView([target.lat, target.lng], 18, { animate: false })
        } else {
          map.setView(DEFAULT_CENTER, 18, { animate: false })
        }
        return
      }

      const s = readMapState(storageKey)
      if (s && Number.isFinite(s.lat) && Number.isFinite(s.lng) && Number.isFinite(s.zoom)) {
        map.setView([s.lat, s.lng], s.zoom, { animate: false })
      } else {
        map.setView(DEFAULT_CENTER, DEFAULT_ZOOM, { animate: false })
      }
    } catch {}
  }, [storageKey, streetViewMode, map, getInitialTarget])

  return null
}

// ====== OSRM ======
async function fetchOsrmRoute(points, { signal } = {}) {
  if (!Array.isArray(points) || points.length < 2) return null
  const coordStr = points.map(p => `${p.lng},${p.lat}`).join(';')
  const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error('OSRM error')
  const data = await res.json()
  const coords = data?.routes?.[0]?.geometry?.coordinates
  if (!coords || coords.length < 2) throw new Error('OSRM no route')
  return coords.map(([lng, lat]) => [lat, lng])
}

// ====== Waypoints Builder ======
function isSameLL(a, b, eps = 1e-6) {
  if (!a || !b) return false
  return Math.abs(a.lat - b.lat) < eps && Math.abs(a.lng - b.lng) < eps
}

function buildCarWaypoints(order, mode, drivers, currentDriverId, frozenStartPos = null) {
  const pickup = order?.pickupLocation
  const dropoff = order?.dropoffLocation
  const stops = Array.isArray(order?.stops) ? order.stops : []
  if (!pickup || !dropoff) return null

  const active = isActiveStatus(order.status)
  const driverId = getOrderDriverId(order)

  const visibleToThisView =
    mode === 'passenger' ||
    (mode === 'driver' &&
      driverId != null &&
      currentDriverId != null &&
      sameId(driverId, currentDriverId))

  const waypoints = []

  if (
    frozenStartPos &&
    Number.isFinite(Number(frozenStartPos.lat)) &&
    Number.isFinite(Number(frozenStartPos.lng))
  ) {
    waypoints.push({
      lat: Number(frozenStartPos.lat),
      lng: Number(frozenStartPos.lng),
    })
  } else if (active && visibleToThisView && driverId != null) {
    const liveDriver = drivers.find(x => sameId(x.id, driverId))
    if (
      liveDriver &&
      Number.isFinite(Number(liveDriver.lat)) &&
      Number.isFinite(Number(liveDriver.lng))
    ) {
      waypoints.push({
        lat: Number(liveDriver.lat),
        lng: Number(liveDriver.lng),
      })
    } else {
      waypoints.push({ lat: pickup.lat, lng: pickup.lng })
    }
  } else {
    return null
  }

  if (!isSameLL(waypoints[waypoints.length - 1], pickup)) {
    waypoints.push({ lat: pickup.lat, lng: pickup.lng })
  }

  for (const s of stops) {
    const lat = Number(s?.lat)
    const lng = Number(s?.lng)
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue

    const type = String(s?.type || '').toLowerCase()
    if (type === 'driver' || type === 'pickup') continue

    const p = { lat, lng }
    if (isSameLL(p, pickup) || isSameLL(p, dropoff)) continue
    if (!isSameLL(waypoints[waypoints.length - 1], p)) waypoints.push(p)
  }

  if (!isSameLL(waypoints[waypoints.length - 1], dropoff)) {
    waypoints.push({ lat: dropoff.lat, lng: dropoff.lng })
  }

  return waypoints
}

function buildPlannedWaypoints(order) {
  const pickup = order?.pickupLocation
  const dropoff = order?.dropoffLocation
  const stops = Array.isArray(order?.stops) ? order.stops : []

  if (!pickup || !dropoff) return null

  const waypoints = [
    { lat: Number(pickup.lat), lng: Number(pickup.lng) }
  ]

  for (const s of stops) {
    const lat = Number(s?.lat)
    const lng = Number(s?.lng ?? s?.lon)
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue

    const type = String(s?.type || '').toLowerCase()
    if (type === 'driver' || type === 'pickup') continue

    waypoints.push({ lat, lng })
  }

  waypoints.push({
    lat: Number(dropoff.lat),
    lng: Number(dropoff.lng),
  })

  return waypoints
}

// ====== Simulation Logic ======
function simKey(k) { return `sim:${k}` }
function readSim(k) { try { return JSON.parse(localStorage.getItem(simKey(k))) } catch { return null } }
function writeSim(k, o) { try { localStorage.setItem(simKey(k), JSON.stringify(o)) } catch {} }

function resetSim(k) {
  const init = { elapsedMs: 0, startedAt: Date.now(), running: true, stepMs: STEP_MS, completed: false }
  writeSim(k, init)
  return init
}
function ensureSim(k) {
  const c = readSim(k)
  if (c) return c
  return resetSim(k)
}
function pauseSim(k) {
  const c = readSim(k)
  if (!c || !c.running || c.completed) return
  writeSim(k, {
    ...c,
    elapsedMs: (c.elapsedMs || 0) + Math.max(0, Date.now() - c.startedAt),
    running: false,
    startedAt: Date.now(),
  })
}
function resumeSim(k) {
  const c = readSim(k)
  if (!c || c.running || c.completed) return
  writeSim(k, { ...c, running: true, startedAt: Date.now() })
}
function computeElapsedMs(k) {
  const now = Date.now()
  const c = ensureSim(k)
  const total = (c.elapsedMs || 0) + (c.running ? Math.max(0, now - (c.startedAt || now)) : 0)
  return { elapsedMs: total, sim: c }
}
function completeSim(k) {
  const c = readSim(k) || {}
  writeSim(k, { ...c, running: false, completed: true, startedAt: Date.now() })
}

// ====== Math & Geo Utils ======
function toRad(d) { return (d * Math.PI) / 180 }
function normDeg(d) { return ((d % 360) + 360) % 360 }
function smoothLerpFactor(dtMs, tauMs) { return 1 - Math.exp(-Math.max(0, dtMs) / Math.max(1, tauMs)) }
function haversineMeters(lat1, lng1, lat2, lng2) {
  const R = 6371000
  const dLat = toRad(lat2 - lat1)
  const dLng = toRad(lng2 - lng1)
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)))
}
function buildCumDist(coords) {
  if (!coords || coords.length < 2) return null
  const cum = new Array(coords.length).fill(0)
  for (let i = 1; i < coords.length; i++) {
    cum[i] = cum[i - 1] + haversineMeters(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
  }
  return cum
}
function positionAtDistance(coords, cum, d) {
  if (!coords || !cum) return null
  const total = cum[cum.length - 1]
  const dist = Math.max(0, Math.min(d, total))
  let i = 0
  while (i < cum.length - 2 && cum[i + 1] < dist) i++
  const seg = (cum[i + 1] - cum[i])
  const ratio = seg > 1e-9 ? ((dist - cum[i]) / seg) : 0
  const [lat0, lng0] = coords[i]
  const [lat1, lng1] = coords[i + 1]
  return { lat: lat0 + (lat1 - lat0) * ratio, lng: lng0 + (lng1 - lng0) * ratio }
}
function computeHeadingDeg(from, to) {
  if (!from || !to) return null
  const lat1 = toRad(from.lat)
  const lat2 = toRad(to.lat)
  const dLng = toRad(to.lng - from.lng)
  const y = Math.sin(dLng) * Math.cos(lat2)
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng)
  return (Math.atan2(y, x) * (180 / Math.PI) + 360) % 360
}

// ====== SUMO Utils ======
function resolveSumoVehicleId(json, order) {
  const v = json?.vehicles
  if (!v) return null

  const ids = Object.keys(v)
  if (!ids.length) return null

  const driverId = getOrderDriverId(order)

  const candidates = [
    order?.sumoVehicleId,
    order?.sumo_vehicle_id,
    order?.vehicleId,
    order?.vehicle_id,
    order?.taxiId,
    order?.taxi_id,
    driverId != null ? String(driverId) : null,
    driverId != null ? `taxi_${driverId}` : null,
    driverId != null ? `veh_${driverId}` : null,
    driverId != null ? `vehicle_${driverId}` : null,
    driverId != null ? `driver_${driverId}` : null,
  ]
    .map(x => (x == null ? null : String(x).trim()))
    .filter(Boolean)

  for (const id of candidates) {
    if (v[id]?.points?.length) return id
  }

  // 只有 trace 裡本來就只有一台車時，才安全地直接使用
  if (ids.length === 1 && v[ids[0]]?.points?.length) {
    return ids[0]
  }

  return null
}

function pickVehiclePoints(json, { order } = {}) {
  const v = json?.vehicles
  if (!v) return null

  const resolvedId = resolveSumoVehicleId(json, order)
  if (!resolvedId) return null

  const pts = v[resolvedId]?.points
  if (!Array.isArray(pts) || pts.length < 2) return null

  return [...pts].sort((a, b) => Number(a.t) - Number(b.t))
}

function buildSpeedProfile(points) {
  const times = []
  const speeds = []
  for (const p of points) {
    times.push(Number(p.t))
    speeds.push(Math.max(0, Number(p.speedMps ?? p.speed_mps)))
  }
  if (times.length < 2) return null

  const prefix = new Array(times.length).fill(0)
  for (let i = 1; i < times.length; i++) {
    const dt = times[i] - times[i - 1]
    prefix[i] = prefix[i - 1] + (speeds[i - 1] + speeds[i]) * 0.5 * dt
  }
  const totalTime = times[times.length - 1]
  const totalDist = prefix[prefix.length - 1]
  const startT = times[0]

  function distAt(t) {
    if (t <= startT) return 0
    if (t >= totalTime) return totalDist
    let l = 0
    let h = times.length - 1
    while (l + 1 < h) {
      const m = (l + h) >> 1
      times[m] <= t ? l = m : h = m
    }
    const i = l
    const ratio = (t - times[i]) / (times[i + 1] - times[i])
    const vAt = speeds[i] + (speeds[i + 1] - speeds[i]) * ratio
    return prefix[i] + (speeds[i] + vAt) * 0.5 * (t - times[i])
  }
  function speedAt(t) {
    if (t <= startT) return speeds[0]
    if (t >= totalTime) return speeds[speeds.length - 1]
    let l = 0
    let h = times.length - 1
    while (l + 1 < h) {
      const m = (l + h) >> 1
      times[m] <= t ? l = m : h = m
    }
    const ratio = (t - times[l]) / (times[l + 1] - times[l])
    return speeds[l] + (speeds[l + 1] - speeds[l]) * ratio
  }
  function angleAt(t) {
    let best = null
    let minD = Infinity
    for (const p of points) {
      const d = Math.abs(p.t - t)
      if (d < minD) {
        minD = d
        best = p
      }
    }
    return best ? Number(best.angle ?? best.heading) : null
  }

  return { totalTime, totalDist, startT, distAt, speedAt, angleAt }
}

// ====== Icons ======
function makeTaxiIcon() {
  const svg = `<svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 8C12 5.79086 13.7909 4 16 4H28C30.2091 4 32 5.79086 32 8V36C32 38.2091 30.2091 40 28 40H16C13.7909 40 12 38.2091 12 36V8Z" fill="black" fill-opacity="0.3" transform="translate(2, 2)"/><path d="M12 8C12 5.79086 13.7909 4 16 4H28C30.2091 4 32 5.79086 32 8V36C32 38.2091 30.2091 40 28 40H16C13.7909 40 12 38.2091 12 36V8Z" fill="#F4C430" stroke="#E6B800" stroke-width="1"/><path d="M14 10H30V16H14V10Z" fill="#333"/><path d="M14 30H30V34H14V30Z" fill="#333"/><rect x="18" y="20" width="8" height="4" rx="1" fill="#FFD700" stroke="#D4AF37" stroke-width="0.5"/><path d="M13 5H15V6H13V5Z" fill="#FFF" /><path d="M29 5H31V6H29V5Z" fill="#FFF" /><path d="M13 38H15V39H13V38Z" fill="#F00" /><path d="M29 38H31V39H29V38Z" fill="#F00" /></svg>`
  const html = `
    <div class="taxi-icon-root"
      style="
        width:44px;height:44px;
        display:flex;align-items:center;justify-content:center;
        transform-origin:22px 22px;
        transform: rotate(var(--rot, 0deg));
        will-change: transform;
      ">
      ${svg}
    </div>`
  return L.divIcon({ className: '', html, iconSize: [44, 44], iconAnchor: [22, 22] })
}

function makeStopNumberIcon(n) {
  const num = String(n)
  const html = `
    <div style="
      width:28px;height:28px;border-radius:14px;
      background:#1976d2;color:#fff;
      display:flex;align-items:center;justify-content:center;
      font-weight:800;font-size:14px;
      border:2px solid rgba(255,255,255,0.9);
      box-shadow:0 3px 10px rgba(0,0,0,0.35);
    ">${num}</div>`
  return L.divIcon({ className: '', html, iconSize: [28, 28], iconAnchor: [14, 14] })
}

// ====== Controls ======
function RecenterControl({ onClick, t, lang, getTargetLatLng }) {
  const map = useMap()
  return (
    <div
      className="leaflet-bottom leaflet-right"
      style={{ pointerEvents: 'none', marginBottom: '160px', marginRight: '10px', zIndex: 9999 }}
    >
      <div className="leaflet-control" style={{ pointerEvents: 'auto' }}>
        <button
          onClick={(e) => {
            e.stopPropagation()
            e.preventDefault()
            const ll = getTargetLatLng?.()
            if (ll?.lat != null && ll?.lng != null) {
              map.setView([ll.lat, ll.lng], 18, { animate: false })
            } else {
              map.setView(DEFAULT_CENTER, 18, { animate: false })
            }
            onClick?.()
          }}
          style={{
            backgroundColor: 'white',
            border: '2px solid rgba(0,0,0,0.2)',
            borderRadius: '24px',
            padding: '10px 20px',
            cursor: 'pointer',
            fontWeight: 'bold',
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            color: '#333',
            fontSize: '14px',
          }}
        >
          <span style={{ fontSize: '18px' }}>⌖</span> {t(lang, 'recenter') || '回到目前位置'}
        </button>
      </div>
    </div>
  )
}

function ReplaySpeedControl({ factor, onChange, debugInfo }) {
  const opts = [0.5, 1, 2, 4, 8]
  const speed = Number(debugInfo?.speed ?? 0)
  const sumoTime = Number(debugInfo?.sumoTime ?? 0)
  const isSumo = Boolean(debugInfo?.isSumo)

  return (
    <div
      className="leaflet-top leaflet-left"
      style={{
        marginTop: `${HEADER_OFFSET_PX + 12}px`,
        marginLeft: '10px',
        zIndex: 9999,
        pointerEvents: 'none',
      }}
    >
      <div
        className="leaflet-control"
        style={{
          pointerEvents: 'auto',
          background: 'rgba(255,255,255,0.92)',
          padding: '8px 10px',
          borderRadius: '10px',
          boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
          fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
          fontSize: '12px',
          minWidth: 220,
        }}
      >
        <div style={{ fontWeight: 800, marginBottom: 6 }}>Replay Speed</div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {opts.map(v => {
            const active = Math.abs(v - Number(factor || 1)) < 1e-9
            return (
              <button
                key={v}
                type="button"
                onClick={(e) => { e.preventDefault(); e.stopPropagation(); onChange?.(v) }}
                style={{
                  padding: '6px 10px',
                  borderRadius: 999,
                  border: active ? '2px solid rgba(0,0,0,0.65)' : '1px solid rgba(0,0,0,0.2)',
                  background: active ? '#111' : '#fff',
                  color: active ? '#fff' : '#111',
                  cursor: 'pointer',
                  fontWeight: 800,
                }}
              >
                {v}x
              </button>
            )
          })}
        </div>

        <div style={{ marginTop: 6, opacity: 0.75 }}>
          Current: <b>{Number(factor || 1).toFixed(2)}x</b>
        </div>

        <div
          style={{
            marginTop: 10,
            paddingTop: 8,
            borderTop: '1px solid rgba(0,0,0,0.12)',
            fontFamily: 'monospace',
          }}
        >
          <div><strong>TRACE</strong> <span style={{ opacity: 0.75 }}>REPLAY</span></div>
          <div>
            Source: <span style={{ fontWeight: 800, color: isSumo ? '#0a0' : '#c60' }}>
              {isSumo ? 'SUMO' : 'PHYSICS'}
            </span>
          </div>
          <div>Speed: {Math.round(speed)} km/h</div>
          <div>t: {Number.isFinite(sumoTime) ? sumoTime.toFixed(2) : '0.00'} s</div>
        </div>
      </div>
    </div>
  )
}

function CarFollowController({ enabled, getActiveCarState, onUserInteraction }) {
  const map = useMap()
  const camRef = useRef({ lat: null, lng: null })
  const lastPerfRef = useRef(0)

  useMapEvents({
    dragstart: () => onUserInteraction && onUserInteraction(),
    zoomstart: () => onUserInteraction && onUserInteraction(),
  })

  useEffect(() => {
    if (!enabled) return
    let raf = 0
    const loop = (t) => {
      raf = requestAnimationFrame(loop)
      const s = getActiveCarState?.()
      if (
        !s ||
        !Number.isFinite(Number(s.lat)) ||
        !Number.isFinite(Number(s.lng)) ||
        (Math.abs(Number(s.lat)) < 1e-9 && Math.abs(Number(s.lng)) < 1e-9)
      ) return

      const dt = Math.max(0, t - (lastPerfRef.current || t))
      lastPerfRef.current = t

      if (camRef.current.lat == null) {
        camRef.current = { lat: Number(s.lat), lng: Number(s.lng) }
        map.setView([s.lat, s.lng], map.getZoom(), { animate: false })
        return
      }

      const kPos = smoothLerpFactor(dt, 40)
      camRef.current.lat += (Number(s.lat) - camRef.current.lat) * kPos
      camRef.current.lng += (Number(s.lng) - camRef.current.lng) * kPos

      map.setView([camRef.current.lat, camRef.current.lng], map.getZoom(), {
        animate: false,
        noMoveStart: true,
      })
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [enabled, getActiveCarState, map])

  return null
}

function CarRuntimeLayer({
  order,
  routeCoords,
  cumDist,
  sumoJson,
  simulateVehicles,
  completedOrderIds,
  completedOnceRef,
  lastCarPosRef,
  stashCarPos,
  onOrderArrived,
  onOrderCompleted,
  setDebugInfo,
  mode,
  driverStartPos,
  playbackFactor = 1,
}) {
  const map = useMap()
  const groupRef = useRef(null)
  const markerRef = useRef(null)
  const aheadRef = useRef(null)
  const rafRef = useRef(0)

  const orderRef = useRef(order)
  const routeRef = useRef(routeCoords)
  const cumRef = useRef(cumDist)
  const simRef = useRef(simulateVehicles)

  const playbackRef = useRef(Number(playbackFactor) || 1)

  const currentDistRef = useRef(0)
  const currentSpeedRef = useRef(INITIAL_V)
  const lastTimeRef = useRef(0)
  const routeLenRef = useRef(0)

  const headingRef = useRef(0)

  function shortestAngleDeltaDeg(a, b) {
    return ((b - a + 540) % 360) - 180
  }

  function computeRouteHeadingDeg(routeCoords_, cumDist_, distM, lookaheadM) {
    if (!routeCoords_ || !cumDist_) return null
    const total = cumDist_[cumDist_.length - 1]
    if (!(total > 1)) return null

    const d0 = Math.max(0, Math.min(distM, total))
    const d1 = Math.max(0, Math.min(distM + Math.max(1, lookaheadM), total))

    const p0 = positionAtDistance(routeCoords_, cumDist_, d0)
    const p1 = positionAtDistance(routeCoords_, cumDist_, d1)
    if (!p0 || !p1) return null

    const h = computeHeadingDeg(p0, p1)
    return Number.isFinite(h) ? h : null
  }

  useEffect(() => { orderRef.current = order }, [order])
  useEffect(() => { routeRef.current = routeCoords }, [routeCoords])
  useEffect(() => { cumRef.current = cumDist }, [cumDist])
  useEffect(() => { simRef.current = simulateVehicles }, [simulateVehicles])
  useEffect(() => { playbackRef.current = Number(playbackFactor) || 1 }, [playbackFactor])

  useEffect(() => {
    if (cumDist && cumDist.length > 0) routeLenRef.current = cumDist[cumDist.length - 1]
    else routeLenRef.current = 0
  }, [cumDist])

  const shouldRender = useMemo(() => {
    if (!order?.id) return false
    if (completedOrderIds?.has(order.id)) return false
    if (mode === 'passenger' && !getOrderDriverId(order)) return false
    return true
  }, [order, mode, completedOrderIds])

  useEffect(() => {
    if (!shouldRender) return
    const g = L.layerGroup().addTo(map)
    groupRef.current = g

    const route0 = Array.isArray(routeCoords) && routeCoords.length ? routeCoords[0] : null
    const startLL =
      (Array.isArray(driverStartPos) && driverStartPos.length === 2)
        ? driverStartPos
        : (route0 ? [route0[0], route0[1]] : [map.getCenter().lat, map.getCenter().lng])

    markerRef.current = L.marker(startLL, { icon: makeTaxiIcon() })
      .bindTooltip('0 km/h', {
        permanent: true,
        direction: 'right',
        offset: [20, 0],
        className: 'speed-tooltip',
      })
      .addTo(g)

    aheadRef.current = L.polyline([startLL, startLL], {
      weight: 2,
      color: 'rgba(0,0,0,0.2)',
    }).addTo(g)

    return () => {
      try { map.removeLayer(g) } catch {}
    }
  }, [map, shouldRender, driverStartPos, routeCoords])

  const profileRef = useRef(null)

  useEffect(() => {
    const ok = getOrderKey(orderRef.current)
    if (!ok) return

    ensureSim(ok)

    const { elapsedMs } = computeElapsedMs(ok)
    currentDistRef.current = (elapsedMs / 1000) * (FALLBACK_SPEED_KPH / 3.6)
    currentSpeedRef.current = INITIAL_V
    lastTimeRef.current = 0
    headingRef.current = 0

    if (!sumoJson) {
      profileRef.current = null
      return
    }

    const pts = pickVehiclePoints(sumoJson, { order: orderRef.current })
profileRef.current = pts ? buildSpeedProfile(pts) : null
  }, [sumoJson, order?.id])

  useEffect(() => {
    if (!shouldRender) return
    const oid = order.id
    const ok = getOrderKey(order)
    if (!ok) return
    ensureSim(ok)

    const loop = (time) => {
      rafRef.current = requestAnimationFrame(loop)
      if (!lastTimeRef.current) lastTimeRef.current = time
      const dtMs = Math.max(0, time - lastTimeRef.current)
      lastTimeRef.current = time

      if (!simRef.current || !isActiveStatus(orderRef.current?.status)) return

      const { elapsedMs } = computeElapsedMs(ok)
      const pf = playbackRef.current || 1

      const tSec = (elapsedMs / 1000) * pf
      const dtSec = (dtMs / 1000) * pf

      let lat = 0
      let lng = 0
      let speedKph = 0
      let heading = 0
      const profile = profileRef.current
      let isSumoActive = false
      let distOnRoute = null
      let sumoAngle = null

      if (profile && routeLenRef.current > 1 && profile.totalDist > 0) {
        isSumoActive = true

        const traceSec = profile.startT + tSec
        const rawDist = profile.distAt(traceSec)
        const rawSpeedMps = profile.speedAt(traceSec)
        sumoAngle = profile.angleAt(traceSec)

        const scale = routeLenRef.current / Math.max(1, profile.totalDist)
        distOnRoute = rawDist * scale

        const shownSpeedMps = rawSpeedMps * scale
        speedKph = Math.round(shownSpeedMps * 3.6)
      } else {
        if (routeRef.current && routeRef.current.length >= 2) {
          const maxSpeedKph = Number(orderRef.current?.simSpeedKph) || FALLBACK_SPEED_KPH
          const maxSpeedMps = maxSpeedKph / 3.6
          const simState = readSim(ok)

          if (!simState || !simState.running) {
            currentSpeedRef.current = 0
          } else {
            const totalDist = cumRef.current ? cumRef.current[cumRef.current.length - 1] : 0
            const distRem = Math.max(0, totalDist - currentDistRef.current)
            const maxSafeSpeed = Math.sqrt(2 * CAR_DECEL * distRem)
            const targetSpeed = Math.min(maxSpeedMps, maxSafeSpeed)

            if (currentSpeedRef.current < targetSpeed) {
              currentSpeedRef.current += CAR_ACCEL * dtSec
              if (currentSpeedRef.current > targetSpeed) currentSpeedRef.current = targetSpeed
            } else {
              currentSpeedRef.current -= CAR_DECEL * dtSec
              if (currentSpeedRef.current < targetSpeed) currentSpeedRef.current = targetSpeed
            }
            if (currentSpeedRef.current < 0) currentSpeedRef.current = 0

            const stepDist = currentSpeedRef.current * dtSec
            currentDistRef.current += stepDist
          }

          distOnRoute = currentDistRef.current
          speedKph = Math.round(currentSpeedRef.current * 3.6)
        }
      }

      let pos = null
      if (distOnRoute != null && cumRef.current) {
        pos = positionAtDistance(routeRef.current, cumRef.current, distOnRoute)
      } else if (!isSumoActive) {
        const idx = Math.min(Math.floor(elapsedMs / STEP_MS), (routeRef.current?.length ?? 1) - 1)
        const p = routeRef.current?.[idx]
        if (p) pos = { lat: p[0], lng: p[1] }
      }

      if (pos) {
        lat = pos.lat
        lng = pos.lng

        const safeDist = Number.isFinite(distOnRoute) ? distOnRoute : 0
        const routeHeading = computeRouteHeadingDeg(
          routeRef.current,
          cumRef.current,
          safeDist,
          HEADING_LOOKAHEAD_METERS
        )

        let rawHeading = null
        if (Number.isFinite(routeHeading)) rawHeading = routeHeading
        else if (isSumoActive && Number.isFinite(sumoAngle)) rawHeading = normDeg(sumoAngle)
        else {
          const prev = lastCarPosRef.current.get(oid)
          const h = prev ? computeHeadingDeg(prev, { lat, lng }) : null
          rawHeading = Number.isFinite(h) ? h : (prev?.heading || 0)
        }

        const kH = smoothLerpFactor(dtMs, HEADING_SMOOTH_TAU_MS)
        const cur = Number(headingRef.current || 0)
        const delta = shortestAngleDeltaDeg(cur, rawHeading)
        const smoothed = normDeg(cur + delta * kH)

        headingRef.current = smoothed
        heading = smoothed
      }

      if (
        Number.isFinite(lat) &&
        Number.isFinite(lng) &&
        !(Math.abs(lat) < 1e-9 && Math.abs(lng) < 1e-9)
      ) {
        const posData = { lat, lng, heading, speedKph }
        lastCarPosRef.current.set(oid, posData)
        stashCarPos(oid, posData)
      }

      if (markerRef.current && lat !== 0) {
        markerRef.current.setLatLng([lat, lng])

        const tooltip = markerRef.current.getTooltip()
        if (tooltip) tooltip.setContent(`${speedKph} km/h`)

        const el = markerRef.current.getElement()
        if (el) {
          const rot = normDeg(heading + ICON_HEADING_OFFSET_DEG)
          el.style.setProperty('--rot', `${rot}deg`)
        }
      }

      const rlen = routeLenRef.current
      const isCompletedCheck =
        !completedOrderIds?.has(oid) &&
        !completedOnceRef.current.has(ok) &&
        Number.isFinite(distOnRoute) &&
        distOnRoute >= Math.max(0, rlen - ARRIVAL_THRESHOLD_METERS)

      if (isCompletedCheck) {
        completedOnceRef.current.add(ok)
        completeSim(ok)
        if (onOrderCompleted) onOrderCompleted(oid, { lat, lng, heading, speedKph })
      }

      if (setDebugInfo) {
        setDebugInfo({ speed: speedKph, sumoTime: tSec, isSumo: isSumoActive })
      }
    }

    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [order?.id, stashCarPos, shouldRender, completedOrderIds, onOrderCompleted, setDebugInfo])

  return null
}

function VisibilitySimSync({ orders = [], simulateVehicles }) {
  useEffect(() => {
    const onVis = () => {
      const hidden = document.visibilityState !== 'visible'
      for (const o of orders) {
        const k = getOrderKey(o)
        if (!k) continue
        if (hidden) pauseSim(k)
        else if (simulateVehicles) resumeSim(k)
      }
    }
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [orders, simulateVehicles])
  return null
}

export default function MapView({
  lang,
  mode,
  drivers = [],
  orders = [],
  currentDriverId,
  onDriverLocationChange,
  simulateVehicles = true,
  apiFetch,
  onCarPosChange,
  previewEnabled = false,
  previewWaypoints = null,
  previewMarkers = null,
  onHotspotSelect = null,
  selectedHotspot = null,
  completedOrderIds,
  onOrderArrived,
  onOrderCompleted,
  usePersistedDriverLoc = true,
  followActiveCar = null,
  hotspots = [],
  driverClickEnabled = true,
  rotateMapWithHeading = true,
}) {
  const isDriverMode = mode === 'driver'
  const useApiFetch = apiFetch || defaultApiFetch

  const [playbackFactor, setPlaybackFactor] = usePlaybackFactorSync()

  const [debugInfo, setDebugInfo] = useState({ speed: 0, sumoTime: 0, isSumo: false })
  const [sumoJson, setSumoJson] = useState(null)
  const [visualRoutes, setVisualRoutes] = useState({})

  useEffect(() => {
    fetch(SUMO_TRACE_URL)
      .then(res => res.json())
      .then(data => setSumoJson(data))
      .catch(err => console.warn('SUMO Trace load failed:', err))
  }, [])

  const streetViewMode = useMemo(() => {
    if (previewEnabled) return false
    const o = orders[0]
    if (!o || !isActiveStatus(o.status)) return false
    const did = getOrderDriverId(o)
    if (!isDriverMode) return did != null
    return currentDriverId != null && did != null && sameId(did, currentDriverId)
  }, [previewEnabled, orders, isDriverMode, currentDriverId])

  // ✅ 修正：
// 跟車不只 streetViewMode，要真正吃 followActiveCar
const [isFollowing, setIsFollowing] = useState(true)

const [tileUrl, setTileUrl] = useState('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png')

const cumRef = useRef(new Map())
const lastCarPosRef = useRef(new Map())
const latestCarStateRef = useRef(new Map())
const arrivedOnceRef = useRef(new Set())
const completedOnceRef = useRef(new Set())

const stashCarPos = useCallback((oid, val) => {
  latestCarStateRef.current.set(oid, { ...val, ts: Date.now() })
  onCarPosChange?.(oid, val)
}, [onCarPosChange])

const runtimeOrders = useMemo(() => {
  const result = []
  const seenDriverIds = new Set()

  const sorted = [...orders].sort((a, b) => {
    const ta = Date.parse(a?.updatedAt || a?.updated_at || a?.createdAt || a?.created_at || 0) || 0
    const tb = Date.parse(b?.updatedAt || b?.updated_at || b?.createdAt || b?.created_at || 0) || 0
    if (ta !== tb) return tb - ta
    return Number(b?.id || 0) - Number(a?.id || 0)
  })

  for (const o of sorted) {
    if (!o?.id) continue
    if (completedOrderIds?.has?.(o.id)) continue

    const driverId = getOrderDriverId(o)

    // 沒派司機的不跑 runtime car
    if (driverId == null) continue

    // 只保留進行中訂單
    if (!isActiveStatus(o?.status)) continue

    // 乘客端未真正派車前，不要跑 runtime 車
    if (mode === 'passenger' && !canShowPassengerRoute(o)) continue

    // 司機端只顯示自己的單
    if (
      mode === 'driver' &&
      currentDriverId != null &&
      !sameId(driverId, currentDriverId)
    ) {
      continue
    }

    // 每個司機只保留當下那張進行中的單
    const key = Number(driverId)
    if (seenDriverIds.has(key)) continue
    seenDriverIds.add(key)

    result.push(o)
  }

  return result
}, [orders, completedOrderIds, mode, currentDriverId])


const frozenStartPosByOrderId = useMemo(() => {
  const out = new Map()

  for (const o of orders) {
    const orderKey = getOrderKey(o)
    if (!orderKey) continue

    let start = readOrderStart(orderKey)

    if (
      !start ||
      !Number.isFinite(Number(start.lat)) ||
      !Number.isFinite(Number(start.lng))
    ) {
      const drvId = getOrderDriverId(o)
      const hasAssignedDriver = drvId != null && isActiveStatus(o?.status)
      if (!hasAssignedDriver) continue

      const livePersisted =
        usePersistedDriverLoc && drvId != null
          ? (() => {
              try {
                const raw = localStorage.getItem(`driverLoc:${drvId}`)
                if (!raw) return null
                const parsed = JSON.parse(raw)
                const lat = Number(parsed?.lat)
                const lng = Number(parsed?.lng)
                if (Number.isFinite(lat) && Number.isFinite(lng)) {
                  return { lat, lng }
                }
                return null
              } catch {
                return null
              }
            })()
          : null

      if (livePersisted) {
        start = { lat: Number(livePersisted.lat), lng: Number(livePersisted.lng) }
        writeOrderStartOnce(orderKey, start)
      } else {
        const drv = drivers.find(d => sameId(d.id, drvId))
        if (
          drv &&
          Number.isFinite(Number(drv.lat)) &&
          Number.isFinite(Number(drv.lng))
        ) {
          start = { lat: Number(drv.lat), lng: Number(drv.lng) }
          writeOrderStartOnce(orderKey, start)
        }
      }
    }

    if (
      start &&
      Number.isFinite(Number(start.lat)) &&
      Number.isFinite(Number(start.lng))
    ) {
      out.set(o.id, start)
    }
  }

  return out
}, [orders, drivers, usePersistedDriverLoc])

const hasTrackTarget = useMemo(() => {
  if (runtimeOrders.length > 0) return true

  const o = orders?.[0]
  if (!o?.id) return false

  const frozenStart = frozenStartPosByOrderId.get(o.id)
  return Boolean(
    frozenStart &&
    Number.isFinite(Number(frozenStart.lat)) &&
    Number.isFinite(Number(frozenStart.lng))
  )
}, [runtimeOrders, orders, frozenStartPosByOrderId])

useEffect(() => {
  const canFollow = !previewEnabled && (streetViewMode || (followActiveCar && hasTrackTarget))
  if (canFollow) setIsFollowing(true)
}, [streetViewMode, followActiveCar, hasTrackTarget, previewEnabled])

const shouldFollow = useMemo(() => {
  if (!isFollowing) return false
  if (previewEnabled) return false
  return Boolean(streetViewMode || (followActiveCar && hasTrackTarget))
}, [isFollowing, streetViewMode, followActiveCar, hasTrackTarget, previewEnabled])

const storageKey = mapStateKey({ mode, driverId: currentDriverId, previewEnabled })

const getActiveCarState = useCallback(() => {
  const active = runtimeOrders[0] || orders?.[0]
  if (!active?.id) return null

  const live = latestCarStateRef.current.get(active.id)
  if (live) return live

  const frozenStart = frozenStartPosByOrderId.get(active.id)
  if (
    frozenStart &&
    Number.isFinite(Number(frozenStart.lat)) &&
    Number.isFinite(Number(frozenStart.lng))
  ) {
    return {
      lat: Number(frozenStart.lat),
      lng: Number(frozenStart.lng),
      heading: 0,
      speedKph: 0,
    }
  }

  return null
}, [runtimeOrders, orders, frozenStartPosByOrderId])

const handleRecenter = useCallback(() => {
  setIsFollowing(true)
}, [])

  // ✅ 修正：
  // 完成訂單後：
  // 1. driver location 寫回終點
  // 2. 下一張訂單若使用新 orderKey，會直接吃最新 driverLoc
  // 3. 當前這張訂單自己的 frozen start 也更新為最後位置，避免畫面回彈
  const handleLocalOrderCompleted = useCallback((oid, lastPos) => {
    const order = orders.find(o => o.id === oid)
    const drvId = getOrderDriverId(order)
    const orderKey = getOrderKey(order)

    if (
      drvId &&
      lastPos &&
      Number.isFinite(Number(lastPos.lat)) &&
      Number.isFinite(Number(lastPos.lng))
    ) {
      const payload = {
        lat: Number(lastPos.lat),
        lng: Number(lastPos.lng),
        heading: lastPos.heading,
        speedKph: lastPos.speedKph,
      }

      onDriverLocationChange?.({ id: drvId, ...payload })

      try {
        localStorage.setItem(`driverLoc:${drvId}`, JSON.stringify({
          lat: payload.lat,
          lng: payload.lng,
        }))
      } catch {}

      if (orderKey) {
        writeOrderStart(orderKey, { lat: payload.lat, lng: payload.lng })
      }

      const f = apiFetch || defaultApiFetch
      f(`/api/drivers/${drvId}/location`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lat: payload.lat, lng: payload.lng }),
      }).catch(() => {})
    }

    onOrderCompleted?.(oid)
  }, [orders, onDriverLocationChange, apiFetch, onOrderCompleted])

  useEffect(() => {
    let cancelled = false

    async function load() {
      const newRoutes = {}

      for (const o of orders) {
        const orderKey = getOrderKey(o)
        const driverId = getOrderDriverId(o)
        const frozenStart = frozenStartPosByOrderId.get(o.id) || null
        const hasAssignedDriver = driverId != null && isActiveStatus(o?.status)

        if (mode === 'passenger' && hasAssignedDriver) {
          const sharedRoute = readOrderRoute(orderKey)
          if (sharedRoute && sharedRoute.length >= 2) {
            newRoutes[o.id] = sharedRoute
            cumRef.current.set(o.id, buildCumDist(sharedRoute))
            continue
          }
        }

        let wps = null
        if (mode === 'passenger') {
          if (hasAssignedDriver) {
            wps = buildCarWaypoints(o, mode, drivers, currentDriverId, frozenStart || null)
          } else {
            wps = buildPlannedWaypoints(o)
          }
        } else {
          wps = buildCarWaypoints(o, mode, drivers, currentDriverId, frozenStart || null)
        }

        if (!wps || wps.length < 2) continue

        try {
          let cs = await fetchOsrmRoute(wps)
          if (cs && cs.length > 0 && o.dropoffLocation) {
            const lastPt = cs[cs.length - 1]
            const dropoff = [o.dropoffLocation.lat, o.dropoffLocation.lng]
            const dist = Math.sqrt(
              Math.pow(lastPt[0] - dropoff[0], 2) +
              Math.pow(lastPt[1] - dropoff[1], 2)
            )
            if (dist > 0.0001) cs.push(dropoff)
          }

          newRoutes[o.id] = cs
          cumRef.current.set(o.id, buildCumDist(cs))

          if (hasAssignedDriver) {
            writeOrderRoute(orderKey, cs)
          }
        } catch (err) {
          console.warn('route build failed', o?.id, err)
        }
      }

      if (!cancelled) setVisualRoutes(newRoutes)
    }

    load()
    return () => {
      cancelled = true
    }
  }, [orders, mode, currentDriverId, drivers, frozenStartPosByOrderId])

  useEffect(() => {
    orders.forEach(o => {
      const k = getOrderKey(o)
      if (k) simulateVehicles ? resumeSim(k) : pauseSim(k)
    })
  }, [simulateVehicles, orders])

  const getHotspotColor = (val) => {
    if (val > 50) return '#ff0000'
    if (val > 20) return '#ff8800'
    if (val > 5) return '#ffff00'
    return '#00c853'
  }

  const stopPinsByOrderId = useMemo(() => {
    const out = new Map()
    for (const o of orders) {
      const list = Array.isArray(o?.stops) ? o.stops : []
      const pins = []
      let n = 0
      for (const s of list) {
        const lat = Number(s?.lat)
        const lng = Number(s?.lng ?? s?.lon)
        if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue

        const type = String(s?.type || '').toLowerCase()
        if (type === 'driver' || type === 'pickup') continue

        n += 1
        pins.push({
          n,
          lat,
          lng,
          label: s?.label || s?.name || `Stop ${n}`,
        })
      }
      out.set(o.id, pins)
    }
    return out
  }, [orders])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <MapContainer
  center={DEFAULT_CENTER}
  zoom={shouldFollow ? 18 : DEFAULT_ZOOM}
  style={{ width: '100%', height: '100%' }}
  zoomControl={!shouldFollow}
  maxZoom={18}
>
        <MapSizeFixer />
<MapViewInitializer
  storageKey={storageKey}
  streetViewMode={shouldFollow}
  getInitialTarget={getActiveCarState}
/>
        <MapStateTracker
          storageKey={storageKey}
          disabled={shouldFollow}
        />

        <VisibilitySimSync orders={orders} simulateVehicles={simulateVehicles} />

        <TileLayer
          url={tileUrl}
          attribution="&copy; OpenStreetMap"
          maxZoom={18}
          maxNativeZoom={17}
        />

        <CarFollowController
          enabled={shouldFollow}
          getActiveCarState={getActiveCarState}
          onUserInteraction={() => setIsFollowing(false)}
        />

        <RecenterControl
          onClick={handleRecenter}
          t={t}
          lang={lang}
          getTargetLatLng={() => getActiveCarState?.()}
        />

        <ReplaySpeedControl
          factor={playbackFactor}
          onChange={setPlaybackFactor}
          debugInfo={debugInfo}
        />

        {isDriverMode && currentDriverId && (
          <DriverClickHandler
            enabled={Boolean(driverClickEnabled)}
            driverId={currentDriverId}
            onLocationChange={onDriverLocationChange}
            apiFetch={useApiFetch}
          />
        )}

        {drivers.map(d => {
          const currentOrder = orders.find(o => sameId(getOrderDriverId(o), d.id))
          if (currentOrder) {
            const isOrderActive = isActiveStatus(currentOrder.status)
            const isOrderCompleted = completedOrderIds?.has(currentOrder.id)
            if (isOrderActive || isOrderCompleted) return null
          }

          if (
            Number.isFinite(Number(d.lat)) &&
            Number.isFinite(Number(d.lng))
          ) {
            return (
              <Marker
                key={`driver-${d.id}`}
                position={[Number(d.lat), Number(d.lng)]}
                icon={makeTaxiIcon()}
                opacity={isDriverMode && sameId(d.id, currentDriverId) ? 1 : 0.7}
              />
            )
          }
          return null
        })}

        {isDriverMode && hotspots.length > 0 && hotspots.map((h, i) => {
          const isSel =
            selectedHotspot &&
            Math.abs(Number(selectedHotspot.lat) - Number(h.lat)) < 1e-9 &&
            Math.abs(Number(selectedHotspot.lon ?? selectedHotspot.lng) - Number(h.lon ?? h.lng)) < 1e-9

          return (
            <CircleMarker
              key={`hotspot-${i}`}
              center={[h.lat, h.lon]}
              radius={Math.min(50, (Number(h.pred_rides) || 0) * 1.5 + 5)}
              pathOptions={{
                color: getHotspotColor(Number(h.pred_rides) || 0),
                fillColor: getHotspotColor(Number(h.pred_rides) || 0),
                fillOpacity: isSel ? 0.75 : 0.4,
                weight: isSel ? 3 : 1,
              }}
              eventHandlers={{ click: () => onHotspotSelect?.(h) }}
            >
              <Popup>
                <div style={{ fontSize: '14px' }}>
                  <strong>{h.Zone}</strong> ({h.Borough})<br />
                  預測需求: <b>{Number(h.pred_rides ?? 0).toFixed(2)}</b> 單/時<br />
                  final_score: <b>{h.final_score != null ? Number(h.final_score).toFixed(2) : 'N/A'}</b>
                </div>
              </Popup>
            </CircleMarker>
          )
        })}

        {previewEnabled ? (
          <>
            {previewWaypoints && (
              <Polyline
                positions={previewWaypoints}
                pathOptions={{ color: 'blue', weight: 4, opacity: 0.6, dashArray: '10, 10' }}
              />
            )}
            {previewMarkers?.pickup && <Marker position={previewMarkers.pickup} icon={passengerIcon} />}
            {previewMarkers?.dropoff && <Marker position={previewMarkers.dropoff} icon={dropoffIcon} />}

            {previewMarkers?.stops?.map((s, idx) => (
              <Marker
                key={`pstop-${idx}`}
                position={[s.lat, s.lng]}
                icon={makeStopNumberIcon(idx + 1)}
              >
                <Tooltip direction="top" offset={[0, -10]} opacity={0.95} permanent>
                  {idx + 1}
                </Tooltip>
              </Marker>
            ))}
          </>
        ) : (
          <>
            {orders.map(o => (
              <div key={o.id}>
                {(
                  mode !== 'passenger'
                    ? Boolean(visualRoutes[o.id])
                    : canShowPassengerPlannedRoute(o) && Boolean(visualRoutes[o.id])
                ) && (
                  <Polyline positions={visualRoutes[o.id]} pathOptions={{ color: '#999', weight: 5 }} />
                )}

                {o.pickupLocation && <Marker position={[o.pickupLocation.lat, o.pickupLocation.lng]} icon={passengerIcon} />}
                {o.dropoffLocation && <Marker position={[o.dropoffLocation.lat, o.dropoffLocation.lng]} icon={dropoffIcon} />}

                {(stopPinsByOrderId.get(o.id) || []).map(p => (
                  <Marker
                    key={`stop-${o.id}-${p.n}`}
                    position={[p.lat, p.lng]}
                    icon={makeStopNumberIcon(p.n)}
                  >
                    <Tooltip direction="top" offset={[0, -10]} opacity={0.95} permanent>
                      {p.n}
                    </Tooltip>
                    <Popup>
                      <div style={{ fontSize: 14 }}>
                        <b>Stop {p.n}</b><br />
                        {p.label}
                      </div>
                    </Popup>
                  </Marker>
                ))}
              </div>
            ))}

            {runtimeOrders.map(o => {
              const frozenStart = frozenStartPosByOrderId.get(o.id)
              const startPos =
                frozenStart &&
                Number.isFinite(Number(frozenStart.lat)) &&
                Number.isFinite(Number(frozenStart.lng))
                  ? [Number(frozenStart.lat), Number(frozenStart.lng)]
                  : null

              return (
                <CarRuntimeLayer
                  key={`run-${o.id}`}
                  order={o}
                  routeCoords={visualRoutes[o.id]}
                  cumDist={cumRef.current.get(o.id)}
                  sumoJson={sumoJson}
                  simulateVehicles={simulateVehicles}
                  completedOrderIds={completedOrderIds}
                  completedOnceRef={completedOnceRef}
                  lastCarPosRef={lastCarPosRef}
                  stashCarPos={stashCarPos}
                  onOrderArrived={onOrderArrived}
                  onOrderCompleted={handleLocalOrderCompleted}
                  setDebugInfo={setDebugInfo}
                  mode={mode}
                  driverStartPos={startPos}
                  playbackFactor={playbackFactor}
                />
              )
            })}
          </>
        )}
      </MapContainer>
    </div>
  )
}

export { usePlaybackFactorSync, readPlaybackFactor, writePlaybackFactor, PLAYBACK_LS_KEY }

