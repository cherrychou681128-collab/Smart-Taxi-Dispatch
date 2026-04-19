// src/views/DriverPage.jsx (或你實際檔名)
// ⚠️ 這支是你「司機派車建議」頁
import React, { useEffect, useState, useRef, useMemo } from 'react'
import {
  MapContainer,
  TileLayer,
  Marker,
  Polyline,
  useMap,
  CircleMarker,
  Popup,
} from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { apiFetch } from './apiBase.js' // ✅ 改：統一用你的 apiFetch（你也可以換成相對路徑）
import { taxiIcon, passengerIcon, dropoffIcon } from './mapIcons'

// ====== 設定 ======
const DEFAULT_CENTER = [40.758, -73.9855]
const ZOOM_LEVEL = 11
const STEP_MS = 60
const FALLBACK_SPEED_KPH = 28

// ====== Helper Functions ======
function sameId(a, b) {
  const A = Number(a)
  const B = Number(b)
  return Number.isFinite(A) && Number.isFinite(B) && A === B
}

function isActiveStatus(status) {
  return new Set([
    'assigned',
    'accepted',
    'en_route',
    'enroute',
    'picked_up',
    'in_progress',
    'on_trip',
    'ongoing',
  ]).has(String(status || '').toLowerCase())
}

function getOrderDriverId(order) {
  return order?.driverId ?? order?.assignedDriverId ?? order?.driver_id ?? null
}

function getOrderKey(order) {
  const id = Number(order?.id)
  if (!Number.isFinite(id)) return null
  const createdAt = order?.createdAt || order?.created_at || order?.updatedAt || ''
  return `${id}::${String(createdAt)}`
}

// ====== SVG 車輛圖標 ======
function makeTaxiIcon() {
  const svg = `
    <svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 8C12 5.79086 13.7909 4 16 4H28C30.2091 4 32 5.79086 32 8V36C32 38.2091 30.2091 40 28 40H16C13.7909 40 12 38.2091 12 36V8Z" fill="black" fill-opacity="0.3" transform="translate(2, 2)"/>
      <path d="M12 8C12 5.79086 13.7909 4 16 4H28C30.2091 4 32 5.79086 32 8V36C32 38.2091 30.2091 40 28 40H16C13.7909 40 12 38.2091 12 36V8Z" fill="#F4C430" stroke="#E6B800" stroke-width="1"/>
      <path d="M14 10H30V16H14V10Z" fill="#333"/>
      <path d="M14 30H30V34H14V30Z" fill="#333"/>
      <rect x="18" y="20" width="8" height="4" rx="1" fill="#FFD700" stroke="#D4AF37" stroke-width="0.5"/>
      <path d="M13 5H15V6H13V5Z" fill="#FFF" />
      <path d="M29 5H31V6H29V5Z" fill="#FFF" />
      <path d="M13 38H15V39H13V38Z" fill="#F00" />
      <path d="M29 38H31V39H29V38Z" fill="#F00" />
    </svg>
  `
  const html = `
    <div style="width:44px;height:44px;display:flex;align-items:center;justify-content:center;">
      <div class="taxi-icon-inner" style="width:44px;height:44px;transform: rotate(0deg); transition: transform 0.1s linear;">
        ${svg}
      </div>
    </div>
  `
  return L.divIcon({ className: '', html, iconSize: [44, 44], iconAnchor: [22, 22] })
}

// ====== 數學運算 ======
function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)))
}

function toRad(d) {
  return (d * Math.PI) / 180
}
function haversineMeters(lat1, lng1, lat2, lng2) {
  const R = 6371000
  const dLat = toRad(lat2 - lat1)
  const dLng = toRad(lng2 - lng1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)))
}
function buildCumDist(coords) {
  if (!coords || coords.length < 2) return null
  const cum = [0]
  for (let i = 1; i < coords.length; i++) {
    cum.push(
      cum[i - 1] +
        haversineMeters(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    )
  }
  return cum
}
function positionAtDistance(coords, cum, d) {
  if (!coords || !cum) return null
  const total = cum[cum.length - 1]
  const dist = Math.max(0, Math.min(d, total))
  let i = 0
  while (i < cum.length - 2 && cum[i + 1] < dist) i++
  const ratio = (dist - cum[i]) / (cum[i + 1] - cum[i]) || 0
  const [lat0, lng0] = coords[i]
  const [lat1, lng1] = coords[i + 1]
  return { lat: lat0 + (lat1 - lat0) * ratio, lng: lng0 + (lng1 - lng0) * ratio }
}

// ====== OSRM ======
async function getRoute(fromLat, fromLon, toLat, toLon) {
  const url = `https://router.project-osrm.org/route/v1/driving/${fromLon},${fromLat};${toLon},${toLat}?overview=full&geometries=geojson`
  try {
    const res = await fetch(url)
    const data = await res.json()
    if (!data.routes || !data.routes.length) return { coords: [], dist: null }
    const r = data.routes[0]
    return {
      coords: r.geometry.coordinates.map((c) => [c[1], c[0]]),
      dist: r.distance / 1000.0,
    }
  } catch {
    return { coords: [], dist: null }
  }
}

// ====== Simulation (localStorage) ======
function simKey(k) {
  return `sim:${k}`
}
function readSim(k) {
  try {
    return JSON.parse(localStorage.getItem(simKey(k)))
  } catch {
    return null
  }
}
function ensureSim(k) {
  const c = readSim(k)
  if (c) return c
  const init = { elapsedMs: 0, startedAt: Date.now(), running: true, stepMs: STEP_MS, completed: false }
  try {
    localStorage.setItem(simKey(k), JSON.stringify(init))
  } catch {}
  return init
}
function computeElapsedMs(k) {
  const now = Date.now()
  const c = ensureSim(k)
  const total = (c.elapsedMs || 0) + (c.running ? Math.max(0, now - (c.startedAt || now)) : 0)
  return { elapsedMs: total }
}

// ====== 311 顏色 ======
function colorByFinalScore(s) {
  const v = Number(s ?? 0)
  if (v >= 80) return '#d50000'
  if (v >= 60) return '#ff6d00'
  if (v >= 40) return '#ffd600'
  if (v >= 20) return '#00c853'
  return '#00b0ff'
}

function computeRadiusScaler(zones) {
  const preds = (zones || [])
    .map((z) => Number(z?.pred_rides ?? 0))
    .filter((x) => Number.isFinite(x) && x >= 0)
  const maxPred = preds.length ? Math.max(...preds) : 0
  const looksLikeCounts = maxPred > 2

  return function radiusFor(pred) {
    const p = Math.max(0, Number(pred ?? 0))
    if (!Number.isFinite(p)) return 5
    if (!looksLikeCounts) return Math.min(35, p * 2 + 5)
    const norm = maxPred > 0 ? p / maxPred : 0
    return Math.min(35, 5 + norm * 30)
  }
}

// ====== CarRuntimeLayer (接單時移動車輛) ======
function CarRuntimeLayer({ order, routeCoords, simulateVehicles, driverPos }) {
  const map = useMap()
  const markerRef = useRef(null)
  const rafRef = useRef(0)

  const cumRef = useRef(null)
  const currentDistRef = useRef(0)
  const lastTimeRef = useRef(0)

  useEffect(() => {
    if (routeCoords) cumRef.current = buildCumDist(routeCoords)
    else cumRef.current = null
  }, [routeCoords])

  useEffect(() => {
    if (!order?.id) return
    const g = L.layerGroup().addTo(map)

    const start =
      (Array.isArray(routeCoords) && routeCoords.length >= 1)
        ? routeCoords[0]
        : (driverPos && Number.isFinite(driverPos.lat) && Number.isFinite(driverPos.lng))
          ? [driverPos.lat, driverPos.lng]
          : [map.getCenter().lat, map.getCenter().lng]

    const m = L.marker(start, { icon: makeTaxiIcon() }).addTo(g)
    markerRef.current = m

    return () => {
      try { map.removeLayer(g) } catch {}
    }
  }, [map, order?.id, routeCoords, driverPos])

  useEffect(() => {
    if (!order?.id || !routeCoords || routeCoords.length < 2) return
    const ok = getOrderKey(order)

    currentDistRef.current = 0
    lastTimeRef.current = 0

    const loop = (time) => {
      rafRef.current = requestAnimationFrame(loop)
      if (!lastTimeRef.current) lastTimeRef.current = time
      const dtMs = Math.max(0, time - lastTimeRef.current)
      lastTimeRef.current = time

      if (!simulateVehicles || !isActiveStatus(order.status)) return

      const { elapsedMs } = computeElapsedMs(ok)
      const spd = FALLBACK_SPEED_KPH
      const speedMps = spd / 3.6
      const stepDist = speedMps * (dtMs / 1000)

      const simState = readSim(ok)
      if (simState && simState.running) currentDistRef.current += stepDist

      let dist = currentDistRef.current
      if (dist === 0 && elapsedMs > 0) {
        dist = (elapsedMs / 1000) * speedMps
        currentDistRef.current = dist
      }

      let pos = null
      if (cumRef.current) {
        pos = positionAtDistance(routeCoords, cumRef.current, dist)
      } else {
        const idx = Math.min(Math.floor(elapsedMs / STEP_MS), routeCoords.length - 1)
        const p = routeCoords[idx]
        pos = { lat: p[0], lng: p[1] }
      }

      if (pos && markerRef.current) markerRef.current.setLatLng([pos.lat, pos.lng])
    }

    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [order, routeCoords, simulateVehicles])

  return null
}

// ====== RecommendationCarLayer ======
function RecommendationCarLayer({ playRouteCoords, playingKey }) {
  const map = useMap()
  const markerRef = useRef(null)
  const timerRef = useRef(null)

  useEffect(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (markerRef.current) {
      try { map.removeLayer(markerRef.current) } catch {}
      markerRef.current = null
    }

    if (!playRouteCoords || playRouteCoords.length < 2) return

    const m = L.marker(playRouteCoords[0], { icon: makeTaxiIcon() }).addTo(map)
    markerRef.current = m

    let i = 0
    timerRef.current = setInterval(() => {
      i++
      if (!markerRef.current) return
      if (i >= playRouteCoords.length) {
        clearInterval(timerRef.current)
        timerRef.current = null
        return
      }
      markerRef.current.setLatLng(playRouteCoords[i])
    }, 80)

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
      if (markerRef.current) {
        try { map.removeLayer(markerRef.current) } catch {}
        markerRef.current = null
      }
    }
  }, [map, playingKey, playRouteCoords])

  return null
}

// ====== AutoRecommendController ======
function AutoRecommendController({
  driverPos,
  zones,
  setTop3,
  setRoutes3,
  setActiveRoute,
  setPlayRoute,
  setPlayingKey,
}) {
  const map = useMap()
  const lastKeyRef = useRef('')

  useEffect(() => {
    if (!driverPos || !Number.isFinite(driverPos.lat) || !Number.isFinite(driverPos.lng)) return
    if (!zones || zones.length === 0) return

    const key = `${driverPos.lat.toFixed(6)},${driverPos.lng.toFixed(6)}::${zones.length}`
    if (key === lastKeyRef.current) return
    lastKeyRef.current = key

    ;(async () => {
      setActiveRoute(null)
      setPlayRoute(null)
      setPlayingKey(String(Date.now()))

      const { lat, lng } = driverPos

      const initial = zones
        .slice()
        .sort((a, b) => Number(b.final_score ?? 0) - Number(a.final_score ?? 0))
        .slice(0, 3)
        .map((z) => {
          const d = haversine(lat, lng, z.lat, z.lon)
          return { info: z, straightDist: d }
        })

      const calculated = []
      for (let item of initial) {
        const z = item.info
        const routeInfo = await getRoute(lat, lng, z.lat, z.lon)
        const dist = routeInfo.dist != null ? routeInfo.dist : item.straightDist

        calculated.push({
          info: z,
          coords:
            routeInfo.coords && routeInfo.coords.length
              ? routeInfo.coords
              : [
                  [lat, lng],
                  [z.lat, z.lon],
                ],
          dist_km: dist,
          score: Number(z.final_score ?? 0),
        })
      }

      calculated.sort((a, b) => Number(b.score ?? 0) - Number(a.score ?? 0))
      setTop3(calculated)
      setRoutes3(calculated.map((x) => x.coords))

      const allCoords = calculated.flatMap((x) => x.coords || []).filter(Boolean)
      if (allCoords.length >= 2) {
        const bounds = L.latLngBounds(allCoords)
        map.fitBounds(bounds, { padding: [50, 50] })
      }
    })()
  }, [driverPos, zones, map, setTop3, setRoutes3, setActiveRoute, setPlayRoute, setPlayingKey])

  return null
}

// ====== 主頁面 ======
export default function DriverPage({ onBack, driverId, drivers, orders, simulateVehicles }) {
  const [zones, setZones] = useState([])
  const [driverPos, setDriverPos] = useState(null)
  const [top3, setTop3] = useState([])

  const [activeRoute, setActiveRoute] = useState(null)
  const [routes3, setRoutes3] = useState([])
  const [playRoute, setPlayRoute] = useState(null)
  const [playingKey, setPlayingKey] = useState('init')

  // ✅ 熱點載入：res.ok + retry（避免 model not ready 時 zones 永遠空）
  useEffect(() => {
    let cancelled = false
    let timer = null

    const load = async () => {
      try {
        const res = await apiFetch('/api/hotspots', { query: { n: 255 }, timeoutMs: 12000 })
        if (!res.ok) {
          const txt = await res.text().catch(() => '')
          throw new Error(`hotspots ${res.status}: ${txt}`)
        }
        const d = await res.json().catch(() => ({}))
        const rows = d.rows || []
        const formatted = rows
          .map((r) => {
            const lat = Number(r.lat_wgs ?? r.lat)
            const lon = Number(r.lon_wgs ?? r.lon)
            if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null
            return {
              ...r,
              lat,
              lon,
              pred_rides: Number(r.pred_rides ?? 0),
              D: Number(r.D ?? 0),
              C: Number(r.C ?? 0),
              DRS: Number(r.DRS ?? 0),
              final_score: Number(r.final_score ?? 0),
              used_hour: r.used_hour ?? '',
            }
          })
          .filter(Boolean)

        if (!cancelled) setZones(formatted)
      } catch (e) {
        console.warn('load hotspots failed, retry...', e)
        if (!cancelled) timer = setTimeout(load, 1500)
      }
    }

    load()
    return () => {
      cancelled = true
      if (timer) clearTimeout(timer)
    }
  }, [])

  // 2. 找出我的活躍訂單
  const myActiveOrder = useMemo(() => {
    if (!driverId || !orders) return null
    return orders.find((o) => sameId(getOrderDriverId(o), driverId) && isActiveStatus(o.status))
  }, [orders, driverId])

  // 4. 找出我的司機位置
  const myStaticPos = useMemo(() => {
    const me = drivers?.find((d) => sameId(d.id, driverId))
    if (me && Number.isFinite(Number(me.lat)) && Number.isFinite(Number(me.lng))) {
      return { lat: Number(me.lat), lng: Number(me.lng) }
    }
    return null
  }, [drivers, driverId])

  // ✅ driverPos 來源：drivers 裡的定位
  useEffect(() => {
    if (!myStaticPos) return
    setDriverPos((prev) => {
      if (!prev) return myStaticPos
      const dLat = Math.abs(prev.lat - myStaticPos.lat)
      const dLng = Math.abs(prev.lng - myStaticPos.lng)
      if (dLat > 1e-6 || dLng > 1e-6) return myStaticPos
      return prev
    })
  }, [myStaticPos])

  // 3. 計算我的活躍訂單路線：✅ 改成 driver -> pickup -> dropoff（避免 pickup 冒出一台）
  const [myOrderRoute, setMyOrderRoute] = useState(null)
  useEffect(() => {
    let cancelled = false

    const run = async () => {
      if (!myActiveOrder) {
        setMyOrderRoute(null)
        return
      }
      const p = myActiveOrder.pickupLocation
      const d = myActiveOrder.dropoffLocation
      if (!p || !d) {
        setMyOrderRoute(null)
        return
      }

      const s = myStaticPos
      const startLat = Number.isFinite(Number(s?.lat)) ? s.lat : p.lat
      const startLng = Number.isFinite(Number(s?.lng)) ? s.lng : p.lng

      const r1 = await getRoute(startLat, startLng, p.lat, p.lng) // driver -> pickup
      const r2 = await getRoute(p.lat, p.lng, d.lat, d.lng)       // pickup -> dropoff

      const c1 = r1.coords || []
      const c2 = r2.coords || []
      const merged =
        c1.length && c2.length ? [...c1, ...c2.slice(1)] : (c1.length ? c1 : c2)

      if (!cancelled) setMyOrderRoute(merged)
    }

    run()
    return () => { cancelled = true }
  }, [myActiveOrder, myStaticPos])

  // 5. 顯示路線
  const showRoute = (routeCoords) => setActiveRoute(routeCoords)

  // 6. 點卡片：顯示路線 + 播放小車
  const playToZone = (routeCoords) => {
    if (!routeCoords || routeCoords.length < 2) return
    setActiveRoute(routeCoords)
    setPlayRoute(routeCoords)
    setPlayingKey(String(Date.now()))
  }

  const radiusFor = useMemo(() => computeRadiusScaler(zones), [zones])

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh' }}>
      <div style={{ flex: 2, position: 'relative' }}>
        <MapContainer center={DEFAULT_CENTER} zoom={ZOOM_LEVEL} style={{ width: '100%', height: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

          <AutoRecommendController
            driverPos={driverPos}
            zones={zones}
            setTop3={setTop3}
            setRoutes3={setRoutes3}
            setActiveRoute={setActiveRoute}
            setPlayRoute={setPlayRoute}
            setPlayingKey={setPlayingKey}
          />

          {/* 熱點圈圈 */}
          {zones.map((z, i) => {
            const fs = Number(z.final_score ?? 0)
            const color = colorByFinalScore(fs)
            const radius = radiusFor(z.pred_rides)

            return (
              <CircleMarker
                key={i}
                center={[z.lat, z.lon]}
                radius={radius}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.6, weight: 1 }}
              >
                <Popup>
                  <strong>{z.Zone}</strong><br />
                  Borough: {z.Borough}<br />
                  pred_rides: {Number(z.pred_rides ?? 0).toFixed(3)}<br />
                  final_score: {Number(z.final_score ?? 0).toFixed(2)}<br />
                  D: {Number(z.D ?? 0)} / C: {Number(z.C ?? 0)}<br />
                  DRS: {Number(z.DRS ?? 0).toFixed(3)}
                  {z.used_hour ? (<><br /><small>311 時段: {z.used_hour}</small></>) : null}
                </Popup>
              </CircleMarker>
            )
          })}

          {/* ✅ 修正：沒接單時才畫靜態車；接單後只留 CarRuntimeLayer 那台 */}
          {myStaticPos && !myActiveOrder && (
            <Marker position={[myStaticPos.lat, myStaticPos.lng]} icon={makeTaxiIcon()} opacity={0.95} />
          )}

          {/* 接單時會動的車 */}
          {myActiveOrder && myOrderRoute && (
            <CarRuntimeLayer
              order={myActiveOrder}
              routeCoords={myOrderRoute}
              simulateVehicles={simulateVehicles}
              driverPos={myStaticPos}
            />
          )}

          {/* 3 條推薦路線 */}
          {routes3?.length
            ? routes3.map((coords, idx) => {
                if (!coords || coords.length < 2) return null
                const colors = ['blue', 'green', 'purple']
                return (
                  <Polyline
                    key={`r3-${idx}`}
                    positions={coords}
                    pathOptions={{ color: colors[idx] || 'blue', weight: 5, opacity: 0.8 }}
                  />
                )
              })
            : null}

          {/* 只顯示單一路線 */}
          {activeRoute && <Polyline positions={activeRoute} pathOptions={{ color: 'blue', weight: 5, opacity: 0.8 }} />}

          {/* 推薦小車動畫 */}
          {playRoute && <RecommendationCarLayer playRouteCoords={playRoute} playingKey={playingKey} />}
        </MapContainer>

        <button
          onClick={onBack}
          style={{
            position: 'absolute',
            top: 20,
            right: 20,
            zIndex: 1000,
            padding: '10px 20px',
            background: '#333',
            color: 'white',
            border: 'none',
            borderRadius: 5,
            cursor: 'pointer',
            fontWeight: 'bold',
          }}
        >
          返回行進頁面
        </button>
      </div>

      <div style={{ flex: 1, borderLeft: '1px solid #ccc', padding: 20, overflowY: 'auto', background: '#f9f9f9' }}>
        <h2 style={{ marginTop: 0 }}>司機派車建議</h2>

        {myActiveOrder ? (
          <div style={{ padding: 15, background: '#e3f2fd', borderRadius: 8, marginBottom: 20, border: '1px solid #90caf9' }}>
            <h3 style={{ margin: 0, color: '#1565c0' }}>正在執行訂單 #{myActiveOrder.id}</h3>
            <p style={{ margin: '5px 0' }}>乘客: {myActiveOrder.customer}</p>
            <small>切換回行進頁面以查看導航詳情</small>
          </div>
        ) : (
          <>
            <p style={{ fontSize: 13, color: '#666' }}>
              系統會直接使用司機目前位置，自動依 <b>311 獎懲 final_score</b> 推薦 Top 3 區域；距離只作為參考與畫路線，不影響 final_score。
            </p>

            <div style={{ marginTop: 10, padding: 12, borderRadius: 10, background: '#fff', border: '1px solid #eee' }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>分數說明（311 獎懲機制）</div>
              <div style={{ fontSize: 12, color: '#555', lineHeight: 1.6 }}>
                • <b>D（需求面）</b>：例如 Noise / Loud Music / Party，代表「可能有人群聚集」。<br />
                • <b>C（阻礙面）</b>：例如 Illegal Parking / Blocked Driveway，代表「可能卡車、路況差」。<br />
                • <b>DRS</b>：<code style={{ background: '#f2f2f2', padding: '1px 4px', borderRadius: 4 }}>DRS = α·Norm(D) − β·Norm(C)</code>（示例 α=0.7、β=0.3）。<br />
                • <b>final_score</b>：將 DRS 再標準化成 0~100，<b>越高越推薦</b>。<br />
                • <b>公平性</b>：沒報案紀錄的區域，D/C 補 0，不會被亂扣分（疑責從無）。<br />
                • <b>距離</b>：僅作為參考與畫路線，不影響 final_score。
              </div>
            </div>
          </>
        )}

        <h3 style={{ fontSize: 16, borderBottom: '1px solid #eee', paddingBottom: 10, marginTop: 16 }}>
          當前司機位置：
          {driverPos ? (
            <span style={{ color: 'blue', fontSize: 14, marginLeft: 10 }}>
              {driverPos.lat.toFixed(4)}, {driverPos.lng.toFixed(4)}
            </span>
          ) : (
            <span style={{ color: 'red', fontSize: 14, marginLeft: 10 }}>尚未取得</span>
          )}
        </h3>

        <h3 style={{ fontSize: 16, marginTop: 10, borderBottom: '1px solid #eee', paddingBottom: 10 }}>
          推薦前 3 名區域
        </h3>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {top3.length === 0 && driverPos && <div style={{ padding: 20, textAlign: 'center' }}>計算中...</div>}

          {top3.map((item, idx) => {
            const z = item.info || {}
            const bg = idx === 0 ? '#ffe0b2' : idx === 1 ? '#e1f5fe' : '#e8f5e9'
            return (
              <div
                key={idx}
                onClick={() => playToZone(item.coords)}
                style={{
                  padding: 15,
                  borderRadius: 10,
                  border: '1px solid #ddd',
                  cursor: 'pointer',
                  background: bg,
                  boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.transform = 'translateY(-2px)')}
                onMouseLeave={(e) => (e.currentTarget.style.transform = 'translateY(0)')}
              >
                <div style={{ fontWeight: 'bold', fontSize: 16, marginBottom: 6 }}>
                  Top {idx + 1}: {z.Zone}
                </div>

                <div style={{ fontSize: 13, color: '#555', lineHeight: 1.65 }}>
                  行政區: {z.Borough}<br />
                  pred_rides: <b>{Number(z.pred_rides ?? 0).toFixed(3)}</b><br />
                  D(需求面): {Number(z.D ?? 0)}、C(阻礙面): {Number(z.C ?? 0)}<br />
                  DRS: {Number(z.DRS ?? 0).toFixed(3)}<br />
                  <span style={{ color: '#d32f2f' }}>
                    <b>final_score: {Number(z.final_score ?? 0).toFixed(2)}</b>
                  </span>
                  {z.used_hour ? (
                    <>
                      <br />
                      <span style={{ fontSize: 11, color: '#777' }}>（311 時段: {z.used_hour}）</span>
                    </>
                  ) : null}
                  <br />
                  距離: {Number(item.dist_km ?? 0).toFixed(2)} km
                </div>

                <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                  <button
                    onClick={(ev) => {
                      ev.stopPropagation()
                      showRoute(item.coords)
                    }}
                    style={{
                      padding: '6px 10px',
                      borderRadius: 8,
                      border: '1px solid #bbb',
                      background: '#fff',
                      cursor: 'pointer',
                      fontSize: 12,
                    }}
                  >
                    只顯示路線
                  </button>

                  <button
                    onClick={(ev) => {
                      ev.stopPropagation()
                      playToZone(item.coords)
                    }}
                    style={{
                      padding: '6px 10px',
                      borderRadius: 8,
                      border: '1px solid #333',
                      background: '#333',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: 12,
                    }}
                  >
                    播放小車
                  </button>
                </div>
              </div>
            )
          })}
        </div>

        <div style={{ marginTop: 12, fontSize: 11, color: '#777' }}>
          ※ 本頁為 Demo UI：展示「模型需求 pred_rides + 311 獎懲 final_score」的視覺化呈現。
        </div>
      </div>
    </div>
  )
}
