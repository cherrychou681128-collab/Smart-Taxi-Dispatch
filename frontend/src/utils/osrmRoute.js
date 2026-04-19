// src/utils/osrmRoute.js

// ===== 簡單快取：避免一直打 OSRM 造成超慢 =====
const routeCache = new Map()
const ROUTE_TTL_MS = 2 * 60 * 1000

function keyFromPoints(points) {
  return points
    .map(p => `${(+p.lat).toFixed(5)},${(+p.lng).toFixed(5)}`)
    .join('|')
}

// OSRM: points -> coords [[lat,lng],...], distKm
async function fetchOsrm(points, { signal } = {}) {
  if (!Array.isArray(points) || points.length < 2) {
    throw new Error('Need at least 2 points')
  }

  const key = keyFromPoints(points)
  const hit = routeCache.get(key)
  if (hit && Date.now() - hit.t < ROUTE_TTL_MS) {
    return hit.v
  }

  const coordStr = points.map(p => `${p.lng},${p.lat}`).join(';')
  const url =
    `https://router.project-osrm.org/route/v1/driving/${coordStr}` +
    `?overview=full&geometries=geojson`

  const res = await fetch(url, { signal })
  const data = await res.json()

  if (!data.routes || !data.routes.length) {
    throw new Error('No OSRM route')
  }

  const route = data.routes[0]
  const result = {
    coords: route.geometry.coordinates.map(([lon, lat]) => [lat, lon]),
    distKm: route.distance / 1000,
  }

  routeCache.set(key, { t: Date.now(), v: result })
  return result
}

// 兼容你原本的兩點版本
export async function getOsrmRoute(from, to, opts = {}) {
  return fetchOsrm([from, to], opts)
}

// ✅ 新增：多點（上車 -> stops... -> 目的地）
export async function getOsrmRouteMulti(points, opts = {}) {
  return fetchOsrm(points, opts)
}

// 距離 -> 預估車資（USD）
export function estimateFareUSD(distKm) {
  const base = 2.5
  const perKm = 1.5
  return base + perKm * Math.max(1, distKm)
}

// 顯示成 $（你原本寫法我保留，但建議之後改成英文函式名）
export function 預估車資(usd) {
  return usd
}
