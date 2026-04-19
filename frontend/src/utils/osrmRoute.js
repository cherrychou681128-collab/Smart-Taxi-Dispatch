const routeCache = new Map()
const ROUTE_TTL_MS = 2 * 60 * 1000

function keyFromPoints(points) {
  return points
    .map(p => `${(+p.lat).toFixed(5)},${(+p.lng).toFixed(5)}`)
    .join('|')
}

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

export async function getOsrmRoute(from, to, opts = {}) {
  return fetchOsrm([from, to], opts)
}

export async function getOsrmRouteMulti(points, opts = {}) {
  return fetchOsrm(points, opts)
}

export function estimateFareUSD(distKm) {
  const base = 2.5
  const perKm = 1.5
  return base + perKm * Math.max(1, distKm)
}

export function 預估車資(usd) {
  return usd
}
