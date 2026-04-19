//driver
import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import MapView from '../components/MapView.jsx'
import OrderList from '../components/OrderList.jsx'
import { t } from '../i18n'
import { apiFetch } from '../apiBase.js'

// --- 工具函數 ---
function normalizeVehicleType(value) {
  if (value == null) return null
  const s = String(value).trim().toUpperCase()
  if (!s) return null
  if (s === 'YELLOW' || s.includes('YELLOW')) return 'YELLOW'
  if (s === 'GREEN' || s.includes('GREEN')) return 'GREEN'
  if (s === 'FHV' || s.includes('FHV')) return 'FHV'
  if (s.includes('黃')) return 'YELLOW'
  if (s.includes('綠')) return 'GREEN'
  if (s.includes('多元')) return 'FHV'
  return null
}

function sameId(a, b) {
  const A = Number(a)
  const B = Number(b)
  return Number.isFinite(A) && Number.isFinite(B) && A === B
}

function isPendingStatus(status) {
  return String(status || '').trim().toLowerCase() === 'pending'
}

function normStatus(status) {
  return String(status || '').trim().toLowerCase()
}

function getAnyDriverId(o) {
  return o?.driverId ?? o?.assignedDriverId ?? o?.driver_id ?? null
}

function getPickupLoc(o) {
  const p = o?.pickupLocation
  if (p && Number.isFinite(Number(p.lat)) && Number.isFinite(Number(p.lng))) {
    return { lat: Number(p.lat), lng: Number(p.lng) }
  }
  const lat = Number(o?.pickupLat)
  const lng = Number(o?.pickupLng)
  if (Number.isFinite(lat) && Number.isFinite(lng)) return { lat, lng }
  return null
}

function makeOrderKey(o) {
  const id = Number(o?.id)
  if (!Number.isFinite(id)) return null
  const createdAt = o?.createdAt || o?.created_at || o?.updatedAt || o?.updated_at || ''
  return `${id}::${String(createdAt)}`
}

async function osrmDistanceKm(from, to) {
  const url = `https://router.project-osrm.org/route/v1/driving/${from.lng},${from.lat};${to.lng},${to.lat}?overview=false`
  const res = await fetch(url)
  if (!res.ok) throw new Error('OSRM error')
  const data = await res.json()
  if (!data.routes || !data.routes.length) throw new Error('OSRM no route')
  return data.routes[0].distance / 1000
}

function round6(x) {
  const n = Number(x)
  if (!Number.isFinite(n)) return null
  return Math.round(n * 1e6) / 1e6
}

async function mapLimit(items, limit, fn) {
  const results = new Array(items.length)
  let idx = 0

  async function worker() {
    while (idx < items.length) {
      const cur = idx++
      results[cur] = await fn(items[cur], cur)
    }
  }

  const workers = Array.from({ length: Math.max(1, limit) }, () => worker())
  await Promise.all(workers)
  return results
}

// ✅ 每位司機獨立的定位鎖
function driverLocLockKey(driverId) {
  return `driverLocConfirmed:${driverId ?? 'na'}`
}
function readLocConfirmed(driverId) {
  try {
    if (driverId == null) return false
    return localStorage.getItem(driverLocLockKey(driverId)) === '1'
  } catch {
    return false
  }
}
function writeLocConfirmed(driverId, val) {
  try {
    if (driverId == null) return
    localStorage.setItem(driverLocLockKey(driverId), val ? '1' : '0')
  } catch {}
}

function readPersistedDriverLoc(driverId) {
  try {
    if (driverId == null) return null
    const raw = localStorage.getItem(`driverLoc:${driverId}`)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    const lat = Number(parsed?.lat)
    const lng = Number(parsed?.lng)
    if (Number.isFinite(lat) && Number.isFinite(lng)) return { lat, lng }
    return null
  } catch {
    return null
  }
}

const DRIVER_LOC_TS_PREFIX = 'driverLocTs:'
const DRIVER_LAST_LOGIN_DRIVER_KEY = 'driverLastLoginDriverId'
const DRIVER_LOC_TTL_MS = 60 * 60 * 1000 // 1小時；你要 12 小時就改成 12 * 60 * 60 * 1000

function driverLocTsKey(driverId) {
  return `${DRIVER_LOC_TS_PREFIX}${driverId ?? 'na'}`
}

function readLocTimestamp(driverId) {
  try {
    if (driverId == null) return null
    const raw = localStorage.getItem(driverLocTsKey(driverId))
    const n = Number(raw)
    return Number.isFinite(n) ? n : null
  } catch {
    return null
  }
}

function writeLocTimestamp(driverId, ts = Date.now()) {
  try {
    if (driverId == null) return
    localStorage.setItem(driverLocTsKey(driverId), String(ts))
  } catch {}
}

function clearDriverLocationState(driverId) {
  try {
    if (driverId == null) return
    localStorage.removeItem(`driverLoc:${driverId}`)
    localStorage.removeItem(driverLocLockKey(driverId))
    localStorage.removeItem(driverLocTsKey(driverId))
  } catch {}
}

export default function DriverView({
  lang,
  drivers = [],
  orders,
  ordersWithLocations,
  loading,
  error,
  currentDriverId,
  setCurrentDriverId,
  acceptOrder,
  refresh,
  currentUser,
  onDriverLocationChange,
  simulateVehicles,
  onOpenAuth,
  onOrderCompleted,
  onCarPosChange,
  hotspots = [],
  showHotspots = true,
}) {
  const goAuth = () => {
    if (onOpenAuth) {
      onOpenAuth('driver', 'driver')
      return
    }
    window.location.href = `${window.location.pathname}?auth=1&role=driver`
  }

  const isLoggedIn = Boolean(currentUser?.username)

  const allOrdersFromProps = useMemo(() => {
    return Array.isArray(ordersWithLocations) && ordersWithLocations.length
      ? ordersWithLocations
      : orders || []
  }, [ordersWithLocations, orders])

  const myDriver = useMemo(() => {
    if (currentDriverId != null) {
      const byId = drivers.find(d => sameId(d?.id, currentDriverId))
      if (byId) return byId
    }
    const u = currentUser?.username
    if (u) {
      const byName = drivers.find(
        d => String(d?.name || d?.username || '').trim() === String(u).trim()
      )
      if (byName) return byName
    }
    return null
  }, [drivers, currentDriverId, currentUser])

  const effectiveDriverId = isLoggedIn
    ? (myDriver?.id ?? null)
    : (currentDriverId ?? null)

  const driverReady = !isLoggedIn || (myDriver && effectiveDriverId != null)

  useEffect(() => {
    if (!setCurrentDriverId || myDriver?.id == null) return
    if (currentDriverId == null || !sameId(currentDriverId, myDriver.id)) {
      setCurrentDriverId(myDriver.id)
    }
  }, [myDriver, currentDriverId, setCurrentDriverId])

  const myCarType = normalizeVehicleType(myDriver?.carType ?? currentUser?.carType ?? null)

  const [locConfirmed, setLocConfirmed] = useState(false)

  useEffect(() => {
  if (!isLoggedIn || effectiveDriverId == null) {
    setLocConfirmed(false)
    return
  }

  const now = Date.now()
  const lastLoginDriverId = localStorage.getItem(DRIVER_LAST_LOGIN_DRIVER_KEY)
  const currentDriverIdStr = String(effectiveDriverId)

  const isDifferentDriver =
    lastLoginDriverId != null && lastLoginDriverId !== currentDriverIdStr

  const ts = readLocTimestamp(effectiveDriverId)
  const isExpired = !ts || (now - ts > DRIVER_LOC_TTL_MS)

  if (isDifferentDriver) {
    clearDriverLocationState(effectiveDriverId)
    setLocConfirmed(false)
  } else if (isExpired) {
    clearDriverLocationState(effectiveDriverId)
    setLocConfirmed(false)
  } else {
    setLocConfirmed(readLocConfirmed(effectiveDriverId))
  }

  localStorage.setItem(DRIVER_LAST_LOGIN_DRIVER_KEY, currentDriverIdStr)
}, [effectiveDriverId, isLoggedIn])

  const [selectedOrderId, setSelectedOrderId] = useState(null)
  const [orderDistances, setOrderDistances] = useState({})
  const [completedNotice, setCompletedNotice] = useState('')
  const completedSeenRef = useRef(new Set())

  useEffect(() => {
    setSelectedOrderId(null)
  }, [effectiveDriverId, isLoggedIn])

  // ✅ 修正：
  // 第一次點地圖定位才需要上鎖
  // 之後完成訂單 / 行程同步更新位置，不能被 locConfirmed 擋掉
  const handleDriverLocationChange = useCallback(
    p => {
      if (!driverReady) return
      if (!p || effectiveDriverId == null) return
      if (!sameId(p.id, effectiveDriverId)) return

      if (!locConfirmed) {
  setLocConfirmed(true)
  writeLocConfirmed(effectiveDriverId, true)
}

writeLocTimestamp(effectiveDriverId, Date.now())
onDriverLocationChange?.(p)
    },
    [onDriverLocationChange, effectiveDriverId, locConfirmed, driverReady]
  )

  // ✅ 新增：重設定位（不影響原流程，純 debug / UX）
  const resetMyLocation = useCallback(() => {
  if (effectiveDriverId == null) return
  clearDriverLocationState(effectiveDriverId)
  setLocConfirmed(false)
}, [effectiveDriverId])

  // ✅ 修正：
  // 每次顯示司機位置優先吃 localStorage 的最新 driverLoc
  // 這樣完成訂單後的新終點就會直接成為下一次起點
  const myDriverLoc = useMemo(() => {
    if (!locConfirmed || effectiveDriverId == null) return null

    const persisted = readPersistedDriverLoc(effectiveDriverId)
    if (persisted) return persisted

    if (!myDriver) return null

    const lat = Number(myDriver.lat)
    const lng = Number(myDriver.lng)
    if (Number.isFinite(lat) && Number.isFinite(lng)) {
      return { lat, lng }
    }

    return null
  }, [myDriver, locConfirmed, effectiveDriverId])

  const visibleDrivers = useMemo(() => {
    if (!driverReady || !locConfirmed || !myDriverLoc) return []
    return [{
      ...myDriver,
      id: effectiveDriverId,
      lat: myDriverLoc.lat,
      lng: myDriverLoc.lng,
    }]
  }, [driverReady, locConfirmed, myDriver, myDriverLoc, effectiveDriverId])

  const myActiveOrder = useMemo(() => {
    if (!driverReady || effectiveDriverId == null) return null

    const ACTIVE = new Set([
      'assigned',
      'accepted',
      'en_route',
      'enroute',
      'picked_up',
      'in_progress',
      'on_trip',
      'ongoing',
    ])

    const candidates = allOrdersFromProps
      .filter(o => o && sameId(getAnyDriverId(o), effectiveDriverId))
      .filter(o => ACTIVE.has(normStatus(o.status)))

    if (!candidates.length) return null

    candidates.sort(
      (a, b) =>
        (b.updatedAt ? Date.parse(b.updatedAt) : 0) -
        (a.updatedAt ? Date.parse(a.updatedAt) : 0)
    )
    return candidates[0]
  }, [allOrdersFromProps, effectiveDriverId, driverReady])

  useEffect(() => {
    if (myActiveOrder?.id != null) {
      setSelectedOrderId(myActiveOrder.id)
    }
  }, [myActiveOrder])

  const pendingHash = useMemo(() => {
    const pending = allOrdersFromProps
      .filter(o => isPendingStatus(o?.status))
      .filter(o => getAnyDriverId(o) == null)
      .map(o => `${o.id}:${round6(getPickupLoc(o)?.lat)}`)
      .join('|')

    return `${round6(myDriverLoc?.lat)}::${pending}`
  }, [allOrdersFromProps, myDriverLoc])

  useEffect(() => {
    let cancelled = false

    async function computeDistances() {
      if (!myDriverLoc) return

      const pending = allOrdersFromProps.filter(
        o => isPendingStatus(o?.status) && getAnyDriverId(o) == null
      )
      if (!pending.length) return

      const results = await mapLimit(pending, 4, async o => {
        const p = getPickupLoc(o)
        if (!p || cancelled) return null
        try {
          return { id: o.id, dKm: await osrmDistanceKm(myDriverLoc, p) }
        } catch {
          return null
        }
      })

      if (!cancelled) {
        const distMap = {}
        results.forEach(r => {
          if (r) distMap[r.id] = r.dKm
        })
        setOrderDistances(distMap)
      }
    }

    computeDistances()
    return () => { cancelled = true }
  }, [pendingHash, allOrdersFromProps, myDriverLoc])

  const pendingOrders = useMemo(() => {
    return allOrdersFromProps
      .filter(o => isPendingStatus(o?.status) && getAnyDriverId(o) == null)
      .filter(o => !myCarType || normalizeVehicleType(o?.vehicleType) === myCarType)
      .map(o => {
        const d = orderDistances[o.id]
        const price = o.estimatedFare || o.price || null
        return {
          ...o,
          pickupLocation: getPickupLoc(o),
          dispatchScore: d ? Math.max(1, Math.min(10, Math.round(11 - d))) : null,
          driverDistanceKm: d || null,
          price,
        }
      })
  }, [allOrdersFromProps, myCarType, orderDistances])

  const displayOrders = useMemo(() => {
    if (myActiveOrder) {
      return [myActiveOrder, ...pendingOrders.filter(o => o.id !== myActiveOrder.id)]
    }
    return pendingOrders
  }, [myActiveOrder, pendingOrders])

  const mapOrders = useMemo(() => {
    if (!driverReady || !locConfirmed || !myDriverLoc) return []

    let baseOrder = null
    if (myActiveOrder) {
      baseOrder = myActiveOrder
    } else if (selectedOrderId) {
      baseOrder = pendingOrders.find(o => o.id === selectedOrderId) || null
    } else {
      baseOrder =
        [...pendingOrders].sort(
          (a, b) => (a.driverDistanceKm || Infinity) - (b.driverDistanceKm || Infinity)
        )[0] || null
    }

    if (!baseOrder) return []

    const pickup = getPickupLoc(baseOrder)
    if (!pickup) return []

    const { polyline, route, directions, path, ...cleanOrder } = baseOrder

    const linearStops = [
      {
        lat: round6(myDriverLoc.lat),
        lng: round6(myDriverLoc.lng),
        text: '我的位置',
        type: 'driver',
      },
      {
        lat: round6(pickup.lat),
        lng: round6(pickup.lng),
        text: '乘客上車',
        type: 'pickup',
      },
    ]

    const originalStops = (cleanOrder.stops || []).filter(s => {
      if (!s) return false
      const type = String(s.type || '').toLowerCase()
      if (type === 'driver' || type === 'pickup') return false

      const lat = Number(s.lat)
      const lng = Number(s.lng)
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) return false

      const isAtPickup =
        Math.abs(lat - pickup.lat) < 0.0001 &&
        Math.abs(lng - pickup.lng) < 0.0001

      return !isAtPickup
    })

    const routeSig =
      `${effectiveDriverId}:${baseOrder.id}:` +
      `${round6(myDriverLoc.lat)},${round6(myDriverLoc.lng)}->` +
      `${round6(pickup.lat)},${round6(pickup.lng)}`

    return [
      {
        ...cleanOrder,
        driverId: effectiveDriverId,
        stops: [...linearStops, ...originalStops],
        _forceUpdate: routeSig,
      },
    ]
  }, [locConfirmed, myDriverLoc, myActiveOrder, selectedOrderId, pendingOrders, effectiveDriverId, driverReady])

const shouldFollowCar = useMemo(() => {
  if (!locConfirmed || !myDriverLoc) return false
  return Boolean(
    myActiveOrder ||
    selectedOrderId != null ||
    mapOrders.length > 0
  )
}, [locConfirmed, myDriverLoc, myActiveOrder, selectedOrderId, mapOrders])

  const myCompletedOrders = useMemo(() => {
    if (effectiveDriverId == null) return []
    const done = allOrdersFromProps
      .filter(o => o && sameId(getAnyDriverId(o), effectiveDriverId))
      .filter(o => normStatus(o.status) === 'completed')

    done.sort(
      (a, b) =>
        (b.completedAt ? Date.parse(b.completedAt) : 0) -
        (a.completedAt ? Date.parse(a.completedAt) : 0)
    )
    return done
  }, [allOrdersFromProps, effectiveDriverId])

  const handleAcceptAndSync = useCallback(async (orderId) => {
  setSelectedOrderId(orderId)
  await acceptOrder(orderId)
  await refresh?.()
}, [acceptOrder, refresh])

  useEffect(() => {
    if (!myCompletedOrders.length) return

    let newlyCompleted = null
    for (const o of myCompletedOrders) {
      const key = makeOrderKey(o) || `id:${o?.id ?? ''}`
      if (!completedSeenRef.current.has(key)) {
        newlyCompleted = o
        completedSeenRef.current.add(key)
        break
      }
    }

    if (!newlyCompleted) return

    const price = newlyCompleted.estimatedFare || newlyCompleted.price || null
    setCompletedNotice(
      `此筆訂單已完成（#${newlyCompleted.id}），金額：${price != null ? `$${price.toFixed(2)}` : '未知'}`
    )
    setTimeout(() => setCompletedNotice(''), 7000)
  }, [myCompletedOrders])

  return (
    <section className="map-section">
      <div className="map-wrapper">
        <MapView
key={`driver-map:${driverReady ? effectiveDriverId : 'pending'}`}          lang={lang}
          drivers={driverReady ? visibleDrivers : []}
          orders={driverReady ? mapOrders : []}
          mode="driver"
          currentDriverId={driverReady ? effectiveDriverId : null}
          onDriverLocationChange={handleDriverLocationChange}
          driverClickEnabled={driverReady && !locConfirmed}
          simulateVehicles={simulateVehicles}
          onOrderCompleted={onOrderCompleted}
          onCarPosChange={onCarPosChange}
          usePersistedDriverLoc={false}
          followActiveCar={shouldFollowCar}
          previewEnabled={false}
          rotateMapWithHeading={true}
          hotspots={showHotspots ? hotspots : []}
        />
      </div>

      <aside className="side-panel">
        <div className="panel-inner">
          <h1 className="panel-title">{t(lang, 'driverMode')}</h1>

          <div className="field-label">{t(lang, 'currentDriverLabel')}</div>
          {isLoggedIn ? (
            <div className="current-driver-box">{currentUser.username}</div>
          ) : (
            <button
              type="button"
              className="current-driver-box"
              onClick={goAuth}
              style={{ cursor: 'pointer' }}
            >
              請先登入
            </button>
          )}

          {completedNotice && <div className="ub-toast ub-toast--success">{completedNotice}</div>}

          {isLoggedIn && !locConfirmed && (
            <div className="ub-toast ub-toast--warn" style={{ marginBottom: 12 }}>
              ⚠️ 請先在地圖上點擊一下，設定您的初始位置。
            </div>
          )}

          {isLoggedIn && effectiveDriverId != null && (
            <div style={{ marginBottom: 12 }}>
              <button
                type="button"
                className="ghost-btn"
                onClick={resetMyLocation}
                style={{ width: '100%' }}
              >
                重設定位（debug）
              </button>
            </div>
          )}

          <section className="orders-block" style={{ marginTop: 14 }}>
            <div className="orders-header">
              <h3>{t(lang, 'ordersTitleDriver')}</h3>
              <button className="ghost-btn" type="button" onClick={refresh} disabled={loading}>
                {t(lang, 'refresh')}
              </button>
            </div>

            <OrderList
              lang={lang}
              orders={locConfirmed ? displayOrders : []}
              isDriverView
              onAcceptOrder={handleAcceptAndSync}
              drivers={drivers}
              currentDriverId={driverReady ? effectiveDriverId : null}
              selectedOrderId={selectedOrderId}
              onSelectOrder={id => setSelectedOrderId(id)}
            />

            {loading && (
              <div className="auth-hint" style={{ marginTop: 8 }}>
                {t(lang, 'loading')}
              </div>
            )}
            {error && <div className="error-box">{error}</div>}
          </section>

          {isLoggedIn && myCompletedOrders.length > 0 && (
            <section className="orders-block" style={{ marginTop: 14 }}>
              <h3>已完成訂單</h3>
              {myCompletedOrders.slice(0, 5).map(o => (
                <div
                  key={o.id}
                  style={{
                    border: '1px solid rgba(0,0,0,0.1)',
                    borderRadius: 12,
                    padding: 12,
                    marginBottom: 8,
                    background: '#fff',
                  }}
                >
                  <div style={{ fontWeight: 800 }}>
                    訂單 #{o.id} <span style={{ color: 'green' }}>已完成</span>
                  </div>
                  <div style={{ fontSize: 13 }}>
                    上車：{o.pickup}
                    <br />
                    下車：{o.dropoff}
                  </div>
                </div>
              ))}
            </section>
          )}
        </div>
      </aside>
    </section>
  )
}