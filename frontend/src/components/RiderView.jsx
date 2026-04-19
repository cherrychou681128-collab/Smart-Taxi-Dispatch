//rider
import { useEffect, useMemo, useRef, useState, useCallback } from 'react'
import MapView from '../components/MapView.jsx'
import OrderList from '../components/OrderList.jsx'
import { t } from '../i18n'

function normalizeGeocodeList(data) {
  if (!Array.isArray(data)) return []
  return data
    .map(x => {
      const label = x?.label ?? x?.display_name ?? x?.name ?? ''
      const lat = Number(x?.lat)
      const lng = Number(x?.lng ?? x?.lon)
      return { label, lat, lng }
    })
    .filter(x => x.label && Number.isFinite(x.lat) && Number.isFinite(x.lng))
}

function isActiveAssignedStatus(status) {
  const s = String(status || '').trim().toLowerCase()
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

function getAnyDriverId(o) {
  return o?.driverId ?? o?.assignedDriverId ?? o?.driver_id ?? null
}

export default function RiderView({
  lang,
  drivers,
  orders,
  ordersWithLocations,
  loading,
  error,
  createOrder,
  refresh,
  currentUser,
  simulateVehicles,

  initialDraft,
  onDraftChange,
  onOpenAuth,

  onOrderCompleted,
  onCarPosChange,
}) {
  const didInitDraft = useRef(false)

  const [pickupText, setPickupText] = useState('')
  const [dropoffText, setDropoffText] = useState('')

  const [pickupLoc, setPickupLoc] = useState(null)
  const [dropoffLoc, setDropoffLoc] = useState(null)

  const [pickupLocked, setPickupLocked] = useState(false)
  const [dropoffLocked, setDropoffLocked] = useState(false)

  const [pickupDirty, setPickupDirty] = useState(false)
  const [dropoffDirty, setDropoffDirty] = useState(false)

  const [pickupSuggestions, setPickupSuggestions] = useState([])
  const [dropoffSuggestions, setDropoffSuggestions] = useState([])

  const [stops, setStops] = useState([])

  const [fareOptions, setFareOptions] = useState(null)
  const [fareError, setFareError] = useState('')
  const [lastDistanceKm, setLastDistanceKm] = useState(null)
  const [selectedVehicle, setSelectedVehicle] = useState(null)
  const [resolvedStops, setResolvedStops] = useState([])
  const [previewRouteCoords, setPreviewRouteCoords] = useState(null)
  const [selectedOrderId, setSelectedOrderId] = useState(null)
  const [completedOrderIds, setCompletedOrderIds] = useState(() => new Set())

  const [toast, setToast] = useState('')
  const toastTimerRef = useRef(null)

  const userManuallySelectedOrder = useRef(false)

  const [composerLocked, setComposerLocked] = useState(false)
  const [composeMode, setComposeMode] = useState(false)

  const stopControllersRef = useRef({})

  useEffect(() => {
    return () => {
      try {
        if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
      } catch {}
    }
  }, [])

  const showToast = (msg, ms = 2800) => {
    try {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
    } catch {}
    setToast(msg)
    toastTimerRef.current = setTimeout(() => setToast(''), ms)
  }

  const resetFarePanel = () => {
    setFareOptions(null)
    setFareError('')
    setLastDistanceKm(null)
    setSelectedVehicle(null)
    setResolvedStops([])
  }

  const resetComposerInputs = () => {
    try {
      Object.values(stopControllersRef.current || {}).forEach(entry => {
        if (entry?.timer) clearTimeout(entry.timer)
        if (entry?.controller) entry.controller.abort()
      })
    } catch {}
    stopControllersRef.current = {}

    setPickupText('')
    setDropoffText('')
    setPickupLoc(null)
    setDropoffLoc(null)

    setPickupLocked(false)
    setDropoffLocked(false)
    setPickupDirty(false)
    setDropoffDirty(false)

    setPickupSuggestions([])
    setDropoffSuggestions([])

    setStops([])
    resetFarePanel()
  }

  useEffect(() => {
    if (didInitDraft.current) return
    didInitDraft.current = true
    if (!initialDraft) return

    setPickupText(initialDraft.pickupText || '')
    setDropoffText(initialDraft.dropoffText || '')
    setPickupLoc(initialDraft.pickupLoc || null)
    setDropoffLoc(initialDraft.dropoffLoc || null)

    setPickupLocked(Boolean(initialDraft.pickupLoc))
    setDropoffLocked(Boolean(initialDraft.dropoffLoc))
    setPickupDirty(false)
    setDropoffDirty(false)
    setPickupSuggestions([])
    setDropoffSuggestions([])

    const draftStops = Array.isArray(initialDraft.stops) ? initialDraft.stops : []
    setStops(
      draftStops.map(s => ({
        text: s?.text || s?.label || '',
        loc: s?.loc
          ? { lat: Number(s.loc.lat), lng: Number(s.loc.lng) }
          : s?.lat && s?.lng
          ? { lat: Number(s.lat), lng: Number(s.lng) }
          : null,
        locked: Boolean(s?.loc || (s?.lat && s?.lng) || s?.locked),
        dirty: false,
        suggestions: [],
      }))
    )

    setComposerLocked(Boolean(initialDraft.composerLocked))
    setComposeMode(Boolean(initialDraft.composeMode))
    setSelectedOrderId(initialDraft.selectedOrderId ?? null)
  }, [initialDraft])

  useEffect(() => {
    if (!onDraftChange) return

    const payload = {
      pickupText,
      dropoffText,
      pickupLoc,
      dropoffLoc,
      stops: stops.map(s => ({
        text: s.text,
        loc: s.loc,
        locked: s.locked,
      })),
      composerLocked,
      composeMode,
      selectedOrderId,
    }

    try {
      onDraftChange(prev => ({ ...(prev || {}), ...payload }))
    } catch {
      onDraftChange(payload)
    }
  }, [pickupText, dropoffText, pickupLoc, dropoffLoc, stops, composerLocked, composeMode, selectedOrderId])

  const passengerOrdersWithLocAll = useMemo(() => {
    return Array.isArray(ordersWithLocations) ? ordersWithLocations : []
  }, [ordersWithLocations])

  const defaultOrder = useMemo(() => {
    if (!passengerOrdersWithLocAll.length) return null
    const isDone = o => completedOrderIds?.has?.(o?.id)

    const notDone = passengerOrdersWithLocAll
      .filter(o => !isDone(o))
      .sort(
        (a, b) =>
          (Date.parse(b.updatedAt || b.createdAt || 0) || b.id) - (Date.parse(a.updatedAt || a.createdAt || 0) || a.id)
      )

    if (notDone[0]) return notDone[0]

    const sorted = [...passengerOrdersWithLocAll].sort(
      (a, b) =>
        (Date.parse(b.updatedAt || b.createdAt || 0) || b.id) - (Date.parse(a.updatedAt || a.createdAt || 0) || a.id)
    )
    return sorted[0] || null
  }, [passengerOrdersWithLocAll, completedOrderIds])

const syncedActivePassengerOrder = useMemo(() => {
  const mine = passengerOrdersWithLocAll.filter(o => !completedOrderIds.has(o?.id))

  const activeAssigned = mine
    .filter(o => getAnyDriverId(o) != null && isActiveAssignedStatus(o.status))
    .sort((a, b) => {
      const ta = Date.parse(a?.updatedAt || a?.createdAt || 0) || 0
      const tb = Date.parse(b?.updatedAt || b?.createdAt || 0) || 0
      return tb - ta
    })

  return activeAssigned[0] || null
}, [passengerOrdersWithLocAll, completedOrderIds])

useEffect(() => {
  if (composeMode) return

  // 只要有司機已接單 / active order，乘客端強制同步那張
  if (syncedActivePassengerOrder?.id != null) {
    setSelectedOrderId(syncedActivePassengerOrder.id)
    userManuallySelectedOrder.current = false
    return
  }

  if (userManuallySelectedOrder.current) return
  if (defaultOrder?.id != null) setSelectedOrderId(defaultOrder.id)
}, [syncedActivePassengerOrder, defaultOrder, composeMode])

  const selectedOrder = useMemo(() => {
    if (selectedOrderId == null) return null
    return passengerOrdersWithLocAll.find(o => o.id === selectedOrderId) || null
  }, [passengerOrdersWithLocAll, selectedOrderId])

const previewStopsResolved = useMemo(() => {
  return stops
    .filter(s => s?.loc && Number.isFinite(s.loc.lat) && Number.isFinite(s.loc.lng))
    .map(s => ({ label: s.text, lat: s.loc.lat, lng: s.loc.lng }))
}, [stops])

const previewPointSequence = useMemo(() => {
  const pts = []

  if (pickupLoc && Number.isFinite(pickupLoc.lat) && Number.isFinite(pickupLoc.lng)) {
    pts.push({
      kind: 'pickup',
      label: pickupText || '起點',
      lat: pickupLoc.lat,
      lng: pickupLoc.lng,
    })
  }

  for (const s of previewStopsResolved) {
    pts.push({
      kind: 'stop',
      label: s.label,
      lat: s.lat,
      lng: s.lng,
    })
  }

  if (dropoffLoc && Number.isFinite(dropoffLoc.lat) && Number.isFinite(dropoffLoc.lng)) {
    pts.push({
      kind: 'dropoff',
      label: dropoffText || '終點',
      lat: dropoffLoc.lat,
      lng: dropoffLoc.lng,
    })
  }

  return pts
}, [pickupLoc, pickupText, previewStopsResolved, dropoffLoc, dropoffText])

useEffect(() => {
  let cancelled = false

  async function loadPreviewRoute() {
    if (previewPointSequence.length < 2) {
      setPreviewRouteCoords(null)
      return
    }

    try {
      const coords = await osrmRouteGeometry(
        previewPointSequence.map(p => ({ lat: p.lat, lng: p.lng }))
      )
      if (!cancelled) setPreviewRouteCoords(coords)
    } catch {
      if (!cancelled) {
        setPreviewRouteCoords(
          previewPointSequence.map(p => ({ lat: p.lat, lng: p.lng }))
        )
      }
    }
  }

  loadPreviewRoute()
  return () => {
    cancelled = true
  }
}, [previewPointSequence])

const previewWaypoints = useMemo(() => {
  if (previewPointSequence.length < 2) return null
  if (Array.isArray(previewRouteCoords) && previewRouteCoords.length >= 2) {
    return previewRouteCoords
  }
  return previewPointSequence.map(p => ({ lat: p.lat, lng: p.lng }))
}, [previewPointSequence, previewRouteCoords])

const shouldShowPreview = useMemo(() => {
  return composeMode && previewPointSequence.length >= 1
}, [composeMode, previewPointSequence.length])

const previewMarkers = useMemo(() => {
  return {
    pickup: pickupLoc,
    dropoff: dropoffLoc,
    stops: previewStopsResolved,
  }
}, [pickupLoc, dropoffLoc, previewStopsResolved])

  async function geocodeOnce(text) {
    if (!text || !text.trim()) return null
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(text.trim())}`)
    if (!res.ok) return null
    const data = normalizeGeocodeList(await res.json().catch(() => []))
    if (!data.length) return null
    return { lat: data[0].lat, lng: data[0].lng, label: data[0].label }
  }

  function distanceKm(a, b) {
    const toRad = d => (d * Math.PI) / 180
    const R = 6371
    const dLat = toRad(b.lat - a.lat)
    const dLng = toRad(b.lng - a.lng)
    const la1 = toRad(a.lat)
    const la2 = toRad(b.lat)
    const s = Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLng / 2) ** 2
    const c = 2 * Math.atan2(Math.sqrt(s), Math.sqrt(1 - s))
    return R * c
  }

  async function osrmRouteDistanceKm(points) {
    if (!Array.isArray(points) || points.length < 2) return null
    const coordStr = points.map(p => `${p.lng},${p.lat}`).join(';')
    const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=false`
    const res = await fetch(url)
    if (!res.ok) throw new Error('OSRM error')
    const data = await res.json()
    if (!data.routes || !data.routes.length) throw new Error('OSRM no route')
    return data.routes[0].distance / 1000
  }
async function osrmRouteGeometry(points) {
  if (!Array.isArray(points) || points.length < 2) return null

  const coordStr = points.map(p => `${p.lng},${p.lat}`).join(';')
  const url = `https://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`

  const res = await fetch(url)
  if (!res.ok) throw new Error('OSRM geometry error')

  const data = await res.json()
  const coords = data?.routes?.[0]?.geometry?.coordinates
  if (!coords || coords.length < 2) throw new Error('OSRM geometry no route')

  return coords.map(([lng, lat]) => ({ lat, lng }))
}
  useEffect(() => {
    if (composerLocked) return
    if (pickupLocked || !pickupDirty) {
      setPickupSuggestions([])
      return
    }
    if (!pickupText || pickupText.trim().length < 2) {
      setPickupSuggestions([])
      return
    }

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/geocode?q=${encodeURIComponent(pickupText.trim())}`, { signal: controller.signal })
        if (!res.ok) return
        const data = normalizeGeocodeList(await res.json().catch(() => []))
        setPickupSuggestions(data)
      } catch {}
    }, 250)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [pickupText, pickupLocked, pickupDirty, composerLocked])

  useEffect(() => {
    if (composerLocked) return
    if (dropoffLocked || !dropoffDirty) {
      setDropoffSuggestions([])
      return
    }
    if (!dropoffText || dropoffText.trim().length < 2) {
      setDropoffSuggestions([])
      return
    }

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/geocode?q=${encodeURIComponent(dropoffText.trim())}`, { signal: controller.signal })
        if (!res.ok) return
        const data = normalizeGeocodeList(await res.json().catch(() => []))
        setDropoffSuggestions(data)
      } catch {}
    }, 250)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [dropoffText, dropoffLocked, dropoffDirty, composerLocked])

  const handleSelectPickup = item => {
    setPickupText(item.label)
    setPickupLoc({ lat: item.lat, lng: item.lng })
    setPickupLocked(true)
    setPickupDirty(false)
    setPickupSuggestions([])
    resetFarePanel()
  }

  const handleSelectDropoff = item => {
    setDropoffText(item.label)
    setDropoffLoc({ lat: item.lat, lng: item.lng })
    setDropoffLocked(true)
    setDropoffDirty(false)
    setDropoffSuggestions([])
    resetFarePanel()
  }

  const addStop = () => {
    if (composerLocked) return
    setStops(prev => [...prev, { text: '', loc: null, locked: false, dirty: false, suggestions: [] }])
    resetFarePanel()
  }

  const removeStop = index => {
    const entry = stopControllersRef.current[index]
    if (entry?.timer) clearTimeout(entry.timer)
    if (entry?.controller) entry.controller.abort()
    delete stopControllersRef.current[index]

    setStops(prev => prev.filter((_, i) => i !== index))
    resetFarePanel()
  }

  const updateStopText = (index, text) => {
    if (composerLocked) return

    setStops(prev => {
      const copy = [...prev]
      if (!copy[index]) return prev
      copy[index] = { ...copy[index], text, loc: null, locked: false, dirty: true, suggestions: [] }
      return copy
    })
    resetFarePanel()

    const trimmed = (text || '').trim()
    if (trimmed.length < 2) {
      const entry = stopControllersRef.current[index]
      if (entry?.timer) clearTimeout(entry.timer)
      if (entry?.controller) entry.controller.abort()
      stopControllersRef.current[index] = null
      return
    }

    const prevEntry = stopControllersRef.current[index]
    if (prevEntry?.timer) clearTimeout(prevEntry.timer)
    if (prevEntry?.controller) prevEntry.controller.abort()

    const controller = new AbortController()
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/geocode?q=${encodeURIComponent(trimmed)}`, { signal: controller.signal })
        if (!res.ok) return
        const data = normalizeGeocodeList(await res.json().catch(() => []))

        setStops(prev => {
          const copy = [...prev]
          if (!copy[index]) return prev
          if (copy[index].locked) return prev
          if (!copy[index].dirty) return prev
          if (copy[index].text !== text) return prev
          copy[index] = { ...copy[index], suggestions: data }
          return copy
        })
      } catch {}
    }, 250)

    stopControllersRef.current[index] = { timer, controller }
  }

  const handleSelectStopSuggestion = (index, item) => {
    const entry = stopControllersRef.current[index]
    if (entry?.timer) clearTimeout(entry.timer)
    if (entry?.controller) entry.controller.abort()
    stopControllersRef.current[index] = null

    setStops(prev => {
      const copy = [...prev]
      if (!copy[index]) return prev
      copy[index] = {
        ...copy[index],
        text: item.label,
        loc: { lat: item.lat, lng: item.lng },
        locked: true,
        dirty: false,
        suggestions: [],
      }
      return copy
    })
    resetFarePanel()
  }

  const handleCheckPrice = async () => {
    if (composerLocked) return

    if (!pickupText.trim() || !dropoffText.trim()) {
      alert(t(lang, 'needPickupDropoff'))
      return
    }

    if (!currentUser || currentUser.role !== 'passenger') {
      onOpenAuth?.('rider', 'passenger')
      return
    }

    resetFarePanel()

    try {
      let pLoc = pickupLoc
      let dLoc = dropoffLoc

      if (!pLoc) {
        const got = await geocodeOnce(pickupText)
        if (got) {
          pLoc = { lat: got.lat, lng: got.lng }
          setPickupLoc(pLoc)
        }
      }
      if (!dLoc) {
        const got = await geocodeOnce(dropoffText)
        if (got) {
          dLoc = { lat: got.lat, lng: got.lng }
          setDropoffLoc(dLoc)
        }
      }

      if (!pLoc || !dLoc) {
        setFareError(t(lang, 'cannotFindPickupOrDropoff'))
        return
      }

      const resolved = []
      for (const s of stops) {
        const label = (s.text || '').trim()
        if (!label) continue

        let loc = s.loc
        if (!loc) {
          const got = await geocodeOnce(label)
          if (got) loc = { lat: got.lat, lng: got.lng }
        }

        if (!loc) {
          setFareError(t(lang, 'cannotFindStop'))
          return
        }
        resolved.push({ label, lat: loc.lat, lng: loc.lng })
      }
      setResolvedStops(resolved)

      const points = [
        { lat: pLoc.lat, lng: pLoc.lng },
        ...resolved.map(s => ({ lat: s.lat, lng: s.lng })),
        { lat: dLoc.lat, lng: dLoc.lng },
      ]

      let totalDist = 0
      try {
        totalDist = await osrmRouteDistanceKm(points)
      } catch {
        for (let i = 0; i < points.length - 1; i++) totalDist += distanceKm(points[i], points[i + 1])
      }

      const distRounded = Math.round(totalDist * 10) / 10
      setLastDistanceKm(distRounded)

      if (totalDist > 80) {
        setFareError(t(lang, 'tripTooFar'))
        return
      }

      const baseFare = 2.5
      const perKm = 1.5
      const rawFare = baseFare + perKm * totalDist

      const estYellow = +(rawFare * 1.0).toFixed(2)
      const estGreen = +(rawFare * 0.9).toFixed(2)
      const estFhv = +(rawFare * 1.3).toFixed(2)

      setFareOptions([
        { type: 'YELLOW', label: t(lang, 'carTypeYellow'), price: estYellow },
        { type: 'GREEN', label: t(lang, 'carTypeGreen'), price: estGreen },
        { type: 'FHV', label: t(lang, 'carTypeFhv'), price: estFhv },
      ])
    } catch (e) {
      console.error(e)
      setFareError(t(lang, 'networkError'))
    }
  }

  const handleChooseFare = async option => {
    if (composerLocked) return

    if (!fareOptions || lastDistanceKm == null) {
      alert(t(lang, 'needPriceFirst'))
      return
    }
    if (!pickupLoc || !dropoffLoc) {
      alert(t(lang, 'needCoordsPrepared'))
      return
    }
    if (!currentUser || currentUser.role !== 'passenger') {
      onOpenAuth?.('rider', 'passenger')
      return
    }

    setSelectedVehicle(option.type)

    await createOrder(
      pickupText.trim(),
      dropoffText.trim(),
      pickupLoc,
      dropoffLoc,
      { distanceKm: lastDistanceKm, vehicleType: option.type, price: option.price },
      resolvedStops
    )

    await refresh?.()

    setComposerLocked(true)
    setComposeMode(false)
    userManuallySelectedOrder.current = false
    setSelectedOrderId(null)

    resetComposerInputs()
  }

  const handleOrderArrived = orderId => {
    showToast('司機已抵達上車點', 2500)
  }

  const handleOrderCompletedLocal = orderId => {
    setCompletedOrderIds(prev => {
      const next = new Set(prev)
      next.add(orderId)
      return next
    })
    showToast('訂單已完成，感謝您的搭乘~', 3500)
    onOrderCompleted?.(orderId)
  }

const ordersForMap = useMemo(() => {
  if (!composeMode) {
    const activeBase = syncedActivePassengerOrder || selectedOrder

    if (activeBase && !completedOrderIds.has(activeBase.id)) {
      return [activeBase]
    }
    return []
  }

  if (pickupLoc && dropoffLoc) {
    return [{
      id: 'preview-draft-id',
      pickupLat: pickupLoc.lat,
      pickupLng: pickupLoc.lng,
      dropoffLat: dropoffLoc.lat,
      dropoffLng: dropoffLoc.lng,
      stops: resolvedStops.length > 0 ? resolvedStops : previewStopsResolved,
      status: 'pending',
    }]
  }

  return []
}, [
  composeMode,
  syncedActivePassengerOrder,
  selectedOrder,
  completedOrderIds,
  pickupLoc,
  dropoffLoc,
  resolvedStops,
  previewStopsResolved,
])

  const loginBtnText =
    lang === 'en'
      ? 'Please log in'
      : lang === 'ja'
      ? 'ログインしてください'
      : lang === 'ko'
      ? '로그인 해주세요'
      : '請先登入'

  const startNextOrder = () => {
    setComposerLocked(false)
    setComposeMode(true)

    userManuallySelectedOrder.current = true
    setSelectedOrderId(null)

    resetComposerInputs()
  }

  // ✅ [修正重點] 核心邏輯：嚴格過濾司機
  // 1. 只有當前「顯示在畫面上」的訂單（ordersForMap）所對應的司機才顯示
  // 2. 或是與當前使用者相關且「非已完成狀態」的司機才顯示
  const visibleDrivers = useMemo(() => {
    if (!currentUser || !drivers || !drivers.length) return []

    // 取得當前畫面上正在活躍的所有訂單 ID (包含預覽單除外)
    const activeOnMapIds = new Set(ordersForMap.map(o => o.id));

    return drivers.filter(d => {
      // 找出這個司機目前正在服務的訂單
      const serviceOrder = orders.find(o => {
          const isMatched = (String(o.driverId) === String(d.id) || String(o.assignedDriverId) === String(d.id));
          const isMine = o.customer === currentUser.username;
          const isNotCompleted = !completedOrderIds.has(o.id);
          const isActiveOnMap = activeOnMapIds.has(o.id);
          
          // 狀態過濾：必須是進行中的狀態
          const s = String(o.status || '').toLowerCase();
          const isActiveStatus = ['assigned', 'accepted', 'en_route', 'enroute', 'picked_up', 'in_progress', 'on_trip', 'ongoing'].includes(s);

          return isMatched && isMine && isNotCompleted && isActiveStatus && isActiveOnMap;
      });

      return !!serviceOrder;
    });
  }, [drivers, orders, currentUser, completedOrderIds, ordersForMap])

  return (
    <section className="map-section">
      <div className="map-wrapper">
        <div style={{ width: '100%', height: '100%' }}>
          <MapView
            key={`rider-map:${syncedActivePassengerOrder?.id ?? selectedOrderId ?? (composeMode ? 'compose' : 'idle')}`}
            lang={lang}
            drivers={visibleDrivers}
            orders={ordersForMap}
            mode="passenger"
            currentDriverId={null}
            simulateVehicles={simulateVehicles}
            completedOrderIds={completedOrderIds}
            onOrderArrived={handleOrderArrived}
            onOrderCompleted={handleOrderCompletedLocal}
            previewEnabled={shouldShowPreview}
            previewWaypoints={previewWaypoints}
            previewMarkers={previewMarkers}
            onCarPosChange={onCarPosChange}
            followActiveCar={true}
            rotateMapWithHeading={true} 
          />
        </div>
      </div>

      <aside className="side-panel">
        <div className="panel-inner">
          <h1 className="panel-title">{t(lang, 'passengerMode')}</h1>

          {toast && (
            <div className="auth-hint" style={{ marginTop: 8, color: '#00e676', fontWeight: 700 }}>
              {toast}
            </div>
          )}

          <div className="field-label">{t(lang, 'currentPassengerLabel')}</div>
          <div className="current-driver-box">
            {currentUser?.username ? (
              currentUser.username
            ) : (
              <button type="button" className="ghost-btn" onClick={() => onOpenAuth?.('rider', 'passenger')}>
                {loginBtnText}
              </button>
            )}
          </div>

          {composerLocked ? (
            <button type="button" className="primary-btn" style={{ marginTop: 18, width: '100%' }} onClick={startNextOrder}>
              新增另一張訂單
            </button>
          ) : (
            <>
              <div className="field-label" style={{ marginTop: 24 }}>
                {t(lang, 'pickupPlaceholder')}
              </div>

              <div className="autocomplete-wrapper">
                <input
                  className="text-input"
                  type="text"
                  placeholder={t(lang, 'pickupPlaceholder')}
                  value={pickupText}
                  onChange={e => {
                    setComposeMode(true)
                    setPickupText(e.target.value)
                    setPickupLoc(null)
                    setPickupLocked(false)
                    setPickupDirty(true)
                    setPickupSuggestions([])
                    resetFarePanel()
                  }}
                  onBlur={() => setTimeout(() => setPickupSuggestions([]), 200)}
                />
                {pickupSuggestions.length > 0 && (
                  <div className="autocomplete-dropdown">
                    {pickupSuggestions.map((item, idx) => (
                      <button key={idx} type="button" className="autocomplete-item" onMouseDown={e => { e.preventDefault(); setComposeMode(true); handleSelectPickup(item) }}>
                        {item.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {stops.map((stop, index) => (
                <div key={index} style={{ marginTop: 12 }}>
                  <div className="field-label">{t(lang, 'stopLabel')} {index + 1}</div>
                  <div className="autocomplete-wrapper">
                    <input className="text-input" type="text" placeholder={t(lang, 'stopPlaceholder')} value={stop.text} onChange={e => { setComposeMode(true); updateStopText(index, e.target.value) }} onBlur={() => setTimeout(() => { setStops(prev => { const copy = [...prev]; if (!copy[index]) return prev; copy[index] = { ...copy[index], suggestions: [] }; return copy }) }, 200)} />
                    {stop.suggestions && stop.suggestions.length > 0 && (
                      <div className="autocomplete-dropdown">
                        {stop.suggestions.map((item, sIdx) => (
                          <button key={sIdx} type="button" className="autocomplete-item" onMouseDown={e => { e.preventDefault(); setComposeMode(true); handleSelectStopSuggestion(index, item) }}>
                            {item.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <button type="button" className="ghost-btn" style={{ marginTop: 4 }} onClick={() => removeStop(index)}>{t(lang, 'removeStop')}</button>
                </div>
              ))}

              <button type="button" className="ghost-btn" style={{ marginTop: 12 }} onClick={addStop}>{t(lang, 'addStop')}</button>

              <div className="field-label" style={{ marginTop: 16 }}>{t(lang, 'dropoffPlaceholder')}</div>
              <div className="autocomplete-wrapper">
                <input className="text-input" type="text" placeholder={t(lang, 'dropoffPlaceholder')} value={dropoffText} onChange={e => { setComposeMode(true); setDropoffText(e.target.value); setDropoffLoc(null); setDropoffLocked(false); setDropoffDirty(true); setDropoffSuggestions([]); resetFarePanel() }} onBlur={() => setTimeout(() => setDropoffSuggestions([]), 200)} />
                {dropoffSuggestions.length > 0 && (
                  <div className="autocomplete-dropdown">
                    {dropoffSuggestions.map((item, idx) => (
                      <button key={idx} type="button" className="autocomplete-item" onMouseDown={e => { e.preventDefault(); setComposeMode(true); handleSelectDropoff(item) }}>
                        {item.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <button type="button" className="primary-btn" style={{ marginTop: 24, width: '100%' }} onClick={handleCheckPrice} disabled={loading}>{t(lang, 'viewPriceAndCars')}</button>

              {fareError && <div className="error-box" style={{ marginTop: 16 }}>{fareError}</div>}

              {fareOptions && !fareError && (
                <div className="fare-panel" style={{ marginTop: 16 }}>
                  {lastDistanceKm != null && <div className="field-label" style={{ marginBottom: 8 }}>{t(lang, 'estimatedDistancePrefix')} {lastDistanceKm} {t(lang, 'distanceKmUnit')}</div>}
                  <ul className="fare-list">
                    {fareOptions.map(opt => (
                      <li key={opt.type} className="fare-item">
                        <button type="button" className={selectedVehicle === opt.type ? 'fare-item-btn selected' : 'fare-item-btn'} onClick={() => handleChooseFare(opt)} disabled={loading}>
                          <span>{opt.label}</span>
                          <span className="fare-price">≈ ${opt.price.toFixed(2)}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}

          <section className="orders-block" style={{ marginTop: 32 }}>
            <div className="orders-header">
              <h3>{t(lang, 'ordersTitlePassenger')}</h3>
              <button className="ghost-btn" type="button" onClick={refresh} disabled={loading}>{t(lang, 'refresh')}</button>
            </div>

            <OrderList
              lang={lang}
              orders={orders}
              drivers={drivers}
              isDriverView={false}
              selectedOrderId={selectedOrderId}
              completedOrderIds={completedOrderIds}
              onSelectOrder={orderId => {
                userManuallySelectedOrder.current = true
                setSelectedOrderId(orderId)
                setComposeMode(false)
                setComposerLocked(true)
              }}
            />

            {loading && <div className="auth-hint" style={{ marginTop: 8 }}>{t(lang, 'loading')}</div>}
            {error && <div className="error-box">{error}</div>}
          </section>
        </div>
      </aside>
    </section>
  )
}