// src/components/OrderList.jsx
import React, { useMemo } from 'react'
import { t } from '../i18n'

function getStr(x) {
  return typeof x === 'string' ? x : x == null ? '' : String(x)
}

function normalizeStatus(s) {
  const k = getStr(s).toLowerCase().trim()
  if (!k) return 'pending'
  if (k === 'enroute') return 'en_route'
  if (k === 'on_trip') return 'in_progress'
  return k
}

function statusLabel(lang, statusKey) {
  const zh = {
    pending: '等待派單',
    assigned: '已指派',
    accepted: '司機已接單',
    en_route: '前往上車點',
    picked_up: '行程中',
    in_progress: '行程中',
    ongoing: '行程中',
    completed: '已完成',
    cancelled: '已取消',
  }
  const dict = lang === 'zh' ? zh : zh
  return dict[statusKey] || (lang === 'zh' ? '進行中' : 'In progress')
}

function statusTone(statusKey) {
  if (statusKey === 'completed') return 'success'
  if (statusKey === 'cancelled') return 'danger'
  if (statusKey === 'en_route') return 'warn'
  if (statusKey === 'accepted' || statusKey === 'assigned') return 'info'
  if (statusKey === 'pending') return 'muted'
  return 'muted'
}

function extractStops(o) {
  const stopsArr = Array.isArray(o?.stops)
    ? o.stops
    : Array.isArray(o?.resolvedStops)
    ? o.resolvedStops
    : Array.isArray(o?.stopovers)
    ? o.stopovers
    : []

  const stopsText = stopsArr
    .map(s => s?.label || s?.text || s?.name || '')
    .map(x => String(x).trim())
    .filter(Boolean)

  const visibleStops = stopsText.slice(0, 2)
  const moreCount = Math.max(0, stopsText.length - visibleStops.length)

  return { visibleStops, moreCount, total: stopsText.length }
}

export default function OrderList({
  lang,
  orders,
  drivers,
  isDriverView,
  currentDriverId, // 司機端會傳
  onAcceptOrder, // 司機端會傳
  selectedOrderId,
  completedOrderIds,
  onSelectOrder,
}) {
  const driverMap = useMemo(() => {
    const m = new Map()
    ;(Array.isArray(drivers) ? drivers : []).forEach(d => {
      if (d?.id != null) m.set(d.id, d)
    })
    return m
  }, [drivers])

  const list = Array.isArray(orders) ? orders : []

  return (
    <div className="ub-orders">
      <div className="ub-orders__header">
        <div className="ub-orders__title">
          {t(lang, isDriverView ? 'ordersTitleDriver' : 'ordersTitlePassenger')}
        </div>
        <div className="ub-orders__count">{list.length} 筆</div>
      </div>

      <div className="ub-orders__grid">
        {list.map(o => {
          const id = o?.id
          const isSelected = id != null && selectedOrderId === id

          // ✅ 只信 completedOrderIds：避免後端狀態亂寫
          const done = Boolean(completedOrderIds?.has?.(id))

          const rawStatus = normalizeStatus(o?.status)
          const statusKey = done ? 'completed' : rawStatus

          const pickup = o?.pickup ?? o?.pickupText ?? o?.pickup_label ?? '-'
          const dropoff = o?.dropoff ?? o?.dropoffText ?? o?.dropoff_label ?? '-'

          const driverId = o?.driverId ?? o?.assignedDriverId ?? o?.driver_id
          const driverName =
            driverId != null
              ? (driverMap.get(driverId)?.name || driverMap.get(driverId)?.username || '')
              : ''

          // ✅ 司機端：pending + 未指派司機 + 有接單 callback => 顯示「接單」
          const canAccept =
            Boolean(isDriverView) &&
            !done &&
            statusKey === 'pending' &&
            driverId == null &&
            typeof onAcceptOrder === 'function'

          const acceptDisabled = !currentDriverId || id == null

          const onAccept = e => {
            e.preventDefault()
            e.stopPropagation()
            if (acceptDisabled) return
            onAcceptOrder(id, currentDriverId)
          }

          const tone = statusTone(statusKey)
          const { visibleStops, moreCount } = extractStops(o)

          return (
            <button
              key={id ?? Math.random()}
              type="button"
              className={['ub-orderCard', isSelected ? 'is-selected' : '', done ? 'is-done' : '']
                .filter(Boolean)
                .join(' ')}
              onClick={() => id != null && onSelectOrder?.(id)}
            >
              <div className="ub-orderCard__top">
                <div className="ub-orderCard__chips">
                  {canAccept ? (
                    <span
                      className={`ub-chip ub-chip--action ${acceptDisabled ? 'is-disabled' : ''}`}
                      role="button"
                      tabIndex={0}
                      aria-disabled={acceptDisabled}
                      onClick={onAccept}
                      onKeyDown={e => {
                        if (e.key === 'Enter' || e.key === ' ') onAccept(e)
                      }}
                      title={acceptDisabled ? '請先登入/選擇司機' : '接單'}
                    >
                      接單
                    </span>
                  ) : (
                    <span className={`ub-chip ub-chip--${tone}`}>{statusLabel(lang, statusKey)}</span>
                  )}

                  {o?.vehicleType ? <span className="ub-chip ub-chip--ghost">{o.vehicleType}</span> : null}

                  {/* 額外資訊可選：顯示司機名（乘客端也可看得到） */}
                  {driverName ? <span className="ub-chip ub-chip--ghost">{driverName}</span> : null}
                </div>

                <div className="ub-orderCard__id">#{id ?? '-'}</div>
              </div>

              <div className="ub-route">
                <div className="ub-route__row">
                  <span className="ub-route__dot" aria-hidden="true" />
                  <div className="ub-route__content">
                    <div className="ub-route__label">上車</div>
                    <div className="ub-route__text">{pickup}</div>
                  </div>
                </div>

                {/* ✅ 停靠點（新增） */}
                {visibleStops.length > 0 && (
                  <div className="ub-route__stops">
                    <div className="ub-route__stopsLabel">停靠點</div>
                    <div className="ub-route__stopsText">
                      {visibleStops.join(' · ')}
                      {moreCount > 0 ? ` · +${moreCount}` : ''}
                    </div>
                  </div>
                )}

                <div className="ub-route__row" style={{ marginTop: 10 }}>
                  <span className="ub-route__dot is-hollow" aria-hidden="true" />
                  <div className="ub-route__content">
                    <div className="ub-route__label">下車</div>
                    <div className="ub-route__text">{dropoff}</div>
                  </div>
                </div>
              </div>

              <div className="ub-meta">
                <div className="ub-meta__item">
                  <span className="ub-meta__k">司機</span>
                  <span className="ub-meta__v">{driverName || driverId || '—'}</span>
                </div>

                <div className="ub-meta__item">
                  <span className="ub-meta__k">距離</span>
                  <span className="ub-meta__v">
                    {typeof o?.distanceKm === 'number' ? `${o.distanceKm.toFixed(1)} km` : '—'}
                  </span>
                </div>

                <div className="ub-meta__item ub-meta__item--full">
                  <span className="ub-meta__k">時間</span>
                  <span className="ub-meta__v">{o?.updatedAt || o?.createdAt || '—'}</span>
                </div>
              </div>
            </button>
          )
        })}

        {!list.length && <div className="ub-empty">目前沒有訂單</div>}
      </div>
    </div>
  )
}
