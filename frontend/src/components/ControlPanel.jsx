// src/components/ControlPanel.jsx
import { useState } from 'react'
import OrderList from './OrderList.jsx'

export default function ControlPanel({
  lang,
  orders = [],
  drivers = [],
  loading,
  error,
  createOrder,
  refresh,
  currentUser,
}) {
  const [pickup, setPickup] = useState('')
  const [dropoff, setDropoff] = useState('')

  const handleSubmit = e => {
    e.preventDefault()
    createOrder(pickup, dropoff)
    setPickup('')
    setDropoff('')
  }

  return (
    <div className="panel-inner">
      <h1 className="panel-title">乘客端</h1>

      <div className="field-label">目前乘客：</div>
      <div className="current-driver-box">
        {currentUser?.username || '尚未登入'}
      </div>

      <form className="order-form" onSubmit={handleSubmit}>
        <div className="field-label">上車地點（例如：Times Square）</div>
        <input
          className="text-input"
          type="text"
          placeholder="上車地點"
          value={pickup}
          onChange={e => setPickup(e.target.value)}
        />

        <div className="field-label">目的地（例如：Central Park）</div>
        <input
          className="text-input"
          type="text"
          placeholder="目的地"
          value={dropoff}
          onChange={e => setDropoff(e.target.value)}
        />

        <button className="primary-btn" type="submit" disabled={loading}>
          叫車
        </button>
      </form>

      <section className="orders-block">
        <div className="orders-header">
          <h3>我的訂單</h3>
          <button
            className="ghost-btn"
            type="button"
            onClick={refresh}
            disabled={loading}
          >
            重新整理
          </button>
        </div>

        <OrderList orders={orders} drivers={drivers} isDriverView={false} />

        {loading && (
          <div className="auth-hint" style={{ marginTop: 8 }}>
            更新中…
          </div>
        )}
        {error && <div className="error-box">{error}</div>}
      </section>
    </div>
  )
}
