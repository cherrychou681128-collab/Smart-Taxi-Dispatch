// src/components/LocationInput.jsx
import { useState } from 'react'

export default function LocationInput({
  label,
  placeholder,
  value,
  onTextChange,      // (text: string) => void
  onLocationSelect,  // (loc: { label, lat, lng }) => void
}) {
  const [query, setQuery] = useState(value || '')
  const [options, setOptions] = useState([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)

  const handleChange = async e => {
    const q = e.target.value
    setQuery(q)
    onTextChange?.(q)

    // 太短就不要打 API，順便把候選清掉
    if (!q.trim() || q.trim().length < 2) {
      setOptions([])
      setOpen(false)
      return
    }

    try {
      setLoading(true)
      const res = await fetch(
        `/api/geocode?q=${encodeURIComponent(q.trim())}`
      )
      if (!res.ok) throw new Error('geocode error')
      const data = await res.json()
      setOptions(Array.isArray(data) ? data : [])
      setOpen(true)
    } catch (err) {
      console.error('[LocationInput] geocode failed', err)
      setOptions([])
      setOpen(false)
    } finally {
      setLoading(false)
    }
  }

  const handleSelect = opt => {
    setQuery(opt.label)
    onTextChange?.(opt.label)
    onLocationSelect?.(opt)
    setOptions([])
    setOpen(false)
  }

  return (
    <div className="location-input-wrapper">
      {label && <div className="field-label">{label}</div>}
      <input
        className="text-input"
        type="text"
        placeholder={placeholder}
        value={query}
        onChange={handleChange}
        onFocus={() => {
          if (options.length > 0) setOpen(true)
        }}
        onBlur={() => {
          // 簡單處理：稍微 delay 一下，讓點擊選項有機會觸發
          setTimeout(() => setOpen(false), 150)
        }}
      />
      {loading && (
        <div className="location-input-hint">查詢中…</div>
      )}

      {open && options.length > 0 && (
        <ul className="location-suggestion-list">
          {options.map((opt, idx) => (
            <li
              key={`${opt.label}-${idx}`}
              className="location-suggestion-item"
              onMouseDown={e => {
                e.preventDefault()
                handleSelect(opt)
              }}
            >
              <div className="location-suggestion-label">
                {opt.label}
              </div>
              <div className="location-suggestion-coord">
                {opt.lat.toFixed(5)}, {opt.lng.toFixed(5)}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
