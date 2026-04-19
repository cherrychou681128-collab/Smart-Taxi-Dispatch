// src/mapIcons.js
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// 司機車子圖示
export const taxiIcon = L.divIcon({
  className: 'taxi-marker',
  html: '🚕',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
})

// 乘客起點（人像）
export const passengerIcon = L.divIcon({
  className: 'passenger-marker',
  html: '🧍',
  iconSize: [28, 28],
  iconAnchor: [14, 28],
})

// 終點（終點旗）
export const dropoffIcon = L.divIcon({
  className: 'dropoff-marker',
  html: '🏁',
  iconSize: [28, 28],
  iconAnchor: [14, 28],
})

// 依照停靠點順序產生 1 / 2 / 3 這種圓形標記
export function createStopIcon(order) {
  const text = String(order)

  return L.divIcon({
    className: 'stop-marker',
    html: `<div class="stop-marker-inner">${text}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 26],
  })
}
