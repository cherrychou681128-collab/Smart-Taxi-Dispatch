import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

export const taxiIcon = L.divIcon({
  className: 'taxi-marker',
  html: '🚕',
  iconSize: [32, 32],
  iconAnchor: [16, 32],
})

export const passengerIcon = L.divIcon({
  className: 'passenger-marker',
  html: '🧍',
  iconSize: [28, 28],
  iconAnchor: [14, 28],
})

export const dropoffIcon = L.divIcon({
  className: 'dropoff-marker',
  html: '🏁',
  iconSize: [28, 28],
  iconAnchor: [14, 28],
})

export function createStopIcon(order) {
  const text = String(order)

  return L.divIcon({
    className: 'stop-marker',
    html: `<div class="stop-marker-inner">${text}</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 26],
  })
}
