const API_ORIGIN = import.meta.env.VITE_API_ORIGIN || 'https://taxi-hotspot-backend.onrender.com'

export function apiUrl(path = '') {
  if (!path) return API_ORIGIN

  if (/^https?:\/\//i.test(path)) return path

  const p = path.startsWith('/') ? path : `/${path}`
  return `${API_ORIGIN.replace(/\/$/, '')}${p}`
}
