// 根據文字關鍵字，回傳紐約的經緯度
// 之後你要再加地點，只要在 places 裡加一個物件就好
const places = [
  {
    name: 'Times Square',
    lat: 40.758,
    lng: -73.9855,
    keywords: ['times square', '時代廣場', 'ts']
  },
  {
    name: 'Central Park',
    lat: 40.7829,
    lng: -73.9654,
    keywords: ['central park', '中央公園']
  },
  {
    name: 'Grand Central Terminal',
    lat: 40.7527,
    lng: -73.9772,
    keywords: ['grand central', '中央車站', 'grand station']
  },
  {
    name: 'JFK Airport',
    lat: 40.6413,
    lng: -73.7781,
    keywords: ['jfk', '肯尼迪', 'kennedy', '機場']
  },
  {
    name: 'Brooklyn Bridge',
    lat: 40.7061,
    lng: -73.9969,
    keywords: ['brooklyn bridge', '布魯克林大橋']
  },
  {
    name: 'Wall Street',
    lat: 40.7065,
    lng: -74.009,
    keywords: ['wall street', '華爾街', '金融區']
  }
]

export function resolveLocation(text) {
  if (!text) return null
  const lower = text.toLowerCase()

  for (const place of places) {
    if (place.keywords.some(k => lower.includes(k.toLowerCase()))) {
      return { lat: place.lat, lng: place.lng, name: place.name }
    }
  }

  // 找不到關鍵字：預設丟在 Times Square 附近
  return { lat: 40.758, lng: -73.9855, name: text }
}
