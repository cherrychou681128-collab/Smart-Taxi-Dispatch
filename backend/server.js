// server.js
import express from "express";
import cors from "cors";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

// ---- fetch polyfill (兼容 Node 16/18/20) ----
const fetchFn =
  globalThis.fetch ||
  (async (...args) => {
    const mod = await import("node-fetch");
    return mod.default(...args);
  });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.set("etag", false); // ✅ 關掉 ETag，避免 304

// ✅ CORS（維持你原本：開放；若之後要鎖 domain 再改）
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ✅ API 全部不快取（雙保險）
app.use("/api", (req, res, next) => {
  res.set("Cache-Control", "no-store");
  next();
});

/* =========================================================
   ✅ JSON 持久化 Store（修正：重啟不再需要重註冊）
   - 預設寫在 ./data/store.json
   - Render：磁碟不是永久的（會重置），但至少本機開發/一般部署會解決
     如果你要 Render 永久保存，必須改用 DB（Postgres/Redis）
   ========================================================= */
const DATA_DIR = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : path.join(__dirname, "data");

const STORE_PATH = process.env.STORE_PATH
  ? path.resolve(process.env.STORE_PATH)
  : path.join(DATA_DIR, "store.json");

function ensureDir(p) {
  try {
    fs.mkdirSync(p, { recursive: true });
  } catch {}
}

function safeReadJson(file, fallback) {
  try {
    if (!fs.existsSync(file)) return fallback;
    const raw = fs.readFileSync(file, "utf-8");
    const obj = JSON.parse(raw);
    return obj ?? fallback;
  } catch {
    return fallback;
  }
}

function safeWriteJson(file, obj) {
  try {
    ensureDir(path.dirname(file));
    fs.writeFileSync(file, JSON.stringify(obj, null, 2), "utf-8");
    return true;
  } catch (e) {
    console.warn("[STORE] write failed:", e?.message || e);
    return false;
  }
}

// ===== 小型「資料庫」：使用者 / 司機 / 訂單 =====
let users = [];
let nextUserId = 1;

let drivers = [];
let nextDriverId = 1;

let orders = [];
let nextOrderId = 1;

// ✅ load store at startup
function loadStore() {
  const init = {
    users: [],
    drivers: [],
    orders: [],
    meta: { nextUserId: 1, nextDriverId: 1, nextOrderId: 1 },
  };

  const store = safeReadJson(STORE_PATH, init);

  users = Array.isArray(store?.users) ? store.users : [];
  drivers = Array.isArray(store?.drivers) ? store.drivers : [];
  orders = Array.isArray(store?.orders) ? store.orders : [];

  const meta = store?.meta && typeof store.meta === "object" ? store.meta : {};
  nextUserId = Number(meta.nextUserId) || 1;
  nextDriverId = Number(meta.nextDriverId) || 1;
  nextOrderId = Number(meta.nextOrderId) || 1;

  // ✅ 保底：避免 meta 壞掉導致 id 重複
  const maxUserId = users.reduce((m, u) => Math.max(m, Number(u?.id) || 0), 0);
  const maxDriverId = drivers.reduce((m, d) => Math.max(m, Number(d?.id) || 0), 0);
  const maxOrderId = orders.reduce((m, o) => Math.max(m, Number(o?.id) || 0), 0);

  if (nextUserId <= maxUserId) nextUserId = maxUserId + 1;
  if (nextDriverId <= maxDriverId) nextDriverId = maxDriverId + 1;
  if (nextOrderId <= maxOrderId) nextOrderId = maxOrderId + 1;

  console.log(
    `[STORE] loaded users=${users.length}, drivers=${drivers.length}, orders=${orders.length} from ${STORE_PATH}`
  );
}

function saveStore() {
  return safeWriteJson(STORE_PATH, {
    users,
    drivers,
    orders,
    meta: { nextUserId, nextDriverId, nextOrderId },
    updatedAt: new Date().toISOString(),
  });
}

// ✅ 啟動載入
loadStore();

/* ========================================================= */

const FALLBACK_GEOCODE_PLACES = [
  { label: "Times Square, Manhattan, New York, NY, USA", lat: 40.758, lng: -73.9855 },
  { label: "Central Park, Manhattan, New York, NY, USA", lat: 40.7829, lng: -73.9654 },
  { label: "Grand Central Terminal, Manhattan, New York, NY, USA", lat: 40.7527, lng: -73.9772 },
  { label: "Wall Street, Manhattan, New York, NY, USA", lat: 40.706, lng: -74.0086 },
  { label: "John F. Kennedy International Airport (JFK), New York, NY, USA", lat: 40.6413, lng: -73.7781 },
  { label: "LaGuardia Airport (LGA), New York, NY, USA", lat: 40.7769, lng: -73.874 },
];

function normalizeType(value) {
  if (typeof value !== "string") return null;
  const s = value.trim().toUpperCase();
  return s ? s : null;
}

function toNum(v) {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function normStatus(s) {
  return String(s || "").trim().toLowerCase();
}

function getAnyDriverId(o) {
  return o?.driverId ?? o?.assignedDriverId ?? o?.driver_id ?? null;
}

/* ===================== SUMO vehicle binding ===================== */
function findSumoTraceFile() {
  const candidates = [
    path.join(__dirname, "public", "sumo_traces", "demo.json"),
    path.join(__dirname, "sumo_traces", "demo.json"),
    path.join(__dirname, "dist", "sumo_traces", "demo.json"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return null;
}

let SUMO_VEHICLE_IDS = [];
let sumoCursor = 0;

function loadSumoVehicleIdsOnce() {
  const file = findSumoTraceFile();
  if (!file) {
    console.warn("[SUMO] demo.json not found (server side). Will skip binding.");
    return;
  }
  try {
    const raw = fs.readFileSync(file, "utf-8");
    const json = JSON.parse(raw);
    const ids = Object.keys(json?.vehicles || {});
    SUMO_VEHICLE_IDS = ids;
    sumoCursor = 0;
    console.log(`[SUMO] Loaded ${ids.length} vehicle ids from ${file}`);
  } catch (e) {
    console.warn("[SUMO] Failed to load demo.json:", e);
  }
}

function pickFreeSumoVehicleId() {
  if (!SUMO_VEHICLE_IDS.length) return null;

  // 盡量避免「尚未 completed 的訂單」共用同一台 SUMO 車
  const used = new Set(
    orders
      .filter((o) => normStatus(o?.status) !== "completed")
      .map((o) => (o?.sumoVehicleId != null ? String(o.sumoVehicleId) : null))
      .filter(Boolean)
  );

  for (let k = 0; k < SUMO_VEHICLE_IDS.length; k++) {
    const idx = (sumoCursor + k) % SUMO_VEHICLE_IDS.length;
    const id = String(SUMO_VEHICLE_IDS[idx]);
    if (!used.has(id)) {
      sumoCursor = (idx + 1) % SUMO_VEHICLE_IDS.length;
      return id;
    }
  }

  const id = String(SUMO_VEHICLE_IDS[sumoCursor % SUMO_VEHICLE_IDS.length]);
  sumoCursor = (sumoCursor + 1) % SUMO_VEHICLE_IDS.length;
  return id;
}

function ensureOrderHasSumoVehicle(order) {
  if (!order) return;
  if (order.sumoVehicleId != null) return;
  const id = pickFreeSumoVehicleId();
  if (id != null) order.sumoVehicleId = id;
}

loadSumoVehicleIdsOnce();
/* =============================================================== */

// =================== Auth API：註冊 / 登入 ===================
app.post("/api/register", (req, res) => {
  const { username, password, role, carType } = req.body;
  if (!username || !password || !role) {
    return res.status(400).json({
      errorCode: "MISSING_FIELDS",
      error: "username / password / role required",
    });
  }

  const existed = users.find((u) => u.username === username);
  if (existed) {
    return res.status(409).json({
      errorCode: "USERNAME_TAKEN",
      error: "Username already taken",
    });
  }

  const carTypeUpper = role === "driver" && carType ? normalizeType(carType) : null;

  const user = {
    id: nextUserId++,
    username,
    password,
    role,
    carType: carTypeUpper,
    createdAt: new Date().toISOString(),
  };

  users.push(user);
  saveStore(); // ✅ 持久化

  const { password: _, ...safeUser } = user;
  return res.json(safeUser);
});

app.post("/api/login", (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({
      errorCode: "MISSING_FIELDS",
      error: "username / password required",
    });
  }

  const user = users.find((u) => u.username === username);
  if (!user) {
    return res.status(400).json({
      errorCode: "NO_SUCH_ACCOUNT",
      error: "Account not found",
    });
  }

  if (user.password !== password) {
    return res.status(400).json({
      errorCode: "WRONG_PASSWORD",
      error: "Wrong password",
    });
  }

  const { password: _, ...safeUser } = user;
  return res.json(safeUser);
});

// =================== Geocode API ===================
function fallbackGeocode(query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return [];

  const tokens = new Set(q.split(/\s+/).filter(Boolean));
  if (q === "nyc") {
    tokens.add("new");
    tokens.add("york");
    tokens.add("manhattan");
    tokens.add("ny");
  }

  const scored = FALLBACK_GEOCODE_PLACES.map((p) => {
    const label = String(p.label || "").toLowerCase();
    let score = 0;
    for (const t of tokens) {
      if (t.length >= 2 && label.includes(t)) score += 1;
    }
    return { p, score };
  });

  scored.sort((a, b) => b.score - a.score);

  const best = scored.filter((x) => x.score > 0).map((x) => x.p);
  return best.length ? best : FALLBACK_GEOCODE_PLACES;
}

app.get("/api/geocode", async (req, res) => {
  const query = String(req.query.q || "").trim();
  if (!query) return res.json([]);

  try {
    const url = `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(
      query
    )}`;

    const response = await fetchFn(url, {
      headers: {
        "User-Agent": "taxi-dispatch-demo/1.0 (contact: vivi39120@gmail.com)",
        Accept: "application/json",
      },
    });

    if (!response.ok) {
      console.warn("Nominatim not ok:", response.status);
      return res.json(fallbackGeocode(query));
    }

    const data = await response.json();
    if (!Array.isArray(data) || data.length === 0) {
      return res.json(fallbackGeocode(query));
    }

    const results = data
      .map((item) => ({
        label: item.display_name,
        lat: Number(item.lat),
        lng: Number(item.lon),
      }))
      .filter((x) => x.label && Number.isFinite(x.lat) && Number.isFinite(x.lng));

    if (!results.length) return res.json(fallbackGeocode(query));
    return res.json(results);
  } catch (err) {
    console.error("Geocode failed, using fallback:", err);
    return res.json(fallbackGeocode(query));
  }
});

// =================== Hotspots proxy API ===================
const HOTSPOT_API_BASE = String(process.env.HOTSPOT_API_BASE || "").replace(/\/+$/, "");

// ✅ 避免 n 太大把 upstream 打爆（不改功能：只是上限防呆）
function clampInt(v, { min = 1, max = 200, fallback = 25 } = {}) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  const nn = Math.floor(n);
  return Math.max(min, Math.min(max, nn));
}

app.get("/api/hotspots", async (req, res) => {
  try {
    if (!HOTSPOT_API_BASE) {
      return res.status(503).json({
        errorCode: "HOTSPOT_PROXY_DISABLED",
        error: "HOTSPOT_API_BASE is not set",
        hint: "Set env HOTSPOT_API_BASE, e.g. https://<your-hotspot-service>",
        rows: [],
      });
    }

    const n = clampInt(req.query.n, { min: 1, max: 200, fallback: 25 });
    const hour = req.query.hour != null ? String(req.query.hour) : "";
    const vehicleType = req.query.vehicleType != null ? String(req.query.vehicleType) : "";
    const lat = req.query.lat != null ? String(req.query.lat) : "";
    const lng = req.query.lng != null ? String(req.query.lng) : "";

    const params = new URLSearchParams();
    params.set("n", String(n));
    if (hour) params.set("hour", hour);
    if (vehicleType) params.set("vehicleType", vehicleType);
    if (lat) params.set("lat", lat);
    if (lng) params.set("lng", lng);

    const url = `${HOTSPOT_API_BASE}/api/hotspots?${params.toString()}`;
    const r = await fetchFn(url, { headers: { Accept: "application/json" } });

    if (!r.ok) {
      const txt = await r.text().catch(() => "");
      return res.status(502).json({
        errorCode: "HOTSPOT_UPSTREAM_ERROR",
        error: `hotspot upstream ${r.status}`,
        detail: txt.slice(0, 300),
        rows: [],
      });
    }

    const data = await r.json().catch(() => null);

    // ✅ 統一回傳格式：前端只要讀 data.rows
    if (Array.isArray(data)) return res.json({ rows: data });
    if (data && typeof data === "object") {
      if (Array.isArray(data.rows)) return res.json(data);
      if (Array.isArray(data.data)) return res.json({ ...data, rows: data.data });
      return res.json({ ...data, rows: [] });
    }

    return res.json({ rows: [] });
  } catch (e) {
    return res.status(500).json({
      errorCode: "HOTSPOT_PROXY_FAILED",
      error: "hotspot proxy failed",
      detail: String(e?.message || e),
      rows: [],
    });
  }
});

// =================== Driver API ===================

// 取得司機列表（前端每 3 秒會打）
app.get("/api/drivers", (req, res) => {
  res.json(drivers);
});

// 司機登入/建立自己的車（前端 login 後會打 /api/driver-login）
// ✅ 每次登入都清空 lat/lng（你要的行為）
app.post("/api/driver-login", (req, res) => {
  const { name, carType } = req.body || {};
  if (!name) return res.status(400).json({ errorCode: "MISSING_FIELDS", error: "name is required" });

  const carTypeUpper = carType ? normalizeType(carType) : null;

  let driver = drivers.find((d) => d.name === name);
  if (!driver) {
    driver = {
      id: nextDriverId++,
      name,
      lat: null,
      lng: null,
      status: "idle",
      carType: carTypeUpper,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    drivers.push(driver);
  } else {
    if (carTypeUpper) driver.carType = carTypeUpper;
    driver.lat = null;
    driver.lng = null;
    driver.status = "idle";
    driver.updatedAt = new Date().toISOString();
  }

  saveStore(); // ✅ 持久化
  return res.json(driver);
});

// 司機更新定位
app.patch("/api/drivers/:id/location", (req, res) => {
  const id = Number(req.params.id);
  const driver = drivers.find((d) => d.id === id);
  if (!driver) return res.status(404).json({ errorCode: "NOT_FOUND", error: "driver not found" });

  const lat = toNum(req.body?.lat);
  const lng = toNum(req.body?.lng);

  if (lat != null) driver.lat = lat;
  if (lng != null) driver.lng = lng;

  if (typeof req.body?.status === "string") driver.status = req.body.status.trim();
  driver.updatedAt = new Date().toISOString();

  saveStore(); // ✅ 持久化
  return res.json(driver);
});

// （可選）方便測試：重置司機定位
app.post("/api/drivers/:id/reset-location", (req, res) => {
  const id = Number(req.params.id);
  const driver = drivers.find((d) => d.id === id);
  if (!driver) return res.status(404).json({ errorCode: "NOT_FOUND", error: "driver not found" });

  driver.lat = null;
  driver.lng = null;
  driver.status = "idle";
  driver.updatedAt = new Date().toISOString();

  saveStore(); // ✅ 持久化
  return res.json(driver);
});

// =================== 訂單 API ===================
app.post("/api/orders", (req, res) => {
  const {
    pickup,
    dropoff,
    customer,
    pickupLat,
    pickupLng,
    dropoffLat,
    dropoffLng,
    distanceKm,
    vehicleType,
    estimatedFare,
    estimatedPrice,
    stops,
  } = req.body;

  if (!pickup || !dropoff) {
    return res.status(400).json({ errorCode: "MISSING_FIELDS", error: "pickup & dropoff are required" });
  }

  const normalizedStops = Array.isArray(stops)
    ? stops.map((s) => ({
        label:
          typeof s?.label === "string"
            ? s.label
            : typeof s?.text === "string"
            ? s.text
            : "",
        lat: toNum(s?.lat ?? s?.loc?.lat),
        lng: toNum(s?.lng ?? s?.loc?.lng),
      }))
    : [];

  const finalPrice =
    typeof estimatedPrice === "number"
      ? estimatedPrice
      : typeof estimatedFare === "number"
      ? estimatedFare
      : toNum(estimatedPrice) ?? toNum(estimatedFare);

  const now = new Date().toISOString();

  const order = {
    id: nextOrderId++,
    pickup,
    dropoff,
    pickupLat: toNum(pickupLat),
    pickupLng: toNum(pickupLng),
    dropoffLat: toNum(dropoffLat),
    dropoffLng: toNum(dropoffLng),
    customer: customer || null,

    status: "pending",

    driverId: null,
    driverName: null,

    assignedDriverId: null,
    assignedDriverName: null,

    sumoVehicleId: null,

    distanceKm: toNum(distanceKm),
    vehicleType: normalizeType(vehicleType),

    estimatedPrice: finalPrice,
    estimatedFare: finalPrice,

    stops: normalizedStops,

    createdAt: now,
    updatedAt: now,
    completedAt: null,
  };

  orders.push(order);
  saveStore(); // ✅ 持久化
  return res.json(order);
});

app.get("/api/orders", (req, res) => {
  res.json(orders);
});

function assignOrderAtomic({ order, driver, driverNameOverride }) {
  if (normStatus(order.status) === "completed") {
    return { ok: false, code: 409, message: "order already completed" };
  }

  if (getAnyDriverId(order) != null) {
    return { ok: false, code: 409, message: "order already taken" };
  }

  if (normStatus(order.status) !== "pending") {
    return { ok: false, code: 409, message: `order not pending (status=${order.status})` };
  }

  const now = new Date().toISOString();

  order.status = "assigned";
  order.driverId = driver.id;
  order.driverName = driverNameOverride || driver.name;

  order.assignedDriverId = driver.id;
  order.assignedDriverName = driverNameOverride || driver.name;

  ensureOrderHasSumoVehicle(order);

  order.updatedAt = now;
  driver.status = "busy";
  driver.updatedAt = now;

  return { ok: true, order };
}

app.post("/api/orders/:id/assign", (req, res) => {
  const id = Number(req.params.id);
  const { driverId, driverName } = req.body;

  const order = orders.find((o) => o.id === id);
  if (!order) return res.status(404).json({ errorCode: "NOT_FOUND", error: "order not found" });

  const driver = drivers.find((d) => d.id === Number(driverId));
  if (!driver) return res.status(404).json({ errorCode: "NOT_FOUND", error: "driver not found" });

  const r = assignOrderAtomic({ order, driver, driverNameOverride: driverName });
  if (!r.ok) return res.status(r.code).json({ errorCode: "ASSIGN_FAILED", error: r.message });

  saveStore(); // ✅ 持久化
  return res.json(r.order);
});

app.post("/api/orders/:id/accept", (req, res) => {
  const id = Number(req.params.id);
  const { driverId, driverName } = req.body;

  const order = orders.find((o) => o.id === id);
  if (!order) return res.status(404).json({ errorCode: "NOT_FOUND", error: "order not found" });

  const driver = drivers.find((d) => d.id === Number(driverId));
  if (!driver) return res.status(404).json({ errorCode: "NOT_FOUND", error: "driver not found" });

  const r = assignOrderAtomic({ order, driver, driverNameOverride: driverName });
  if (!r.ok) return res.status(r.code).json({ errorCode: "ACCEPT_FAILED", error: r.message });

  saveStore(); // ✅ 持久化
  return res.json(r.order);
});

app.patch("/api/orders/:id/status", (req, res) => {
  const id = Number(req.params.id);
  const { status } = req.body;

  const order = orders.find((o) => o.id === id);
  if (!order) return res.status(404).json({ errorCode: "NOT_FOUND", error: "order not found" });

  if (typeof status !== "string" || !status.trim()) {
    return res.status(400).json({ errorCode: "MISSING_FIELDS", error: "status is required" });
  }

  order.status = status.trim();
  order.updatedAt = new Date().toISOString();

  if (normStatus(order.status) === "completed") {
    order.completedAt = new Date().toISOString();

    const did = getAnyDriverId(order);
    if (did != null) {
      const driver = drivers.find((d) => d.id === Number(did));
      if (driver) {
        driver.status = "idle";
        driver.updatedAt = new Date().toISOString();
      }
    }
  }

  saveStore(); // ✅ 持久化
  return res.json(order);
});

app.post("/api/orders/:id/complete", (req, res) => {
  const id = Number(req.params.id);

  const order = orders.find((o) => o.id === id);
  if (!order) return res.status(404).json({ errorCode: "NOT_FOUND", error: "order not found" });

  order.status = "completed";
  order.updatedAt = new Date().toISOString();
  order.completedAt = new Date().toISOString();

  const did = getAnyDriverId(order);
  if (did != null) {
    const driver = drivers.find((d) => d.id === Number(did));
    if (driver) {
      driver.status = "idle";
      driver.updatedAt = new Date().toISOString();
    }
  }

  saveStore(); // ✅ 持久化
  return res.json(order);
});

// =================== Production：前端靜態檔案（dist 存在才掛） ===================
const distPath = path.join(__dirname, "dist");
const publicPath = path.join(__dirname, "public");

if (fs.existsSync(publicPath)) {
  app.use(express.static(publicPath));
}

if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get(/^(?!\/api).*/, (req, res) => {
    res.sendFile(path.join(distPath, "index.html"));
  });
}

const PORT = process.env.PORT || 8000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
  console.log(`[STORE] path = ${STORE_PATH}`);
  if (HOTSPOT_API_BASE) console.log(`[HOTSPOT] proxy enabled -> ${HOTSPOT_API_BASE}`);
  else console.log("[HOTSPOT] proxy disabled (set HOTSPOT_API_BASE to enable)");
});
