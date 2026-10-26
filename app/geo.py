import os, json, math, time, logging, threading

# Attempt to import the Earth Engine Python API. If unavailable, set a placeholder so that tests can patch it.
try:
    import ee  # type: ignore
except Exception:  # pragma: no cover
    ee = None  # placeholder for mocking in tests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.spatial import cKDTree

log = logging.getLogger("agrinet.geo")
ROOT = Path(__file__).resolve().parent.parent
LABELS = json.loads((Path(__file__).parent / "label_map.json").read_text())
RADIUS_M = float(os.getenv("PLANET_RADIUS_M", 300))
B = [f"B{i}" for i in range(1, 9)]
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash(lat, lon, precision=7):
    lat_r, lon_r, bits, ch, even, out = [-90.0, 90.0], [-180.0, 180.0], 0, 0, True, ""
    while len(out) < precision:
        rng, v = (lon_r, lon) if even else (lat_r, lat)
        mid = (rng[0] + rng[1]) / 2
        ch <<= 1
        if v >= mid: ch |= 1; rng[0] = mid
        else: rng[1] = mid
        even = not even; bits += 1
        if bits == 5: out += _B32[ch]; bits = ch = 0
    return out

# ---- Planet (local, high-res) path ----
_lookup = _tree = _clf = _enc = _scaler = None
def _load():
    global _lookup, _tree, _clf, _enc, _scaler
    if _lookup is None:
        p = ROOT / "data" / "lookup.parquet"
        if p.exists():
            _lookup = pd.read_parquet(p)
            _tree = cKDTree(np.c_[np.radians(_lookup.Lat), np.radians(_lookup.Lon) * math.cos(math.radians(22))])
        c = ROOT / "data" / "classifier_emb.joblib"
        if c.exists() and (ROOT / "models" / "planet_mae_ckpt_32.pt").exists():
            from .spear_encoder import PlanetEncoder
            _clf = joblib.load(c); _scaler = joblib.load(ROOT / "models" / "planet_scaler_32.joblib")
            _enc = PlanetEncoder(str(ROOT / "models" / "planet_mae_ckpt_32.pt"))
    return _lookup is not None

def spear_predict(bands):
    """SPEAR MAE embedding + bands -> land-cover head. Returns (id, prob, top3) or None if bands incomplete."""
    if _clf is None or np.isnan(bands).any(): return None
    xs = _scaler.transform(bands.reshape(1, -1).astype(np.float32)).astype(np.float32)
    pr = _clf.predict_proba(np.c_[_enc.embed(xs), xs])[0]; k = int(np.argmax(pr))
    top = sorted(zip(_clf.classes_, pr), key=lambda t: -t[1])[:3]
    return int(_clf.classes_[k]), float(pr[k]), [{"class": LABELS["planet_classes"][str(int(c))], "p": round(float(p), 2)} for c, p in top]

def planet_lookup(lat, lon):
    """Return Planet-derived result if a stored pixel lies within RADIUS_M, else None.
    land_cover = stored Dynamic World label for that cell (reference); model_prediction = SPEAR head (second opinion)."""
    if not _load(): return None
    d, i = _tree.query([math.radians(lat), math.radians(lon) * math.cos(math.radians(22))])
    dist_m = float(d) * 6371000
    if dist_m > RADIUS_M: return None
    r = _lookup.iloc[int(i)]
    bands = r[B].astype(float).values
    ndvi = float((bands[7] - bands[5]) / (bands[7] + bands[5] + 1e-6))
    # NDWI = (Green - NIR) / (Green + NIR); PlanetScope SuperDove B4=Green(565nm), B8=NIR(865nm) → index 3,7 (0-based)
    ndwi_denom = bands[3] + bands[7]
    ndwi = float((bands[3] - bands[7]) / (ndwi_denom + 1e-6)) if ndwi_denom != 0 else None
    ref = int(r.label); out = {"source": "planet_highres", "resolution": "~3 m PlanetScope", "observed": str(r.date),
        "distance_m": round(dist_m), "land_cover_id": ref, "land_cover": LABELS["planet_classes"][str(ref)],
        "label_origin": "Dynamic World label stored with this Planet cell", "is_cropland": ref in LABELS["cropland_ids"],
        "ndvi_planet": None if math.isnan(ndvi) else round(ndvi, 3),
        "ndwi_planet": None if (ndwi is None or math.isnan(ndwi)) else round(ndwi, 3)}
    sp = spear_predict(np.array(bands))
    if sp:
        out["model_prediction"] = {"land_cover": LABELS["planet_classes"][str(sp[0])], "confidence": round(sp[1], 2),
                                   "top3": sp[2], "agrees_with_label": sp[0] == ref, "model": "SPEAR PlanetScope MAE (32-d) + head"}
        out["confidence"] = round(sp[1], 2)
    return out

# ---- Earth Engine (live, everywhere) path ----
_ee_ok = None
_ee_failed_at = 0.0
_EE_RETRY_S = 300
_ee_lock = threading.Lock()
def _ee():
    """Initialise Earth Engine once; a failed init is retried after a cool-down instead of disabling EE for good.
    Supports service‑account authentication via EE_SERVICE_ACCOUNT_JSON and EE_PROJECT_ID.
    Falls back to project‑only init for local development when no service account is provided.
    """
    global _ee_ok, _ee_failed_at
    with _ee_lock:
        # Reset after cool‑down if previous init failed
        if _ee_ok is False and time.monotonic() - _ee_failed_at > _EE_RETRY_S:
            _ee_ok = None
        if _ee_ok is None:
            try:
                # Use the module-level ee (which may be a mock in tests)
                if ee is None:
                    raise RuntimeError("earthengine-api library is not available")
                # Prefer explicit Render/production variables
                project_id = os.getenv("EE_PROJECT_ID") or os.getenv("EE_PROJECT")
                sa_json_str = os.getenv("EE_SERVICE_ACCOUNT_JSON")
                if sa_json_str:
                    # Parse and minimally validate the service‑account payload
                    try:
                        sa_json = json.loads(sa_json_str)
                        required_keys = {"type", "project_id", "private_key", "client_email"}
                        if not required_keys.issubset(sa_json):
                            raise ValueError("Service account JSON missing required fields")
                    except Exception as exc:
                        raise RuntimeError("EE_SERVICE_ACCOUNT_JSON is malformed or invalid") from exc
                    credentials = ee.ServiceAccountCredentials(email=None, key_data=json.dumps(sa_json))
                    ee.Initialize(credentials, project=project_id)
                else:
                    if not project_id:
                        raise RuntimeError("EE_PROJECT_ID environment variable is required for Earth Engine initialization")
                    ee.Initialize(project=project_id)
                _ee_ok = True
            except Exception as e:
                # Log a generic warning; avoid leaking credential details.
                log.warning("Earth Engine unavailable")
                _ee_ok = False
                _ee_failed_at = time.monotonic()
                # Re‑raise only for missing configuration (project env) – all other RuntimeErrors are treated as init failures.
                if isinstance(e, RuntimeError):
                    msg = str(e).lower()
                    if "environment variable" in msg or "required" in msg:
                        raise


        return _ee_ok

def warm_up():
    """Initialise Earth Engine and the Planet lookup ahead of the first request (called from app startup)."""
    try:
        _ee()
    except RuntimeError:
        pass  # Missing config is reported in logs; do not crash the startup thread.
    _load()


def ee_config_status() -> dict:
    """Return a read-only snapshot of the Earth Engine configuration state.

    Intentionally lightweight: no remote EE call is made, no credentials are read,
    and no exception text is returned to the caller.

    Fields:
      - ``sdk_initialized`` (bool): True when ``ee.Initialize()`` completed without error.
      - ``sdk_available``   (bool): True when the ``earthengine-api`` package is importable.
      - ``configuration_present`` (bool): True when at least ``EE_PROJECT_ID`` or ``EE_PROJECT``
          is set in the environment (value is not exposed).
      - ``note`` (str): Reminder that SDK init alone does not prove remote EE reachability.
    """
    sdk_available = ee is not None
    project_set = bool(os.getenv("EE_PROJECT_ID") or os.getenv("EE_PROJECT"))
    return {
        "sdk_available": sdk_available,
        "sdk_initialized": _ee_ok is True,
        "configuration_present": project_set,
        "note": (
            "sdk_initialized=true means ee.Initialize() succeeded locally. "
            "It does not guarantee the service account can reach the Earth Engine API."
        ),
    }
EE_CONNECTIVITY_CACHE_TTL_S = 5 * 60  # cache result for 5 minutes
_ee_connectivity_cache: dict = {}
_ee_connectivity_lock = threading.Lock()
# Hold the in‑flight future so concurrent requests share the same probe.
_ee_connectivity_future: object = None

def _run_connectivity_check() -> dict:
    """Execute a minimal EE request to verify remote API availability.

    Returns a dict with ``reachable`` (bool) and an ``error`` code when
    unavailable. The function extracts HTTP status codes when present, but
    never returns raw exception messages or credential data.
    """
    try:
        img = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').limit(1).first()
        img.getInfo()
        return {"reachable": True}
    except Exception as exc:
        # Attempt to extract an HTTP status code from known exception shapes.
        status = None
        # google.api_core.exceptions.HttpError-like
        if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
            status = exc.response.status_code
        # ee exception may expose a 'code' attribute.
        elif hasattr(exc, "code"):
            status = getattr(exc, "code")
        # Fallback: inspect args for a dict with 'status_code'.
        elif isinstance(exc.args, tuple) and len(exc.args) > 0 and isinstance(exc.args[0], dict):
            status = exc.args[0].get("status_code")
        # Map status to a sanitized error identifier.
        if status == 403:
            err = "forbidden"
        elif status == 401:
            err = "unauthorized"
        elif status is not None:
            err = f"http_{status}"
        else:
            err = "request_failed"
        return {"reachable": False, "error": err}


def ee_live_status(timeout: float = None) -> dict:
    global _ee_connectivity_future
    """Return cached remote EE connectivity status.

    Implements a single‑flight probe: concurrent calls will share the same in‑flight
    future rather than spawning multiple network requests. The result is cached for
    ``EE_CONNECTIVITY_CACHE_TTL_S`` seconds.
    """
    now = time.monotonic()
    # Fast path: fresh cache.
    with _ee_connectivity_lock:
        entry = _ee_connectivity_cache.get("status")
        if entry and now - entry[0] < EE_CONNECTIVITY_CACHE_TTL_S:
            return entry[1]
        # Re‑use existing in‑flight future if present.
        future = _ee_connectivity_future
        if future is None:
            if timeout is None:
                timeout = EE_TIMEOUT_S
            if not _ee():
                result = {"reachable": False, "error": "init_failed"}
                _ee_connectivity_cache["status"] = (now, result)
                return result
            # Start a new probe.
            future = _ee_pool.submit(_run_connectivity_check)
            _ee_connectivity_future = future
    try:
        # Wait for the probe respecting the timeout.
        if timeout is None:
            timeout = EE_TIMEOUT_S
        result = future.result(timeout=timeout)
    except FutureTimeout:
        result = {"reachable": False, "error": "timeout"}
        # Cancel the underlying request if possible to avoid orphaned work.
        try:
            future.cancel()
        except Exception:
            pass
    # Store result and clear the in‑flight marker.
    with _ee_connectivity_lock:
        _ee_connectivity_cache["status"] = (now, result)
        _ee_connectivity_future = None
    return result

def ee_health() -> dict:
    """Combined EE health information for the ``/health/earth-engine`` endpoint.

    Returns the static configuration snapshot (via ``ee_config_status``) plus the
    result of ``ee_live_status`` under the ``live`` key. Errors are sanitized.
    """
    status = ee_config_status()
    status["live"] = ee_live_status()
    return status

def _s2_clear(img):
    """Mask clouds/shadows: keep vegetation, bare soil, water and unclassified pixels from the S2 scene classification."""
    scl = img.select("SCL")
    return img.updateMask(scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)))

def ee_analysis(lat, lon, days=180):
    """Dynamic World land cover (mode, last 90d) + Sentinel-2 NDVI/NDWI series + MODIS land/water mask at the point.
    Everything is evaluated in a single getInfo() round trip to keep latency low."""
    if not _ee(): return None
    import ee, datetime as dt
    pt = ee.Geometry.Point([lon, lat]); end = dt.date.today(); start = end - dt.timedelta(days=days)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(pt)
          .filterDate(str(end - dt.timedelta(days=90)), str(end)))
    probability_bands = ["water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"]
    # -1 marks "no value": the whole query is one round trip, so a missing Dynamic World scene (e.g. open
    # sea) must not raise and take the NDVI series and water mask down with it. ee.Algorithms.If is lazy.
    has_dw = dw.size().gt(0)
    label_value = ee.Algorithms.If(has_dw, dw.select("label").mode()
                                   .reduceRegion(ee.Reducer.first(), pt, 10).get("label", -1), -1)
    confidence_value = ee.Algorithms.If(has_dw, dw.select(probability_bands).mean().reduce(ee.Reducer.max())
                                        .reduceRegion(ee.Reducer.first(), pt, 10).get("max", -1), -1)
    # MODIS land/water mask also covers open sea and coasts, where Dynamic World has no scenes.
    water_mask = (ee.ImageCollection("MODIS/061/MOD44W").select("water_mask").sort("system:time_start", False).first()
                  .reduceRegion(ee.Reducer.first(), pt, 250).get("water_mask", -1))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(str(start), str(end))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
          .sort("system:time_start", False).limit(20).map(_s2_clear))
    def f(img):
        ndvi_v = img.normalizedDifference(["B8", "B4"]).reduceRegion(ee.Reducer.mean(), pt.buffer(30), 10).get("nd")
        # NDWI = (Green - NIR) / (Green + NIR) using Sentinel-2 B3 (Green) and B8 (NIR)
        ndwi_v = img.normalizedDifference(["B3", "B8"]).reduceRegion(ee.Reducer.mean(), pt.buffer(30), 10).get("nd")
        return ee.Feature(None, {"d": img.date().format("YYYY-MM-dd"), "ndvi": ndvi_v, "ndwi": ndwi_v})
    fc = s2.map(f)
    ndvi_fc = fc.filter(ee.Filter.notNull(["ndvi"]))
    ndwi_fc = fc.filter(ee.Filter.notNull(["ndwi"]))
    v = ee.Dictionary({
        "label": label_value, "confidence": confidence_value, "water_mask": water_mask,
        "ndvi_d": ndvi_fc.aggregate_array("d"), "ndvi": ndvi_fc.aggregate_array("ndvi"),
        "ndwi_d": ndwi_fc.aggregate_array("d"), "ndwi": ndwi_fc.aggregate_array("ndwi"),
    }).getInfo()
    ndvi_series = sorted(({"date": d, "ndvi": round(x, 3)} for d, x in zip(v.get("ndvi_d") or [], v.get("ndvi") or [])),
                         key=lambda r: r["date"])
    ndwi_series = sorted(({"date": d, "ndwi": round(x, 3)} for d, x in zip(v.get("ndwi_d") or [], v.get("ndwi") or [])),
                         key=lambda r: r["date"])
    lab, confidence = (None if v.get(k) in (None, -1) else v[k] for k in ("label", "confidence"))
    base = {"series": ndvi_series, "ndwi_series": ndwi_series}
    if lab is None:
        if v.get("water_mask") == 1:
            return base | {"source": "earth_engine_live", "resolution": "250 m MODIS land/water mask", "observed": str(end),
                           "land_cover_id": 0, "land_cover": LABELS["dynamic_world"]["0"], "confidence": None,
                           "is_cropland": False, "label_origin": "MODIS MOD44W water mask (no Dynamic World scene)"}
        return base
    return base | {"source": "earth_engine_live", "resolution": "10 m Dynamic World / Sentinel-2",
            "observed": str(end), "land_cover_id": int(lab), "land_cover": LABELS["dynamic_world"][str(int(lab))],
            "confidence": round(float(confidence), 3) if confidence is not None else None,
            "is_cropland": int(lab) in LABELS["dynamic_world_cropland_ids"]}

# EE results are cached per ~150 m geohash cell. A query that overruns the deadline keeps running
# and fills the cache, so retrying the same place is instant instead of timing out again.
EE_TIMEOUT_S = 18.0
_EE_CACHE_TTL_S = 6 * 3600
_ee_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ee")
_ee_cache: dict = {}
_ee_inflight: dict = {}
_ee_cache_lock = threading.Lock()

def _ee_job(key, lat, lon):
    try:
        res = ee_analysis(lat, lon)
    except Exception as e:
        log.warning("EE query failed: %s", e); res = None
    with _ee_cache_lock:
        _ee_inflight.pop(key, None)
        if res is not None:
            _ee_cache[key] = (time.monotonic(), res)
            if len(_ee_cache) > 512: _ee_cache.pop(next(iter(_ee_cache)))
    return res

def ee_analysis_cached(lat, lon, timeout=EE_TIMEOUT_S):
    """Cached, deadline-bounded ee_analysis(). Returns None when EE is unavailable, failed or is still running."""
    key = geohash(lat, lon, 7)
    with _ee_cache_lock:
        hit = _ee_cache.get(key)
        if hit and time.monotonic() - hit[0] < _EE_CACHE_TTL_S:
            return hit[1]
        fut = _ee_inflight.get(key)
        if fut is None:
            fut = _ee_inflight[key] = _ee_pool.submit(_ee_job, key, lat, lon)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout:
        log.warning("EE query for %s exceeded %.0fs; continuing without it", key, timeout)
        return None

def _looks_like_water(res):
    """Spectral fallback when no land-cover label exists: open water has high NDWI and negative NDVI."""
    ndwi, ndvi = res.get("ndwi_latest"), res.get("ndvi_latest")
    return ndwi is not None and ndvi is not None and ndwi >= 0.3 and ndvi < 0.0

def analyze_land(lat, lon):
    """Routing: Planet data if it exists nearby, otherwise Earth Engine. NDVI/NDWI trends from EE when available."""
    planet = planet_lookup(lat, lon)
    ee_res = ee_analysis_cached(lat, lon)
    ndvi_series = (ee_res or {}).get("series", [])
    ndwi_series = (ee_res or {}).get("ndwi_series", [])
    if planet:
        res = dict(planet)
        if ee_res and ee_res.get("land_cover"): res["cross_check"] = {"dynamic_world": ee_res["land_cover"], "note": "independent live check"}
    elif ee_res and ee_res.get("land_cover"):
        res = {k: v for k, v in ee_res.items() if k not in ("series", "ndwi_series")}
    else:
        note = ("Earth Engine returned no land-cover label for this point." if ee_res
                else "No Planet data nearby and Earth Engine is unavailable or timed out.")
        res = {"source": "none", "land_cover": "unknown", "note": note}
    res["geohash"] = geohash(lat, lon)
    # NDVI
    res["ndvi_series"] = ndvi_series
    if ndvi_series: res["ndvi_latest"] = ndvi_series[-1]["ndvi"]
    elif planet and planet.get("ndvi_planet") is not None:
        res["ndvi_latest"] = planet["ndvi_planet"]
    # NDWI
    res["ndwi_series"] = ndwi_series
    if ndwi_series:
        res["ndwi_latest"] = ndwi_series[-1]["ndwi"]
    elif planet and planet.get("ndwi_planet") is not None:
        res["ndwi_latest"] = planet["ndwi_planet"]
    if res["source"] == "none" and _looks_like_water(res):
        # Spectral water test uses EE-derived NDWI/NDVI series but is NOT a confirmed EE
        # land-cover label. Keep the source distinct so callers cannot mistake this
        # heuristic for a verified Earth Engine result.
        res.update({"source": "spectral_heuristic", "land_cover_id": 0, "land_cover": LABELS["dynamic_world"]["0"],
                    "is_cropland": False, "resolution": "10 m Sentinel-2", "observed": ndwi_series[-1]["date"],
                    "label_origin": "Spectral water test (NDWI >= 0.3, NDVI < 0) — not a verified EE land-cover label"})
    return res
