"""
Field State Assembler.
Creates the central Field State representation from raw data sources.
This is the primary object flowing through the AgriNet pipeline.

FIELD → DATA → FIELD STATE → EVIDENCE → GEMINI → ADVISORY
"""
import datetime
from . import evidence as ev
from . import field_health as fh


def _today():
    return datetime.date.today().isoformat()


def assemble_field_state(
    lat: float,
    lon: float,
    land: dict,
    weather: dict,
    soil: dict,
    crop: str = "",
    is_demo: bool = False,
    scenario_key: str = "",
) -> dict:
    """Assemble a complete Field State from raw data sources.

    Returns a JSON-serializable dictionary containing all field information,
    health score, and evidence bundle.
    """

    # --- Determine Snapshot Date ---
    is_agra = (is_demo and scenario_key == "agra_wheat")
    snapshot_date = "2024-02-15" if is_agra else _today()

    # --- Compute Field Health (deterministic, NOT Gemini) ---
    health = fh.compute_field_health(land, weather, soil)

    # --- Build Evidence Bundle ---
    evidence_bundle = ev.build_evidence(land, weather, soil, health, is_demo=is_demo, snapshot_date=snapshot_date)

    # --- Extract key indicators ---
    ndvi = land.get("ndvi_latest")
    ndwi = land.get("ndwi_latest")
    ndvi_series = land.get("ndvi_series", [])
    ndwi_series = land.get("ndwi_series", [])

    # --- Weather summary ---
    weather_summary = None
    if not weather.get("error"):
        tmax_list = weather.get("tmax", [])
        tmin_list = weather.get("tmin", [])
        recent_tmax = [t for t in tmax_list[-7:] if t is not None]
        recent_tmin = [t for t in tmin_list[-7:] if t is not None]
        weather_summary = {
            "past7_rain_mm": weather.get("past7_rain_mm"),
            "next7_rain_mm": weather.get("next7_rain_mm"),
            "avg_tmax_recent": round(sum(recent_tmax) / len(recent_tmax), 1) if recent_tmax else None,
            "avg_tmin_recent": round(sum(recent_tmin) / len(recent_tmin), 1) if recent_tmin else None,
            "days": weather.get("days", []),
            "tmax": tmax_list,
            "tmin": tmin_list,
            "rain_mm": weather.get("rain_mm", []),
            "source": "Open-Meteo",
            "status": "available",
        }
    else:
        weather_summary = {
            "status": "unavailable",
            "error": weather.get("error", "Weather data unavailable"),
        }

    # --- Soil summary ---
    soil_summary = None
    top = soil.get("topsoil_0_5cm") if not soil.get("error") else None
    if top:
        soil_summary = {
            "ph": top.get("phh2o"),
            "soc": top.get("soc"),
            "nitrogen": top.get("nitrogen"),
            "clay": top.get("clay"),
            "sand": top.get("sand"),
            "depth": "0-5cm",
            "source": soil.get("source", "SoilGrids (ISRIC)"),
            "status": "available",
        }
    else:
        soil_summary = {
            "status": "unavailable",
            "error": soil.get("error", "Soil data unavailable"),
        }

    # --- SPEAR summary ---
    spear_summary = None
    mp = land.get("model_prediction")
    if mp:
        spear_summary = {
            "predicted_class": mp.get("land_cover"),
            "confidence": mp.get("confidence"),
            "agrees_with_reference": mp.get("agrees_with_label"),
            "top3": mp.get("top3", []),
            "model": mp.get("model", "SPEAR PlanetScope MAE"),
            "reference_label": land.get("land_cover"),
            "note": "SPEAR provides an additional Earth-observation representation signal used as contextual evidence alongside satellite observations.",
        }

    # --- Provenance ---
    source = land.get("source", "none")
    provenance = {
        "satellite_source": source,
        "satellite_badge": (
            "DEMO" if is_demo
            else "PRECOMPUTED" if source == "planet_highres"
            else "LIVE" if source == "earth_engine_live"
            else "UNAVAILABLE"
        ),
        "weather_badge": "DEMO" if is_demo else ("LIVE" if not weather.get("error") else "UNAVAILABLE"),
        "soil_badge": "DEMO" if is_demo else ("LIVE" if top else "UNAVAILABLE"),
        "spear_badge": "DEMO" if is_demo else ("MODEL" if mp else "UNAVAILABLE"),
        "health_badge": "DEMO" if is_demo else ("MODEL" if health.get("score") is not None else "UNAVAILABLE"),
        "timestamp": snapshot_date,
    }

    if is_demo:
        provenance["data_status"] = "DEMO"
        provenance["live"] = False
        if scenario_key:
            provenance["scenario_key"] = scenario_key
            if is_agra:
                provenance["scenario_date"] = "15 Feb 2024"

    # --- Assemble ---
    field_state = {
        "field": {
            "lat": lat,
            "lon": lon,
            "geohash": land.get("geohash", ""),
            "crop": crop,
        },
        "land_cover": {
            "class": land.get("land_cover", "unknown"),
            "id": land.get("land_cover_id"),
            "is_cropland": land.get("is_cropland", False),
            "source": source,
            "resolution": land.get("resolution", ""),
            "observed": land.get("observed", ""),
            "distance_m": land.get("distance_m"),
            "label_origin": land.get("label_origin", ""),
        },
        "vegetation": {
            "ndvi": ndvi,
            "ndwi": ndwi,
            "ndvi_series": ndvi_series[-10:] if ndvi_series else [],
            "ndwi_series": ndwi_series[-10:] if ndwi_series else [],
        },
        "weather": weather_summary,
        "soil": soil_summary,
        "health": health,
        "spear": spear_summary,
        "cross_check": land.get("cross_check"),
        "evidence": evidence_bundle,
        "provenance": provenance,
        "is_demo": is_demo,
    }

    return field_state


def field_state_for_gemini(fs: dict) -> dict:
    """Create a compact version of the field state suitable for Gemini prompting.
    Strips large series and keeps only summary data."""
    compact = {
        "field": fs["field"],
        "land_cover": fs["land_cover"]["class"],
        "is_cropland": fs["land_cover"]["is_cropland"],
        "ndvi": fs["vegetation"]["ndvi"],
        "ndwi": fs["vegetation"]["ndwi"],
    }

    w = fs.get("weather", {})
    if w.get("status") == "available":
        compact["weather"] = {
            "past7_rain_mm": w.get("past7_rain_mm"),
            "next7_rain_mm": w.get("next7_rain_mm"),
            "avg_tmax": w.get("avg_tmax_recent"),
            "avg_tmin": w.get("avg_tmin_recent"),
            "rain_next7": w.get("rain_mm", [])[7:] if len(w.get("rain_mm", [])) > 7 else [],
            "tmax_next7": w.get("tmax", [])[7:] if len(w.get("tmax", [])) > 7 else [],
        }
    else:
        compact["weather"] = {"status": "unavailable"}

    s = fs.get("soil", {})
    if s.get("status") == "available":
        compact["soil"] = {k: v for k, v in s.items() if k not in ("status", "source", "depth")}
    else:
        compact["soil"] = {"status": "unavailable"}

    h = fs.get("health", {})
    if h.get("score") is not None:
        compact["health_score"] = h["score"]
        compact["health_status"] = h["status"]
        compact["health_components"] = h["components"]

    if fs.get("spear"):
        compact["spear"] = {
            "prediction": fs["spear"]["predicted_class"],
            "confidence": fs["spear"]["confidence"],
            "agrees": fs["spear"]["agrees_with_reference"],
        }

    # Include evidence summaries (IDs + interpretations only — Gemini references these)
    compact["evidence"] = [
        {"id": e["id"], "indicator": e["indicator"], "value": e["value"],
         "interpretation": e["interpretation"]}
        for e in fs.get("evidence", [])
    ]

    return compact
