"""
Evidence Bundle Builder.
Converts raw agricultural observations (satellite, soil, weather, model outputs)
into structured evidence items with provenance badges.

Each evidence item has:
  id, type, indicator, indicator_key, value, unit, source, date, interpretation, interpretation_key, badge, confidence
"""
import datetime

BADGES = {
    "live": "LIVE",
    "model": "MODEL",
    "precomputed": "PRECOMPUTED",
    "demo": "DEMO",
    "unavailable": "UNAVAILABLE",
}


def _today():
    return datetime.date.today().isoformat()


def _ndvi_interpretation(v):
    if v is None:
        return ("No vegetation data available.", "ndvi_no_data")
    if v >= 0.6:
        return ("Strong, healthy vegetation signal.", "ndvi_strong")
    if v >= 0.4:
        return ("Moderate vegetation → active growth likely.", "ndvi_moderate")
    if v >= 0.2:
        return ("Low vegetation → sparse or stressed cover.", "ndvi_low")
    return ("Very low vegetation → bare or severely stressed.", "ndvi_very_low")


def _ndwi_interpretation(v):
    if v is None:
        return ("No water index data available.", "ndwi_no_data")
    if v >= 0.3:
        return ("High water content → possible waterlogging.", "ndwi_high")
    if v >= 0.1:
        return ("Adequate moisture content.", "ndwi_adequate")
    if v >= 0.0:
        return ("Low moisture → irrigation may be needed.", "ndwi_low")
    return ("Very low moisture → significant water stress.", "ndwi_very_low")


def _rain_interpretation(mm, period="past 7 days"):
    p_key = "past" if period == "past 7 days" else "next"
    if mm is None:
        return ("Rainfall data unavailable.", f"rain_no_data_{p_key}")
    if mm < 2:
        return (f"Very low rainfall ({period}) → drought risk.", f"rain_very_low_{p_key}")
    if mm < 10:
        return (f"Low rainfall ({period}) → monitor moisture.", f"rain_low_{p_key}")
    if mm < 50:
        return (f"Moderate rainfall ({period}) → adequate for most crops.", f"rain_mod_{p_key}")
    if mm < 100:
        return (f"Heavy rainfall ({period}) → waterlogging risk.", f"rain_heavy_{p_key}")
    return (f"Very heavy rainfall ({period}) → flood risk.", f"rain_very_heavy_{p_key}")


def _temp_interpretation(avg_tmax):
    if avg_tmax is None:
        return ("Temperature data unavailable.", "temp_no_data")
    if avg_tmax > 42:
        return ("Extreme heat → severe crop stress likely.", "temp_extreme")
    if avg_tmax > 38:
        return ("High temperatures → heat stress possible.", "temp_high")
    if avg_tmax > 35:
        return ("Warm conditions → monitor heat-sensitive crops.", "temp_warm")
    if avg_tmax > 25:
        return ("Favorable temperature range for most crops.", "temp_favorable")
    if avg_tmax > 15:
        return ("Cool conditions → suitable for rabi crops.", "temp_cool")
    return ("Cold conditions → frost risk for sensitive crops.", "temp_cold")


def _ph_interpretation(ph):
    if ph is None:
        return ("Soil pH data unavailable.", "ph_no_data")
    if 6.0 <= ph <= 7.5:
        return (f"pH {ph} → optimal range for most crops.", "ph_optimal")
    if 5.5 <= ph <= 8.0:
        return (f"pH {ph} → slightly outside optimal; most crops tolerate this.", "ph_slightly_outside")
    return (f"pH {ph} → significantly acidic or alkaline; remediation may help.", "ph_significantly_outside")


def _soc_interpretation(soc):
    if soc is None:
        return ("Soil organic carbon data unavailable.", "soc_no_data")
    if soc > 25:
        return (f"SOC {soc} g/kg → high organic matter, excellent.", "soc_high")
    if soc > 15:
        return (f"SOC {soc} g/kg → good organic matter level.", "soc_good")
    if soc > 8:
        return (f"SOC {soc} g/kg → moderate; organic amendments could help.", "soc_mod")
    if soc > 3:
        return (f"SOC {soc} g/kg → low; regenerative practices recommended.", "soc_low")
    return (f"SOC {soc} g/kg → very low organic matter.", "soc_very_low")


def build_evidence(land: dict, weather: dict, soil: dict, health: dict,
                   is_demo: bool = False, snapshot_date: str = None) -> list:
    """Build a list of evidence items from raw data sources."""
    evidence = []
    counter = {"sat": 0, "weather": 0, "soil": 0, "land": 0, "spear": 0, "health": 0}

    base_badge = BADGES["demo"] if is_demo else None

    # --- Satellite / Vegetation ---
    ndvi = land.get("ndvi_latest")
    if ndvi is not None:
        src = land.get("source", "unknown")
        badge = base_badge or (BADGES["precomputed"] if src == "planet_highres" else BADGES["live"])
        counter["sat"] += 1
        i_txt, i_key = _ndvi_interpretation(ndvi)
        evidence.append({
            "id": f"SAT-NDVI-{counter['sat']:03d}",
            "type": "satellite",
            "indicator": "NDVI",
            "indicator_key": "ind_ndvi",
            "value": round(ndvi, 3),
            "unit": "index (0–1)",
            "source": "Sentinel-2" if src == "earth_engine_live" else "PlanetScope",
            "date": land.get("observed", (snapshot_date or _today())),
            "interpretation": i_txt,
            "interpretation_key": i_key,
            "badge": badge,
            "confidence": land.get("confidence"),
        })

    ndwi = land.get("ndwi_latest")
    if ndwi is not None:
        src = land.get("source", "unknown")
        badge = base_badge or (BADGES["precomputed"] if src == "planet_highres" else BADGES["live"])
        counter["sat"] += 1
        i_txt, i_key = _ndwi_interpretation(ndwi)
        evidence.append({
            "id": f"SAT-NDWI-{counter['sat']:03d}",
            "type": "satellite",
            "indicator": "NDWI",
            "indicator_key": "ind_ndwi",
            "value": round(ndwi, 3),
            "unit": "index (-1 to 1)",
            "source": "Sentinel-2" if src == "earth_engine_live" else "PlanetScope",
            "date": land.get("observed", (snapshot_date or _today())),
            "interpretation": i_txt,
            "interpretation_key": i_key,
            "badge": badge,
            "confidence": None,
        })

    # --- Land Cover ---
    lc = land.get("land_cover")
    if lc and lc != "unknown":
        src = land.get("source", "unknown")
        badge = base_badge or (BADGES["precomputed"] if src == "planet_highres" else BADGES["live"])
        counter["land"] += 1
        is_crop = land.get("is_cropland")
        i_txt = f"Classified as {lc}." + (" Identified as cropland." if is_crop else "")
        i_key = "lc_cropland" if is_crop else "lc_non_cropland"
        evidence.append({
            "id": f"LANDCOVER-{counter['land']:03d}",
            "type": "land_cover",
            "indicator": "Land Cover Class",
            "indicator_key": "ind_land_cover",
            "value": lc,
            "unit": "class",
            "source": "Dynamic World" if src == "earth_engine_live" else "Dynamic World (Planet cell label)",
            "date": land.get("observed", (snapshot_date or _today())),
            "interpretation": i_txt,
            "interpretation_key": i_key,
            "badge": badge,
            "confidence": land.get("confidence"),
        })

    # --- SPEAR ---
    mp = land.get("model_prediction")
    if mp:
        counter["spear"] += 1
        agrees = mp.get('agrees_with_label')
        conf = round(mp.get('confidence', 0) * 100)
        agrees_txt = 'agrees' if agrees else 'disagrees'
        i_txt = (
            f"SPEAR predicts {mp.get('land_cover')} "
            f"({agrees_txt} with reference label). "
            f"Confidence: {conf}%."
        )
        i_key = "spear_agrees" if agrees else "spear_disagrees"
        evidence.append({
            "id": f"SPEAR-{counter['spear']:03d}",
            "type": "model",
            "indicator": "SPEAR EO Second Opinion",
            "indicator_key": "ind_spear",
            "value": mp.get("land_cover", "unknown"),
            "unit": "class",
            "source": mp.get("model", "SPEAR PlanetScope MAE"),
            "date": land.get("observed", (snapshot_date or _today())),
            "interpretation": i_txt,
            "interpretation_key": i_key,
            "badge": base_badge or BADGES["model"],
            "confidence": mp.get("confidence"),
        })

    # --- Weather ---
    if not weather.get("error"):
        past_rain = weather.get("past7_rain_mm")
        if past_rain is not None:
            counter["weather"] += 1
            i_txt, i_key = _rain_interpretation(past_rain, "past 7 days")
            evidence.append({
                "id": f"WEATHER-RAIN-{counter['weather']:03d}",
                "type": "weather",
                "indicator": "Rainfall (past 7 days)",
                "indicator_key": "ind_rain_past",
                "value": past_rain,
                "unit": "mm",
                "source": "Open-Meteo",
                "date": (snapshot_date or _today()),
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

        next_rain = weather.get("next7_rain_mm")
        if next_rain is not None:
            counter["weather"] += 1
            i_txt, i_key = _rain_interpretation(next_rain, "next 7 days")
            evidence.append({
                "id": f"WEATHER-FRAIN-{counter['weather']:03d}",
                "type": "weather",
                "indicator": "Rainfall forecast (next 7 days)",
                "indicator_key": "ind_rain_next",
                "value": next_rain,
                "unit": "mm",
                "source": "Open-Meteo",
                "date": (snapshot_date or _today()),
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

        tmax_list = weather.get("tmax", [])
        recent_tmax = [t for t in tmax_list[-7:] if t is not None]
        if recent_tmax:
            avg = round(sum(recent_tmax) / len(recent_tmax), 1)
            counter["weather"] += 1
            i_txt, i_key = _temp_interpretation(avg)
            evidence.append({
                "id": f"WEATHER-TEMP-{counter['weather']:03d}",
                "type": "weather",
                "indicator": "Average max temperature (recent)",
                "indicator_key": "ind_temp",
                "value": avg,
                "unit": "°C",
                "source": "Open-Meteo",
                "date": (snapshot_date or _today()),
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

    # --- Soil ---
    top = soil.get("topsoil_0_5cm") if not soil.get("error") else None
    if top:
        ph = top.get("phh2o")
        if ph is not None:
            counter["soil"] += 1
            i_txt, i_key = _ph_interpretation(ph)
            evidence.append({
                "id": f"SOIL-PH-{counter['soil']:03d}",
                "type": "soil",
                "indicator": "pH",
                "indicator_key": "ind_ph",
                "value": ph,
                "unit": "pH",
                "source": "SoilGrids (ISRIC)",
                "date": "climatology",
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

        soc = top.get("soc")
        if soc is not None:
            counter["soil"] += 1
            i_txt, i_key = _soc_interpretation(soc)
            evidence.append({
                "id": f"SOIL-SOC-{counter['soil']:03d}",
                "type": "soil",
                "indicator": "Soil Organic Carbon",
                "indicator_key": "ind_soc",
                "value": soc,
                "unit": "g/kg",
                "source": "SoilGrids (ISRIC)",
                "date": "climatology",
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

        nitrogen = top.get("nitrogen")
        if nitrogen is not None:
            counter["soil"] += 1
            adeq = nitrogen > 1.0
            i_txt = f"Nitrogen {nitrogen} g/kg." + (" Adequate." if adeq else " Low → organic sources recommended.")
            i_key = "n_adequate" if adeq else "n_low"
            evidence.append({
                "id": f"SOIL-N-{counter['soil']:03d}",
                "type": "soil",
                "indicator": "Nitrogen",
                "indicator_key": "ind_nitrogen",
                "value": nitrogen,
                "unit": "g/kg",
                "source": "SoilGrids (ISRIC)",
                "date": "climatology",
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

        clay = top.get("clay")
        if clay is not None:
            counter["soil"] += 1
            good_whc = clay > 200
            i_txt = f"Clay {clay} g/kg." + (" Good water holding." if good_whc else " Sandy tendency → drainage is high.")
            i_key = "clay_good" if good_whc else "clay_sandy"
            evidence.append({
                "id": f"SOIL-CLAY-{counter['soil']:03d}",
                "type": "soil",
                "indicator": "Clay content",
                "indicator_key": "ind_clay",
                "value": clay,
                "unit": "g/kg",
                "source": "SoilGrids (ISRIC)",
                "date": "climatology",
                "interpretation": i_txt,
                "interpretation_key": i_key,
                "badge": base_badge or BADGES["live"],
                "confidence": None,
            })

    # --- Field Health ---
    if health.get("score") is not None:
        counter["health"] += 1
        i_txt = (
            f"Overall field health: {health['score']}/100 ({health.get('status', 'unknown').upper()}). "
            f"Components: " + ", ".join(
                f"{k}={v}" for k, v in health.get("components", {}).items() if v is not None
            ) + "."
        )
        i_key = "health_score"
        evidence.append({
            "id": f"HEALTH-{counter['health']:03d}",
            "type": "computed",
            "indicator": "Field Health Score",
            "indicator_key": "ind_health",
            "value": health["score"],
            "unit": "/ 100",
            "source": "AgriNet Field Health Engine",
            "date": (snapshot_date or _today()),
            "interpretation": i_txt,
            "interpretation_key": i_key,
            "badge": base_badge or BADGES["model"],
            "confidence": None,
        })

    return evidence
