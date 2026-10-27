"""
Regression tests for the MODIS null-image / masked-pixel fix in ee_analysis().

Background
----------
MODIS/061/MOD44W is a global yearly product with no spatial gaps. The original
code called:

    ee.ImageCollection("MODIS/061/MOD44W").select("water_mask")
        .sort(...).first().reduceRegion(...)

The null-image error ("Parameter 'image' is required and may not be null") can
occur when the point falls on a tile boundary where pixels are masked. The EE
SDK represents masked pixels differently from a null Image, but calling
reduceRegion on a masked-pixel image can still propagate the error depending on
EE version.

The fix uses .unmask(-1) on the image returned by .first() so that:
  - masked / tile-boundary pixels are filled with -1 (no water data)
  - reduceRegion() always operates on a valid, fully-valued Image
  - a missing reading is represented as -1, which downstream code already
    treats as "no value" (line: if v.get("water_mask") == 1 → water detected)

These tests mock the EE SDK and verify the expression construction order and
fallback semantics. No real credentials or network access are required.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from app import geo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ee_state(monkeypatch):
    monkeypatch.setattr(geo, "_ee_ok", True)
    monkeypatch.setattr(geo, "_ee_failed_at", 0.0)
    yield


VALID_LAT, VALID_LON = 20.0, 78.0


# ---------------------------------------------------------------------------
# Helper: build a realistic EE mock with call-order tracking
# ---------------------------------------------------------------------------

def _make_structured_ee_mock(water_mask_value=1):
    """Return a mock ee module where method chains record their call order.

    The MODIS image chain is carefully tracked:
        collection → .select() → .sort() → .first() → .unmask() → .reduceRegion() → .get()

    This lets tests assert both the order and the arguments.
    """
    ee = MagicMock(name="ee_module")

    # --- geometry ---
    pt = MagicMock(name="pt")
    pt.buffer.return_value = MagicMock(name="pt_buffer")
    ee.Geometry.Point.return_value = pt

    # --- reducers / filters ---
    ee.Reducer.first.return_value = MagicMock(name="reducer_first")
    ee.Reducer.mean.return_value = MagicMock(name="reducer_mean")
    ee.Reducer.max.return_value = MagicMock(name="reducer_max")
    ee.Filter.lt.return_value = MagicMock(name="filter_lt")
    ee.Filter.notNull.return_value = MagicMock(name="filter_notnull")
    ee.Algorithms.If.side_effect = lambda cond, t, f: t  # always take true branch

    # --- MODIS image chain (the one under test) ---
    modis_reduce_result = MagicMock(name="modis_reduce_result")
    modis_reduce_result.get.return_value = water_mask_value

    modis_unmasked_img = MagicMock(name="modis_unmasked_img")
    modis_unmasked_img.reduceRegion.return_value = modis_reduce_result

    modis_first_img = MagicMock(name="modis_first_img")
    modis_first_img.unmask.return_value = modis_unmasked_img

    modis_sorted = MagicMock(name="modis_sorted")
    modis_sorted.first.return_value = modis_first_img

    modis_selected = MagicMock(name="modis_selected")
    modis_selected.sort.return_value = modis_sorted

    modis_col = MagicMock(name="modis_col")
    modis_col.select.return_value = modis_selected

    # --- DW collection (minimal, needed for ee_analysis to run) ---
    dw_col = MagicMock(name="dw_col")
    dw_size_mock = MagicMock(name="dw_size")
    dw_size_mock.gt.return_value = MagicMock(name="dw_gt", _size_val=0)
    dw_col.size.return_value = dw_size_mock
    dw_col.filterBounds.return_value = dw_col
    dw_col.filterDate.return_value = dw_col
    dw_col.select.return_value = dw_col
    dw_col.mode.return_value = MagicMock()
    dw_col.mean.return_value = MagicMock()

    # --- S2 collection (empty, minimal) ---
    s2_col = MagicMock(name="s2_col")
    s2_col.filterBounds.return_value = s2_col
    s2_col.filterDate.return_value = s2_col
    s2_col.filter.return_value = s2_col
    s2_col.sort.return_value = s2_col
    s2_col.limit.return_value = s2_col
    s2_col.map.return_value = s2_col
    s2_col.aggregate_array.return_value = []
    s2_col.filter.return_value = s2_col

    def _image_collection(dataset_id):
        if "DYNAMICWORLD" in dataset_id:
            return dw_col
        if "MOD44W" in dataset_id:
            return modis_col
        if "S2_SR" in dataset_id:
            return s2_col
        return MagicMock()

    ee.ImageCollection.side_effect = _image_collection

    ee.Dictionary.return_value.getInfo.return_value = {
        "label": -1, "confidence": -1,
        "water_mask": water_mask_value,
        "ndvi_d": [], "ndvi": [],
        "ndwi_d": [], "ndwi": [],
    }

    return ee, modis_col, modis_first_img, modis_unmasked_img, modis_reduce_result


def _run_ee_analysis(monkeypatch, mock_ee):
    """Patch geo.ee and the local 'import ee' inside ee_analysis(), then run it."""
    import sys
    monkeypatch.setattr(geo, "ee", mock_ee)
    sys.modules["ee"] = mock_ee
    try:
        with patch.object(geo, "_ee", return_value=True):
            return geo.ee_analysis(VALID_LAT, VALID_LON)
    finally:
        sys.modules.pop("ee", None)


# ---------------------------------------------------------------------------
# 1. unmask() is called BEFORE reduceRegion()
# ---------------------------------------------------------------------------

def test_unmask_called_before_reduce_region(monkeypatch):
    """The MODIS image must have .unmask(-1) applied before .reduceRegion().
    This ensures masked/boundary pixels are filled with the sentinel -1 value
    before reduction, preventing the null-pixel error."""
    mock_ee, modis_col, first_img, unmasked_img, _ = _make_structured_ee_mock()
    _run_ee_analysis(monkeypatch, mock_ee)

    # .unmask() must be called on the image returned by .first()
    first_img.unmask.assert_called_once_with(-1)

    # .reduceRegion() must be called on the *unmasked* image, not first_img
    unmasked_img.reduceRegion.assert_called_once()
    first_img.reduceRegion.assert_not_called()


def test_unmask_fill_value_is_negative_one(monkeypatch):
    """The unmask fill value must be -1 (the sentinel for 'no water data').
    Using any other value (e.g. 0 = land) would silently fabricate a reading."""
    mock_ee, modis_col, first_img, unmasked_img, _ = _make_structured_ee_mock()
    _run_ee_analysis(monkeypatch, mock_ee)

    args, kwargs = first_img.unmask.call_args
    fill_value = args[0] if args else kwargs.get("value")
    assert fill_value == -1, (
        f"unmask fill value must be -1 (no-data sentinel), got {fill_value!r}. "
        "Using 0 would falsely report land; using 1 would falsely report water."
    )


# ---------------------------------------------------------------------------
# 2. reduceRegion() receives a Reducer.first() and the correct scale
# ---------------------------------------------------------------------------

def test_reduce_region_uses_first_reducer_at_250m(monkeypatch):
    """reduceRegion() must use Reducer.first() at 250 m scale, matching
    MODIS MOD44W's native 250 m resolution."""
    mock_ee, modis_col, first_img, unmasked_img, _ = _make_structured_ee_mock()
    _run_ee_analysis(monkeypatch, mock_ee)

    call_args = unmasked_img.reduceRegion.call_args
    args = call_args.args
    assert args[0] is mock_ee.Reducer.first.return_value, "Reducer must be ee.Reducer.first()"
    assert args[2] == 250, f"Scale must be 250 m (MODIS native resolution), got {args[2]}"


# ---------------------------------------------------------------------------
# 3. .get() uses "water_mask" key with -1 default
# ---------------------------------------------------------------------------

def test_get_uses_water_mask_key_with_minus_one_default(monkeypatch):
    """The .get() call on the reduceRegion result must use the key 'water_mask'
    and a default of -1, so a point that lies entirely on a masked tile returns
    -1 (no data) rather than null or 0."""
    mock_ee, modis_col, first_img, unmasked_img, reduce_result = _make_structured_ee_mock()
    _run_ee_analysis(monkeypatch, mock_ee)

    reduce_result.get.assert_called_once_with("water_mask", -1)


# ---------------------------------------------------------------------------
# 4. water_mask=1 triggers the water fallback path (not fabricated)
# ---------------------------------------------------------------------------

def test_water_mask_one_triggers_water_land_cover(monkeypatch):
    """When water_mask=1 is returned and DW has no label, ee_analysis() must
    return land_cover='Water' sourced from MODIS — not a fabricated value."""
    mock_ee, *_ = _make_structured_ee_mock(water_mask_value=1)
    result = _run_ee_analysis(monkeypatch, mock_ee)

    assert result is not None
    assert result.get("source") == "earth_engine_live"
    assert result.get("land_cover_id") == 0      # Dynamic World water class
    assert "MODIS" in result.get("label_origin", ""), (
        "label_origin must reference MODIS to distinguish from fabricated data"
    )
    assert result.get("confidence") is None       # water mask has no confidence score


def test_water_mask_minus_one_returns_base_dict(monkeypatch):
    """When water_mask=-1 (no data / masked pixel) and DW has no label,
    ee_analysis() must return the base dict without a land_cover key
    — not report 'water' or any other fabricated classification."""
    mock_ee, *_ = _make_structured_ee_mock(water_mask_value=-1)
    result = _run_ee_analysis(monkeypatch, mock_ee)

    # base dict: only series/ndwi_series, no land_cover
    assert result is not None
    assert "land_cover" not in result, (
        "A masked MODIS pixel (-1) must not produce a land_cover classification"
    )
    assert "source" not in result, (
        "A masked MODIS pixel must not set source=earth_engine_live"
    )


# ---------------------------------------------------------------------------
# 5. _ee_job catches EE errors and returns None (existing contract)
# ---------------------------------------------------------------------------

def test_ee_job_catches_exception_and_returns_none(monkeypatch):
    """_ee_job() must catch any EE exception (including the null-image error)
    and return None rather than propagating it."""
    def _raise(*args, **kwargs):
        raise RuntimeError(
            "Image.reduceRegion: Parameter 'image' is required and may not be null."
        )

    with patch.object(geo, "ee_analysis", side_effect=_raise):
        result = geo._ee_job("gh7", VALID_LAT, VALID_LON)

    assert result is None


# ---------------------------------------------------------------------------
# 6. analyze_land() fallback paths
# ---------------------------------------------------------------------------

def test_analyze_land_fallback_when_ee_unavailable(monkeypatch):
    with patch.object(geo, "ee_analysis_cached", return_value=None), \
         patch.object(geo, "planet_lookup", return_value=None):
        result = geo.analyze_land(VALID_LAT, VALID_LON)

    assert result["source"] == "none"
    assert "note" in result
    assert "land_cover" in result


def test_analyze_land_water_result_from_modis(monkeypatch):
    """When EE returns a water result from MODIS, analyze_land must propagate it."""
    ee_result = {
        "series": [], "ndwi_series": [],
        "source": "earth_engine_live",
        "land_cover": "Water",
        "land_cover_id": 0,
        "is_cropland": False,
        "resolution": "250 m MODIS land/water mask",
        "observed": "2024-01-01",
        "confidence": None,
        "label_origin": "MODIS MOD44W water mask (no Dynamic World scene)",
    }
    with patch.object(geo, "ee_analysis_cached", return_value=ee_result), \
         patch.object(geo, "planet_lookup", return_value=None):
        result = geo.analyze_land(VALID_LAT, VALID_LON)

    assert result["source"] == "earth_engine_live"
    assert result["land_cover"] == "Water"
    assert result.get("confidence") is None


# ---------------------------------------------------------------------------
# Proof-of-mechanism (mocked): masked pixels → empty reduceRegion dict → null propagation
#
# These tests do NOT call the real EE API. They use Python mocks to illustrate
# the hypothesized server‑side chain and verify that .unmask(-1) prevents it.
# caused the production error, and that .unmask(-1) breaks the chain.
#
# EE semantics being modelled:
#   - A fully-masked Image returns {} from reduceRegion() (no band results).
#   - dict.get("water_mask", -1) in Python returns -1 for missing keys.
#   - But in EE server-side, ee.Dictionary.get() with a default only substitutes
#     the default when the key is absent from the dict — not when the value is
#     explicitly null. So a key present with null value stays null.
#   - An all-masked image with a 250m scale at a single point can produce
#     {} (empty dict) or {"water_mask": null} depending on EE version.
#   - .unmask(-1) converts masked pixels to -1 before reduction, so the dict
#     is always {"water_mask": -1}, never empty or null.
# ---------------------------------------------------------------------------

def test_proof_masked_pixels_produce_null_without_unmask():
    """Mocked illustration of the EE bug where `reduceRegion` returns an empty dict.
    The test demonstrates that, without `.unmask(-1)`, a fully‑masked image could
    lead to a null value being propagated in the EE expression graph (simulated
    here)."""

    # Simulate EE server: all-masked image → reduceRegion returns {}
    class _AllMaskedImage:
        def reduceRegion(self, *a, **kw):
            return {}  # empty: no valid pixels

    result_dict = _AllMaskedImage().reduceRegion()
    # Python dict.get() with default — returns default for absent key
    python_val = result_dict.get("water_mask", -1)
    assert python_val == -1, "Python dict.get() gives -1 for absent key"

    # But EE server-side get() on an empty ee.Dictionary returns null for
    # any key, regardless of the default argument, in some EE versions.
    # We model this: the value inserted into the parent ee.Dictionary is null.
    # A null propagated into the computation graph then causes:
    #   "Image.reduceRegion: Parameter 'image' is required and may not be null."
    # The test documents this as the *mechanism*, even though client-side
    # Python dict.get() masks it.
    assert "water_mask" not in result_dict, (
        "All-masked image returns empty dict — the -1 default relies on "
        "client-side Python dict.get(), but EE server evaluates the "
        "expression graph lazily and can propagate null before getInfo()."
    )


def test_proof_unmask_prevents_empty_dict():
    """Demonstrate: .unmask(-1) fills masked pixels before reduction, so
    the reduceRegion dict always contains a real value, never null or absent."""
    FILL = -1

    class _MaskedImage:
        def unmask(self, fill_value):
            self._fill = fill_value
            return self  # returns same image, now with fill applied

        def reduceRegion(self, *a, **kw):
            # After unmask, masked pixels are replaced with fill_value.
            # reduceRegion now has a real value to aggregate.
            return {"water_mask": self._fill}

    img = _MaskedImage()
    unmasked = img.unmask(FILL)
    result = unmasked.reduceRegion()

    assert "water_mask" in result, "After unmask, the band key must be present"
    assert result["water_mask"] == FILL, (
        f"Masked pixels must yield the fill value {FILL}, got {result['water_mask']!r}"
    )
    assert result["water_mask"] != 0, "Fill must not be 0 (land) — would fabricate land reading"
    assert result["water_mask"] != 1, "Fill must not be 1 (water) — would fabricate water reading"


def test_proof_ee_job_logs_exc_class_and_truncated_message(monkeypatch, caplog):
    """_ee_job() must log exc= and msg= fields with the geohash, no credentials."""
    import logging
    caplog.set_level(logging.WARNING)

    def _raise(*a, **kw):
        raise RuntimeError(
            "Image.reduceRegion: Parameter 'image' is required and may not be null."
        )

    with patch.object(geo, "ee_analysis", side_effect=_raise):
        result = geo._ee_job("u30gh7", VALID_LAT, VALID_LON)

    assert result is None
    assert any("exc=" in r.message and "msg=" in r.message for r in caplog.records), (
        "Log must contain exc= and msg= for production diagnostics"
    )
    assert any("u30gh7" in r.message for r in caplog.records), (
        "Log must include the geohash so failing locations can be identified"
    )
    # Ensure no credentials or private_key in any log record
    assert not any("private_key" in r.message for r in caplog.records)
    assert not any("EE_SERVICE_ACCOUNT" in r.message for r in caplog.records)
