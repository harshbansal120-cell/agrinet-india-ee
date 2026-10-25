"""
Production-readiness tests for Earth Engine integration.

Covers:
  - Malformed / invalid service-account JSON
  - Missing project configuration
  - ee.Initialize() failure (non-Runtime exception)
  - ee_config_status() health-response shape and semantics
  - analyze_land() source tagging (planet, earth_engine_live, spectral_heuristic, none)
  - EE errors must never be surfaced as successful Earth Engine results
  - /health/earth-engine HTTP endpoint shape

All external Earth Engine calls are mocked; no network traffic is made.
"""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_env(*names: str):
    """Remove named env vars if present; call from setUp."""
    for n in names:
        os.environ.pop(n, None)


def _reset_geo(geo_module):
    """Reset geo module EE state between tests."""
    geo_module._ee_ok = None
    geo_module._ee_failed_at = 0.0


_EE_ENV_VARS = ("EE_PROJECT_ID", "EE_PROJECT", "EE_SERVICE_ACCOUNT_JSON")

_VALID_SA = {
    "type": "service_account",
    "project_id": "my-project",
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nFAKE\n-----END RSA PRIVATE KEY-----\n",
    "client_email": "sa@my-project.iam.gserviceaccount.com",
}


# ===========================================================================
# 1.  _ee() — init / auth path
# ===========================================================================
class TestEEInit(unittest.TestCase):
    def setUp(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def tearDown(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    # -----------------------------------------------------------------------
    # 1a. Malformed SA JSON — not parseable
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_malformed_sa_json_not_parseable(self, mock_ee):
        """ee._ee() must return False when SA JSON is not valid JSON, without raising."""
        os.environ["EE_PROJECT_ID"] = "proj"
        os.environ["EE_SERVICE_ACCOUNT_JSON"] = "NOT { valid json }"

        from app import geo
        _reset_geo(geo)

        result = geo._ee()
        self.assertFalse(result)
        self.assertFalse(geo._ee_ok)
        mock_ee.Initialize.assert_not_called()

    # -----------------------------------------------------------------------
    # 1b. Malformed SA JSON — valid JSON but missing required fields
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_malformed_sa_json_missing_fields(self, mock_ee):
        """ee._ee() must return False when SA JSON lacks required keys, without raising."""
        os.environ["EE_PROJECT_ID"] = "proj"
        incomplete = {"type": "service_account"}  # missing project_id, private_key, client_email
        os.environ["EE_SERVICE_ACCOUNT_JSON"] = json.dumps(incomplete)

        from app import geo
        _reset_geo(geo)

        result = geo._ee()
        self.assertFalse(result)
        self.assertFalse(geo._ee_ok)
        mock_ee.Initialize.assert_not_called()

    # -----------------------------------------------------------------------
    # 1c. Missing project configuration (no EE_PROJECT_ID, no EE_PROJECT)
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_missing_project_raises(self, mock_ee):
        """ee._ee() must raise RuntimeError when no project is configured."""
        # No env vars set at all
        from app import geo
        _reset_geo(geo)

        with self.assertRaises(RuntimeError) as ctx:
            geo._ee()
        msg = str(ctx.exception).lower()
        self.assertTrue(
            "ee_project" in msg or "project" in msg,
            f"Expected project-related error message, got: {ctx.exception}",
        )
        mock_ee.Initialize.assert_not_called()

    # -----------------------------------------------------------------------
    # 1d. Init failure — ee.Initialize raises a non-Runtime exception
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_init_failure_non_runtime(self, mock_ee):
        """A non-RuntimeError from ee.Initialize must set _ee_ok=False (not re-raise)."""
        os.environ["EE_PROJECT_ID"] = "proj"
        mock_ee.Initialize.side_effect = ConnectionError("network failure")

        from app import geo
        _reset_geo(geo)

        # _ee() must NOT re-raise non-RuntimeError; it logs and returns False
        result = geo._ee()
        self.assertFalse(result)
        self.assertFalse(geo._ee_ok)

    # -----------------------------------------------------------------------
    # 1e. Valid service-account JSON — happy path
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_valid_sa_json(self, mock_ee):
        """Valid SA JSON and project -> Initialize called with credentials."""
        os.environ["EE_PROJECT_ID"] = "my-project"
        os.environ["EE_SERVICE_ACCOUNT_JSON"] = json.dumps(_VALID_SA)
        mock_ee.ServiceAccountCredentials.return_value = MagicMock()
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)

        result = geo._ee()
        self.assertTrue(result)
        mock_ee.Initialize.assert_called_once()

    # -----------------------------------------------------------------------
    # 1f. Project-only path (no SA JSON)
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_project_only_init(self, mock_ee):
        """Without SA JSON, Initialize is called with project kwarg only."""
        os.environ["EE_PROJECT_ID"] = "proj-only"
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)

        result = geo._ee()
        self.assertTrue(result)
        mock_ee.Initialize.assert_called_once_with(project="proj-only")
        mock_ee.ServiceAccountCredentials.assert_not_called()

    # -----------------------------------------------------------------------
    # 1g. _ee_ok=False is retried after cool-down (not permanent)
    # -----------------------------------------------------------------------
    @patch("app.geo.ee")
    def test_retry_after_cooldown(self, mock_ee):
        """After the cool-down period, a previously-failed init may be reattempted."""
        import time

        os.environ["EE_PROJECT_ID"] = "proj"
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)
        # Simulate a prior failure that is now beyond the retry window
        geo._ee_ok = False
        geo._ee_failed_at = time.monotonic() - geo._EE_RETRY_S - 1

        result = geo._ee()
        self.assertTrue(result)


# ===========================================================================
# 2.  ee_config_status() — health response semantics
# ===========================================================================
class TestEEConfigStatus(unittest.TestCase):
    def setUp(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def tearDown(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def _status(self):
        from app import geo
        return geo.ee_config_status()

    def test_required_keys_present(self):
        """Health status must contain exactly the documented generic fields."""
        s = self._status()
        for key in ("sdk_available", "sdk_initialized", "configuration_present", "note"):
            self.assertIn(key, s, f"Missing key: {key}")

    def test_no_secrets_in_status(self):
        """Status must never expose env values, project IDs, or private keys."""
        os.environ["EE_PROJECT_ID"] = "secret-project-id"
        os.environ["EE_SERVICE_ACCOUNT_JSON"] = json.dumps(_VALID_SA)
        s = self._status()
        payload = json.dumps(s)
        self.assertNotIn("secret-project-id", payload)
        self.assertNotIn("FAKE", payload)           # private key content
        self.assertNotIn("sa@my-project", payload)  # client_email

    def test_sdk_initialized_false_when_not_initialized(self):
        """sdk_initialized must be False when _ee() has never succeeded."""
        from app import geo
        geo._ee_ok = None  # uninitialised
        s = self._status()
        self.assertFalse(s["sdk_initialized"])

    def test_sdk_initialized_false_when_failed(self):
        """sdk_initialized must be False when _ee() previously failed."""
        from app import geo
        geo._ee_ok = False
        s = self._status()
        self.assertFalse(s["sdk_initialized"])

    @patch("app.geo.ee")
    def test_sdk_initialized_true_after_success(self, mock_ee):
        """sdk_initialized must be True only after ee.Initialize() succeeded."""
        os.environ["EE_PROJECT_ID"] = "proj"
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)
        geo._ee()  # trigger actual init path

        s = geo.ee_config_status()
        self.assertTrue(s["sdk_initialized"])

    def test_note_distinguishes_init_from_reachability(self):
        """The note field must clarify that sdk_initialized != remote EE reachability."""
        s = self._status()
        note_lower = s["note"].lower()
        self.assertTrue(
            "does not guarantee" in note_lower or "not guarantee" in note_lower
            or "does not prove" in note_lower or "not prove" in note_lower,
            f"Note does not clearly disclaim remote reachability: {s['note']}",
        )

    def test_configuration_present_false_without_env(self):
        """configuration_present must be False when no project env vars are set."""
        s = self._status()
        self.assertFalse(s["configuration_present"])

    def test_configuration_present_true_with_ee_project(self):
        """configuration_present must be True when EE_PROJECT is set."""
        os.environ["EE_PROJECT"] = "some-proj"
        s = self._status()
        self.assertTrue(s["configuration_present"])

    def test_configuration_present_true_with_ee_project_id(self):
        """configuration_present must be True when EE_PROJECT_ID is set."""
        os.environ["EE_PROJECT_ID"] = "some-proj"
        s = self._status()
        self.assertTrue(s["configuration_present"])

    def test_all_values_are_generic_types(self):
        """All status values must be bool or str — no complex objects."""
        s = self._status()
        for k, v in s.items():
            self.assertIsInstance(
                v, (bool, str),
                f"Field '{k}' has unexpected type {type(v).__name__}: {v!r}",
            )


# ===========================================================================
# 3.  /health/earth-engine HTTP endpoint
# ===========================================================================
class TestEEHealthEndpoint(unittest.TestCase):
    def setUp(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def tearDown(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def _client(self):
        from app.main import app
        return TestClient(app)

    def test_endpoint_returns_200(self):
        resp = self._client().get("/health/earth-engine")
        self.assertEqual(resp.status_code, 200)

    def test_endpoint_response_shape(self):
        resp = self._client().get("/health/earth-engine")
        body = resp.json()
        for key in ("sdk_available", "sdk_initialized", "configuration_present", "note"):
            self.assertIn(key, body, f"Missing key in /health/earth-engine response: {key}")

    def test_endpoint_sdk_initialized_false_without_config(self):
        """Without any EE config, sdk_initialized must be False in HTTP response."""
        resp = self._client().get("/health/earth-engine")
        body = resp.json()
        self.assertFalse(body["sdk_initialized"])

    def test_endpoint_no_secrets_in_body(self):
        """HTTP response must never contain secret values even if env vars are set."""
        os.environ["EE_PROJECT_ID"] = "hidden-project"
        resp = self._client().get("/health/earth-engine")
        body_text = resp.text
        self.assertNotIn("hidden-project", body_text)

    @patch("app.geo.ee")
    def test_endpoint_sdk_initialized_true_after_init(self, mock_ee):
        """sdk_initialized in HTTP response is True only after successful ee.Initialize."""
        os.environ["EE_PROJECT_ID"] = "proj"
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)
        geo._ee()  # force init so state is set

        resp = self._client().get("/health/earth-engine")
        self.assertTrue(resp.json()["sdk_initialized"])

    def test_endpoint_note_is_non_empty_string(self):
        resp = self._client().get("/health/earth-engine")
        note = resp.json().get("note", "")
        self.assertIsInstance(note, str)
        self.assertGreater(len(note), 0)


# ===========================================================================
# 4.  analyze_land() — source tagging and fallback correctness
# ===========================================================================
class TestAnalyzeLandSources(unittest.TestCase):
    """
    Verify that analyze_land() never mislabels the data source and that
    an Earth Engine error cannot be returned as a successful EE result.
    """

    def setUp(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def tearDown(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    # -----------------------------------------------------------------------
    # 4a. Planet path — source must be planet_highres
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_planet_source_tag(self, mock_planet, mock_ee_cached):
        """When planet data is available it must be the primary source."""
        mock_planet.return_value = {
            "source": "planet_highres",
            "land_cover": "Cropland",
            "land_cover_id": 4,
            "is_cropland": True,
        }
        mock_ee_cached.return_value = None

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "planet_highres")

    # -----------------------------------------------------------------------
    # 4b. Earth Engine path — source must be earth_engine_live
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_ee_source_tag(self, mock_planet, mock_ee_cached):
        """When only EE data is available the source must be earth_engine_live."""
        mock_planet.return_value = None
        mock_ee_cached.return_value = {
            "source": "earth_engine_live",
            "land_cover": "Trees",
            "land_cover_id": 1,
            "is_cropland": False,
            "series": [],
            "ndwi_series": [],
        }

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "earth_engine_live")

    # -----------------------------------------------------------------------
    # 4c. EE returns data but NO land_cover label — source must NOT be EE
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_ee_no_label_falls_back_to_none(self, mock_planet, mock_ee_cached):
        """EE returning series but no land-cover label must yield source='none', not EE."""
        mock_planet.return_value = None
        mock_ee_cached.return_value = {
            "series": [],
            "ndwi_series": [],
            # No 'land_cover' key -> ee_res.get("land_cover") is falsy
        }

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "none")
        self.assertNotEqual(result["source"], "earth_engine_live")

    # -----------------------------------------------------------------------
    # 4d. Spectral water heuristic — source must be spectral_heuristic, NOT EE
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_spectral_heuristic_not_labelled_as_ee(self, mock_planet, mock_ee_cached):
        """
        When NDWI/NDVI series indicate water but EE returned no land-cover label,
        the result source must be 'spectral_heuristic' not 'earth_engine_live'.
        """
        mock_planet.return_value = None
        # EE returned series but no land_cover — triggers spectral fallback
        mock_ee_cached.return_value = {
            "series": [{"date": "2024-01-01", "ndvi": -0.2}],
            "ndwi_series": [{"date": "2024-01-01", "ndwi": 0.5}],
        }

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "spectral_heuristic")
        self.assertNotEqual(result["source"], "earth_engine_live")
        self.assertIn("spectral", result.get("label_origin", "").lower())

    # -----------------------------------------------------------------------
    # 4e. Both Planet and EE unavailable — source must be none
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_no_data_source_none(self, mock_planet, mock_ee_cached):
        """With no planet data and EE unavailable, source must be 'none'."""
        mock_planet.return_value = None
        mock_ee_cached.return_value = None

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["land_cover"], "unknown")

    # -----------------------------------------------------------------------
    # 4f. EE error during analysis — must NOT appear as a successful EE result
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_ee_error_not_presented_as_success(self, mock_planet, mock_ee_cached):
        """
        If ee_analysis_cached() returns None (e.g. due to an internal exception),
        the result must not carry source='earth_engine_live'.
        """
        mock_planet.return_value = None
        mock_ee_cached.return_value = None  # simulates EE failure / timeout

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertNotEqual(result.get("source"), "earth_engine_live")
        # note field should indicate unavailability
        self.assertIn("earth engine", result.get("note", "").lower())

    # -----------------------------------------------------------------------
    # 4g. geohash always present in result
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_geohash_always_present(self, mock_planet, mock_ee_cached):
        """analyze_land() must always include a geohash field."""
        mock_planet.return_value = None
        mock_ee_cached.return_value = None

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertIn("geohash", result)
        self.assertIsInstance(result["geohash"], str)
        self.assertGreater(len(result["geohash"]), 0)

    # -----------------------------------------------------------------------
    # 4h. Planet + EE cross-check — primary source remains planet_highres
    # -----------------------------------------------------------------------
    @patch("app.geo.ee_analysis_cached")
    @patch("app.geo.planet_lookup")
    def test_planet_with_ee_crosscheck_primary_source(self, mock_planet, mock_ee_cached):
        """When both planet and EE are available, primary source must be planet_highres."""
        mock_planet.return_value = {
            "source": "planet_highres",
            "land_cover": "Crops",
            "land_cover_id": 4,
            "is_cropland": True,
        }
        mock_ee_cached.return_value = {
            "source": "earth_engine_live",
            "land_cover": "Crops",
            "land_cover_id": 4,
            "is_cropland": True,
            "series": [],
            "ndwi_series": [],
        }

        from app import geo
        result = geo.analyze_land(20.0, 78.0)
        self.assertEqual(result["source"], "planet_highres")
        # cross_check should be present as a secondary reference
        self.assertIn("cross_check", result)


# ===========================================================================
# 5.  warm_up() safety — must not raise even under bad config
# ===========================================================================
class TestWarmUp(unittest.TestCase):
    def setUp(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    def tearDown(self):
        _clean_env(*_EE_ENV_VARS)
        from app import geo
        _reset_geo(geo)

    @patch("app.geo._load")
    def test_warm_up_does_not_raise_on_missing_config(self, mock_load):
        """warm_up() must never propagate exceptions due to missing EE config."""
        # No EE env vars; _ee() will raise RuntimeError internally
        from app import geo
        try:
            geo.warm_up()
        except Exception as exc:
            self.fail(f"warm_up() raised unexpectedly: {exc}")

    @patch("app.geo._load")
    @patch("app.geo.ee")
    def test_warm_up_calls_load(self, mock_ee, mock_load):
        """warm_up() must always call _load() to prepare the Planet lookup."""
        os.environ["EE_PROJECT_ID"] = "proj"
        mock_ee.Initialize.return_value = None

        from app import geo
        _reset_geo(geo)
        geo.warm_up()

        mock_load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
