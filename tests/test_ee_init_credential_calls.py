"""
Tests that verify Earth Engine credential-construction and initialization call counts.

Contract:
- ServiceAccountCredentials is called *exactly once* per _ee() invocation when
  EE_SERVICE_ACCOUNT_JSON is supplied and valid.
- If credential construction raises, ee.Initialize must receive *zero* calls.
- If ServiceAccountCredentials is absent or non-callable, no project-only
  initialization fallback occurs – _ee() returns False.
- Project-only initialization (ee.Initialize(project=...)) is used only when
  EE_SERVICE_ACCOUNT_JSON is absent.
- Cache: _ee_ok is True means _ee() returns True immediately without re-initializing.
- Retry cooldown: after a failure, _ee_ok is reset to None once _EE_RETRY_S has elapsed.
"""
import time
import logging
import pytest
from app import geo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_ee_state(monkeypatch):
    """Reset module-level EE state before every test."""
    monkeypatch.setattr(geo, "_ee_ok", None)
    monkeypatch.setattr(geo, "_ee_failed_at", 0.0)
    yield


VALID_SA_JSON = (
    '{"type": "service_account", "project_id": "p", '
    '"private_key": "key", "client_email": "email@example.com"}'
)


def _make_mock_ee(*, sc_raises=False, init_raises=False, has_sc=True, sc_callable=True):
    """Factory for a fake *ee* module with call-count tracking."""
    sc_call_count = {"n": 0}
    init_call_count = {"n": 0}

    class _Creds:
        def __init__(self, email=None, key_data=None):
            sc_call_count["n"] += 1
            if sc_raises:
                raise RuntimeError("credential construction error")
            self.email = email
            self.key_data = key_data

    class _MockEE:
        if has_sc:
            ServiceAccountCredentials = _Creds if sc_callable else object()

        def Initialize(self, *args, **kwargs):
            init_call_count["n"] += 1
            if init_raises:
                raise RuntimeError("init failure")

    return _MockEE(), sc_call_count, init_call_count


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_credentials_constructed_exactly_once(monkeypatch):
    """ServiceAccountCredentials must be called exactly once when SA JSON is valid."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    mock_ee, sc_count, init_count = _make_mock_ee()
    monkeypatch.setattr(geo, "ee", mock_ee)

    result = geo._ee()

    assert result is True
    assert sc_count["n"] == 1, f"Expected 1 credential construction call, got {sc_count['n']}"
    assert init_count["n"] == 1, "Initialize must be called exactly once"


def test_initialize_receives_credentials_and_project(monkeypatch):
    """ee.Initialize must be called with the credentials object and project kwarg."""
    monkeypatch.setenv("EE_PROJECT_ID", "my-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    received = {}

    class _Creds:
        def __init__(self, email=None, key_data=None):
            pass

    class _MockEE:
        ServiceAccountCredentials = _Creds

        def Initialize(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

    monkeypatch.setattr(geo, "ee", _MockEE())

    assert geo._ee() is True
    assert received["kwargs"].get("project") == "my-project"
    assert isinstance(received["args"][0], _Creds)


# ---------------------------------------------------------------------------
# Credential construction failure → Initialize must not be called
# ---------------------------------------------------------------------------

def test_credential_construction_failure_prevents_initialize(monkeypatch):
    """If ServiceAccountCredentials raises, ee.Initialize must receive zero calls."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    mock_ee, sc_count, init_count = _make_mock_ee(sc_raises=True)
    monkeypatch.setattr(geo, "ee", mock_ee)

    result = geo._ee()

    assert result is False
    assert sc_count["n"] == 1, "Construction must be attempted once before failing"
    assert init_count["n"] == 0, "Initialize must never be called after credential failure"


def test_malformed_json_prevents_initialize(monkeypatch):
    """Malformed JSON must prevent both credential construction and Initialize."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", "{invalid_json}")

    mock_ee, sc_count, init_count = _make_mock_ee()
    monkeypatch.setattr(geo, "ee", mock_ee)

    result = geo._ee()

    assert result is False
    assert sc_count["n"] == 0, "Credential factory must not be called for malformed JSON"
    assert init_count["n"] == 0, "Initialize must not be called for malformed JSON"


# ---------------------------------------------------------------------------
# Missing / non-callable credential factory → no project-only fallback
# ---------------------------------------------------------------------------

def test_missing_credential_factory_prevents_project_only_init(monkeypatch):
    """When SA JSON is present but ServiceAccountCredentials is absent, return False."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    mock_ee, _, init_count = _make_mock_ee(has_sc=False)
    monkeypatch.setattr(geo, "ee", mock_ee)

    result = geo._ee()

    assert result is False
    assert init_count["n"] == 0, "Project-only Initialize must not be called when SA JSON is configured"


def test_non_callable_credential_factory_prevents_project_only_init(monkeypatch):
    """When ServiceAccountCredentials is not callable, return False; no project-only init."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    mock_ee, _, init_count = _make_mock_ee(sc_callable=False)
    monkeypatch.setattr(geo, "ee", mock_ee)

    result = geo._ee()

    assert result is False
    assert init_count["n"] == 0, "Project-only Initialize must not be called for non-callable factory"


# ---------------------------------------------------------------------------
# Project-only initialization (no SA JSON)
# ---------------------------------------------------------------------------

def test_project_only_init_when_no_sa_json(monkeypatch):
    """When EE_SERVICE_ACCOUNT_JSON is absent, use ee.Initialize(project=...) only."""
    monkeypatch.setenv("EE_PROJECT_ID", "only-project")
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_JSON", raising=False)

    received = {}

    class _MockEE:
        def Initialize(self, *args, **kwargs):
            received["args"] = args
            received["kwargs"] = kwargs

    monkeypatch.setattr(geo, "ee", _MockEE())

    result = geo._ee()

    assert result is True
    assert received["kwargs"].get("project") == "only-project"
    assert len(received["args"]) == 0, "No credentials positional arg expected for project-only init"


# ---------------------------------------------------------------------------
# Cache and retry-cooldown
# ---------------------------------------------------------------------------

def test_cache_returns_true_without_reinitializing(monkeypatch):
    """Once _ee_ok is True, _ee() must return True without calling Initialize again."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setattr(geo, "_ee_ok", True)

    init_count = {"n": 0}

    class _MockEE:
        def Initialize(self, *args, **kwargs):
            init_count["n"] += 1

    monkeypatch.setattr(geo, "ee", _MockEE())

    result = geo._ee()

    assert result is True
    assert init_count["n"] == 0, "Initialize must not be called when cache says True"


def test_retry_after_cooldown(monkeypatch):
    """After _EE_RETRY_S has elapsed since a failure, _ee() must retry initialization."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.delenv("EE_SERVICE_ACCOUNT_JSON", raising=False)

    # Simulate a past failure that is older than the retry window.
    monkeypatch.setattr(geo, "_ee_ok", False)
    monkeypatch.setattr(geo, "_ee_failed_at", time.monotonic() - geo._EE_RETRY_S - 1)

    init_count = {"n": 0}

    class _MockEE:
        def Initialize(self, *args, **kwargs):
            init_count["n"] += 1

    monkeypatch.setattr(geo, "ee", _MockEE())

    result = geo._ee()

    assert result is True
    assert init_count["n"] == 1, "Initialize must be called once after cooldown expires"


# ---------------------------------------------------------------------------
# Regression: None result from ServiceAccountCredentials
# ---------------------------------------------------------------------------

def test_none_credentials_prevent_initialize(monkeypatch):
    """If ServiceAccountCredentials returns None, _ee() must return False and
    never call ee.Initialize() — not even with project-only initialization."""
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", VALID_SA_JSON)

    init_count = {"n": 0}

    class _MockEE:
        @staticmethod
        def ServiceAccountCredentials(email=None, key_data=None):
            # Deliberately returns None to simulate a broken factory.
            return None

        def Initialize(self, *args, **kwargs):
            init_count["n"] += 1

    monkeypatch.setattr(geo, "ee", _MockEE())

    result = geo._ee()

    assert result is False, "A None credential result must be treated as malformed_credentials"
    assert init_count["n"] == 0, (
        "ee.Initialize must not be called when credential construction returns None"
    )
