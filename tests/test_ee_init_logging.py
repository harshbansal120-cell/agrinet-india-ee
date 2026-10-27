import logging
import pytest
from app import geo

# Ensure each test starts with a clean EE state
@pytest.fixture(autouse=True)
def reset_ee_state(monkeypatch):
    monkeypatch.setattr(geo, "_ee_ok", None)
    monkeypatch.setattr(geo, "_ee_failed_at", 0.0)
    yield

def test_missing_configuration_logs(caplog, monkeypatch):
    # No EE environment variables
    monkeypatch.delenv("EE_PROJECT_ID", raising=False)
    monkeypatch.delenv("EE_PROJECT", raising=False)
    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        geo._ee()
    # Verify sanitized category appears, without leaking raw exception text
    assert any("missing_configuration" in rec.message for rec in caplog.records)
    # Ensure no private key or raw exception text leaked
    assert not any("private_key" in rec.message for rec in caplog.records)

def test_malformed_credentials_logs(caplog, monkeypatch):
    # Provide a project ID so we reach the service‑account branch
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    # Malformed JSON payload
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", "{invalid_json}")
    caplog.set_level(logging.ERROR)
    # Expect no exception, _ee should return False and log malformed_credentials
    result = geo._ee()
    assert result is False
    assert any("malformed_credentials" in rec.message for rec in caplog.records)
    assert not any("invalid_json" in rec.message for rec in caplog.records)

# Helper mock EE that raises an auth error with a given HTTP code
class _MockAuthError(Exception):
    def __init__(self, code):
        self.code = code

class _DummyCreds:
    """Returned by the mock ServiceAccountCredentials factory."""

class _MockEE:
    def __init__(self, code):
        self._code = code

    @staticmethod
    def ServiceAccountCredentials(email=None, key_data=None):
        return _DummyCreds()

    def Initialize(self, *args, **kwargs):
        raise _MockAuthError(self._code)

def _setup_mock_ee(monkeypatch, code):
    mock_ee = _MockEE(code)
    monkeypatch.setattr(geo, "ee", mock_ee)
    # Provide valid JSON so parsing succeeds
    valid_json = "{\"type\": \"service_account\", \"project_id\": \"p\", \"private_key\": \"key\", \"client_email\": \"email@example.com\"}"
    monkeypatch.setenv("EE_PROJECT_ID", "dummy-project")
    monkeypatch.setenv("EE_SERVICE_ACCOUNT_JSON", valid_json)
    return mock_ee


def test_unauthorized_error_logging(caplog, monkeypatch):
    _setup_mock_ee(monkeypatch, 401)
    caplog.set_level(logging.ERROR)
    with pytest.raises(_MockAuthError):
        geo._ee()
    assert any("unauthorized" in rec.message for rec in caplog.records)
    # No raw exception text leaked
    assert not any("401" in rec.message for rec in caplog.records if "unauthorized" not in rec.message)

def test_forbidden_error_logging(caplog, monkeypatch):
    _setup_mock_ee(monkeypatch, 403)
    caplog.set_level(logging.ERROR)
    with pytest.raises(_MockAuthError):
        geo._ee()
    assert any("forbidden" in rec.message for rec in caplog.records)
    # Ensure no raw exception text leaked
    assert not any("403" in rec.message for rec in caplog.records if "forbidden" not in rec.message)
