import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Import the module containing the _ee function
from app import geo

class TestEarthEngineInit(unittest.TestCase):
    def setUp(self):
        # Ensure environment is clean before each test
        for var in ["EE_PROJECT_ID", "EE_SERVICE_ACCOUNT_JSON", "EE_PROJECT"]:
            if var in os.environ:
                del os.environ[var]
        # Reset internal flags in geo module
        geo._ee_ok = None
        geo._ee_failed_at = 0

    @patch("app.geo.ee")
    def test_init_with_service_account(self, mock_ee):
        # Mock environment variables for service account
        os.environ["EE_PROJECT_ID"] = "test-project"
        sa_json = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key": "----PRIVATE KEY----",
            "client_email": "service-account@test.iam.gserviceaccount.com"
        }
        os.environ["EE_SERVICE_ACCOUNT_JSON"] = json.dumps(sa_json)

        # Mock ServiceAccountCredentials and Initialize
        mock_credentials = MagicMock()
        mock_ee.ServiceAccountCredentials.return_value = mock_credentials
        mock_ee.Initialize.return_value = None

        result = geo._ee()
        self.assertTrue(result)
        mock_ee.ServiceAccountCredentials.assert_called_once()
        mock_ee.Initialize.assert_called_once_with(mock_credentials, project="test-project")

    @patch("app.geo.ee")
    def test_init_with_project_only(self, mock_ee):
        os.environ["EE_PROJECT_ID"] = "proj-only"
        # No service account JSON set
        mock_ee.Initialize.return_value = None

        result = geo._ee()
        self.assertTrue(result)
        mock_ee.Initialize.assert_called_once_with(project="proj-only")
        mock_ee.ServiceAccountCredentials.assert_not_called()

    def test_init_missing_configuration(self):
        # No environment variables set, should raise RuntimeError
        with self.assertRaises(RuntimeError):
            geo._ee()

if __name__ == "__main__":
    unittest.main()
