from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

response = client.post(
    "/api/v1/analyze",
    json={
        "lat": 27.1767,
        "lon": 78.0081,
        "lang": "en",
        "crop": "Wheat",
        "is_demo": True,
        "scenario_key": "agra_wheat"
    }
)

print(json.dumps(response.json()["provenance"], indent=2))
