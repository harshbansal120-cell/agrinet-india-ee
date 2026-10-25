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

print(f"Content-Type header: {response.headers.get('content-type')}")
print(f"Requests inferred encoding: {response.encoding}")

content = response.text
assert "index (0–1)" in content, "Missing 'index (0–1)'"
assert "°C" in content, "Missing '°C'"

print("Verification passed! Response contains 'index (0–1)' and '°C' properly encoded.")
