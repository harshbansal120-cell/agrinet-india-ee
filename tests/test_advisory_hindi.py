from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

response = client.post(
    "/api/v1/analyze",
    json={
        "lat": 27.1767,
        "lon": 78.0081,
        "lang": "hi",
        "crop": "Wheat",
        "is_demo": True,
        "scenario_key": "agra_wheat"
    }
)

if response.status_code == 200:
    field_state = response.json()["field_state"]
    evidence = response.json()["evidence"]
    print("Analyze SUCCESS")
    
    # Try advisory with Hindi
    adv_resp = client.post(
        "/api/v1/advisory",
        json={
            "field_state": field_state,
            "evidence": evidence,
            "lang": "hi"
        }
    )
    if adv_resp.status_code == 200:
        print("Advisory SUCCESS (Hindi)")
        print(json.dumps(adv_resp.json(), indent=2, ensure_ascii=False)[:300])
    else:
        print("Advisory Error", adv_resp.text)
else:
    print("Analyze Error", response.text)
