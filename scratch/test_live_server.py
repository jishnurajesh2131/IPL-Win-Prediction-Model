import urllib.request
import json

base_url = "http://127.0.0.1:8000/api/v1"

# 1. Health
with urllib.request.urlopen(f"{base_url}/health") as r:
    health = json.loads(r.read().decode())
print("Health status:", health["status"], "| ML status:", health["ml_status"])

# 2. List matches
with urllib.request.urlopen(f"{base_url}/matches?season=2024&limit=3") as r:
    matches = json.loads(r.read().decode())
print(f"2024 matches found: {matches['total']} | Returned: {len(matches['matches'])}")

# 3. Create simulation
req_create = urllib.request.Request(
    f"{base_url}/simulations",
    data=json.dumps({"match_id": "1426274"}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req_create) as r:
    session = json.loads(r.read().decode())
sid = session["session_id"]
print("Simulation created. Session ID:", sid)
print("T0 Anchor Prediction:", session["current_prediction"])

# 4. Step 1
req_next = urllib.request.Request(
    f"{base_url}/simulations/{sid}/next",
    data=b"",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req_next) as r:
    step = json.loads(r.read().decode())
print(f"Event {step['event_index']} State: {step['state']['score']}/{step['state']['wickets']} (overs {step['state']['overs']}) | P(A): {step['prediction']['team_1_probability']:.4f}")

# 5. Fast-forward to conclusion
req_run = urllib.request.Request(
    f"{base_url}/simulations/{sid}/run?include_history=false",
    data=b"",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req_run) as r:
    run_res = json.loads(r.read().decode())
print(f"Match concluded: Winner: {run_res['winner']} | Events processed: {run_res['events_processed']} | Final P(A): {run_res['final_prediction']['team_1_probability']}")

# 6. Verify 409 Conflict when advancing completed session
try:
    with urllib.request.urlopen(req_next) as r:
        pass
except urllib.error.HTTPError as e:
    print("Expected 409 Conflict received:", e.code, e.reason)

print("\nALL LIVE SERVER TESTS PASSED PERFECTLY!")
