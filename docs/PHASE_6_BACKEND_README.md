# Phase 6: Local Backend & REST API Guide

**Project:** Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Component:** Local Backend / REST API Layer  
**Framework:** FastAPI 0.141.1 + Uvicorn 0.52.4 + Pydantic v2  
**Status:** COMPLETE & TESTED (Local-only, Zero AWS)

---

## 1. Prerequisites & Environment

The backend runs locally on Python 3.14+ (or compatible virtual environment).

Ensure dependencies are installed:
```powershell
pip install -r requirements.txt
```

Pinned dependencies include:
- `fastapi>=0.115.0`
- `uvicorn>=0.30.0`
- `pydantic>=2.8.0`
- `httpx>=0.27.0`
- Core scientific stack: `numpy`, `pandas`, `scipy`, `scikit-learn==1.9.0`, `xgboost==3.4.1`, `joblib==1.6.0`, `matplotlib>=3.10.0`.

---

## 2. Starting the Backend Server

Start the local development server using Uvicorn:

```powershell
uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Output on startup:
```text
============================================================
Starting IPL Win Probability & Live Match Analytics API (v1.0.0)
============================================================
Match catalog loaded: 1,243 matches available.
Loading deliveries from data/processed/deliveries_clean.csv...
Frozen Two-Model live simulator engine ready.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The server binds to:
- **Base URL:** `http://127.0.0.1:8000`
- **API v1 Prefix:** `http://127.0.0.1:8000/api/v1`
- **Interactive Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc UI:** `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON Schema:** `http://127.0.0.1:8000/openapi.json`

---

## 3. API Endpoint Inventory

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root service metadata and navigation endpoints |
| `GET` | `/api/v1/health` | Service health, dataset availability, and frozen model check |
| `GET` | `/api/v1/matches` | Catalog of historical matches with pagination, season, and team filters |
| `GET` | `/api/v1/matches/{match_id}` | Complete metadata for a specific match ID |
| `POST` | `/api/v1/simulations` | Create a new replay session; establishes the $T_0$ pre-match anchor |
| `GET` | `/api/v1/simulations/{session_id}` | Retrieve instantaneous match state and prediction without advancing |
| `POST` | `/api/v1/simulations/{session_id}/next` | Advance simulation by exactly one delivery event |
| `GET` | `/api/v1/simulations/{session_id}/probabilities` | Retrieve full probability trajectory up to current position |
| `POST` | `/api/v1/simulations/{session_id}/run` | Replay all remaining events until match conclusion |

---

## 4. Usage Examples (cURL & Python)

### Health Check
```powershell
curl -X GET "http://127.0.0.1:8000/api/v1/health"
```
Response:
```json
{
  "status": "ok",
  "service": "ipl-live-analytics-backend",
  "version": "1.0.0",
  "ml_status": "loaded",
  "model_artifacts": {
    "live_model_inn1": true,
    "live_model_inn2": true,
    "live_preprocessor_inn1": true,
    "live_preprocessor_inn2": true,
    "matches_csv": true,
    "deliveries_csv": true,
    "walk_forward_csv": true
  },
  "matches_available": 1243
}
```

### Discover Historical Matches
```powershell
# Filter by season 2024 with limit 3
curl -X GET "http://127.0.0.1:8000/api/v1/matches?season=2024&limit=3"
```

### Initialize a Simulation Session
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/simulations" `
  -H "Content-Type: application/json" `
  -d '{"match_id": "1426274"}'
```
Response (T0 Anchor):
```json
{
  "session_id": "497a01a9-4dfa-458b-ba24-6ba9795c3ead",
  "match_id": "1426274",
  "status": "ready",
  "current_event_index": 0,
  "total_events": 254,
  "current_prediction": {
    "team_1": "Kolkata Knight Riders",
    "team_2": "Royal Challengers Bengaluru",
    "team_1_probability": 0.4956,
    "team_2_probability": 0.5044,
    "prediction_type": "pre_match"
  }
}
```

### Advance One Delivery Event
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/simulations/497a01a9-4dfa-458b-ba24-6ba9795c3ead/next"
```
Response:
```json
{
  "session_id": "497a01a9-4dfa-458b-ba24-6ba9795c3ead",
  "match_id": "1426274",
  "event_index": 1,
  "total_events": 254,
  "status": "active",
  "delivery": {
    "innings": 1,
    "ball": 0.1,
    "is_legal": true,
    "batting_team": "Kolkata Knight Riders",
    "bowling_team": "Royal Challengers Bengaluru",
    "striker": "PD Salt",
    "non_striker": "SP Narine",
    "bowler": "Mohammed Siraj",
    "runs_off_bat": 0,
    "extras": 0,
    "wides": 0,
    "noballs": 0,
    "wicket_type": null,
    "player_dismissed": null
  },
  "state": {
    "innings": 1,
    "score": 0,
    "wickets": 0,
    "legal_balls": 1,
    "overs": "0.1",
    "overs_completed": 0.17,
    "current_run_rate": 0.0,
    "target_runs": null,
    "runs_needed": null,
    "required_run_rate": null,
    "rrr_crr_differential": null,
    "striker": "PD Salt",
    "non_striker": "SP Narine",
    "bowler": "Mohammed Siraj"
  },
  "prediction": {
    "team_1": "Kolkata Knight Riders",
    "team_2": "Royal Challengers Bengaluru",
    "team_1_probability": 0.4541,
    "team_2_probability": 0.5459,
    "prediction_type": "in_play_xgb"
  }
}
```

### Run to Completion
```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/simulations/497a01a9-4dfa-458b-ba24-6ba9795c3ead/run?include_history=true"
```

---

## 5. Running Tests

Run backend-specific test suites:
```powershell
# 1. API endpoint suite
python -m unittest tests/test_backend_api.py -v

# 2. Session concurrency & isolation suite
python -m unittest tests/test_backend_session_isolation.py -v

# 3. Critical ML & state parity suite (Match 1426274 and Match 419155)
python -m unittest tests/test_backend_ml_parity.py -v
```

Run complete project regression suite:
```powershell
python -m unittest discover tests/ -v
```

---

## 6. Architecture & Local Boundaries

- **Zero AWS:** Strictly local implementation without cloud SDKs or dependencies.
- **Zero Model Retraining:** Frozen Two-Model XGBoost engine (`models/live/`) invoked read-only.
- **Thread-Safe In-Memory Sessions:** Multiple clients can run simulations simultaneously without state collisions.
