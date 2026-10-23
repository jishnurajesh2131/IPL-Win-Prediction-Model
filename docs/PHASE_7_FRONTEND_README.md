# Phase 7 — Local Frontend / Live Match Analytics Dashboard

This document provides complete instructions and architecture documentation for the **Phase 7 Local Frontend Dashboard** for the **Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System**.

---

## 1. Overview & Architecture

The Phase 7 frontend is a high-performance, dark stadium floodlight-themed sports analytics dashboard built using **React 18 + TypeScript + Vite**. It consumes the **Phase 6 FastAPI REST API** running locally on port 8000, presenting live delivery-by-delivery simulation states, win probabilities, and interactive probability trajectories without requiring AWS cloud infrastructure or direct access to ML models.

### Key Principles & Boundaries
* **Strict ML Pipeline Freeze**: The frontend has no direct access to Python ML files or datasets. All predictions are fetched via the FastAPI REST API.
* **Pure REST Stepping**: The dashboard advances matches delivery-by-delivery using `POST /api/v1/simulations/{session_id}/next` or runs to completion via `POST /api/v1/simulations/{session_id}/run`.
* **Zero AWS Dependency**: All components execute entirely locally (`http://localhost:5173` talking to `http://127.0.0.1:8000/api/v1`).

---

## 2. Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Framework** | React + React DOM | `^18.3.1` | Declarative UI component architecture |
| **Language** | TypeScript | `^5.7.3` | End-to-end type safety for API schemas and component props |
| **Bundler & Dev Server** | Vite | `^6.0.7` | Ultra-fast HMR and optimized production bundling |
| **Charts** | Recharts | `^2.15.0` | High-fidelity, responsive win probability trajectory charts |
| **Icons** | Lucide React | `^0.475.0` | Minimalist icons for stadium, ball, wickets, runs, and controls |
| **Styling** | Vanilla CSS | CSS3 Custom Properties | Deep stadium dark theme (`#080c14`, `#0d1527`), glassmorphism, cyan/crimson team accents |

---

## 3. Directory Layout

```text
frontend/
├── index.html                   # HTML5 entrypoint with Google Outfit and JetBrains Mono fonts
├── package.json                 # Project dependencies and npm scripts
├── tsconfig.json                # TypeScript project configuration (strict mode)
├── tsconfig.node.json           # Vite Node configuration
├── vite.config.ts               # Vite configuration with proxy to http://127.0.0.1:8000
└── src/
    ├── main.tsx                 # React DOM mount point
    ├── App.tsx                  # Master application orchestrator and layout grid
    ├── index.css                # CSS design system (tokens, glassmorphism, responsive grid)
    ├── vite-env.d.ts            # Vite client type declarations
    ├── types/
    │   ├── match.ts             # MatchListItem, MatchDetail, MatchFilter types
    │   ├── prediction.ts        # PredictionResponse, ProbabilityPoint, ProbabilityHistory
    │   └── simulation.ts       # DeliverySnapshot, MatchStateSnapshot, Simulation responses
    ├── api/
    │   ├── client.ts            # Typed HTTP request client with base URL resolution
    │   ├── matches.ts           # fetchMatches, fetchMatchDetail
    │   └── simulations.ts       # checkBackendHealth, createSimulation, advanceSimulation, runSimulation
    ├── hooks/
    │   ├── useMatches.ts        # Match selection and filtering hook
    │   └── useSimulation.ts     # Core simulation state machine hook (T0 to terminal)
    └── components/
        ├── Header.tsx           # Stadium navigation header with live API status & match toggle
        ├── MatchHeader.tsx      # Match venue, toss, date, and team badge banner
        ├── WinProbability.tsx   # Hero probability meters and dynamic probability bar
        ├── ProbabilityChart.tsx # Recharts live trajectory line chart with event tooltips
        ├── ScoreCard.tsx        # Inning score, wickets, overs, CRR, target, and RRR
        ├── DeliveryCard.tsx     # Animated last delivery event card with runs & extras
        ├── CurrentPlayers.tsx   # Striker, non-striker, and current bowler tracking
        ├── SimulationControls.tsx # Next Ball, Run to End, and Change Match buttons
        ├── MatchSelector.tsx    # Modal/drawer dialog for browsing and searching 1,243 matches
        └── LoadingState.tsx     # Skeleton loading state and error alert banners
```

---

## 4. How to Run Locally

### Step 1: Start Backend API Service
From the project root directory, activate your Python virtual environment and start the Phase 6 FastAPI backend:
```powershell
uvicorn src.backend.main:app --host 127.0.0.1 --port 8000
```
Verify the API is online by navigating to `http://127.0.0.1:8000/api/v1/health` or `http://127.0.0.1:8000/docs`.

### Step 2: Start Frontend Development Server
In a new terminal window:
```powershell
cd frontend
npm run dev
```
Open `http://localhost:5173` in your browser. The dashboard automatically detects the backend, displays `API Online (1,243 matches)`, and pre-warms Match 1426274 (KKR vs RCB 2024 thriller).

### Step 3: Production Build (Optional)
To verify that the frontend bundles without errors:
```powershell
cd frontend
npm run build
npm run preview
```

---

## 5. UI Components & User Flow

### 1. Stadium Header (`Header.tsx`)
* Displays application branding and current backend operational status.
* Live pulse indicator: Green `API Online (X matches)` when connected, red `API Offline` when disconnected.
* Includes a quick "Select Match" trigger.

### 2. Match Header Banner (`MatchHeader.tsx`)
* Details the match metadata: Season, Date, Venue, Toss Winner, and Toss Decision.
* Badges indicate current simulation state: `READY (T0)`, `IN PROGRESS`, or `COMPLETED`.

### 3. Hero Win Probability (`WinProbability.tsx`)
* Large numeric percentage displays for Team 1 (cyan `#38bdf8`) and Team 2 (crimson `#f43f5e`).
* A dual-colored probability bar visually represents the balance of power.
* Identifies the current prediction engine: `pre_match`, `in_play_xgb`, or `terminal_win`.

### 4. Probability Trajectory Chart (`ProbabilityChart.tsx`)
* Built using `Recharts.ResponsiveContainer`, `LineChart`, and `XAxis`/`YAxis`.
* X-axis tracks sequential delivery events (from Event 0 pre-match up to ball 250+).
* Y-axis plots win probability percentage from 0% to 100%.
* Reference lines demarcate Innings 1 and Innings 2.
* Custom tooltip displays runs, wickets, and exact win probabilities for any hovered ball.

### 5. Live Score & Match Metrics (`ScoreCard.tsx`)
* Prominently displays current innings score, wickets, and legal overs (e.g., `187/6 (17.4 ov)`).
* Computes Current Run Rate (CRR).
* In 2nd innings: Dynamically shows Target, Runs Needed, Balls Remaining, and Required Run Rate (RRR).

### 6. Delivery Event & Player Cards (`DeliveryCard.tsx` & `CurrentPlayers.tsx`)
* Highlights the outcome of the most recent delivery (Runs off bat, Extras, Wides, No-balls, Wicket).
* Displays active on-field players: Striker, Non-Striker, and Current Bowler.

### 7. Simulation Controls (`SimulationControls.tsx`)
* **[ Next Ball ]**: Advances the simulation by exactly one delivery event.
* **[ Run to End ]**: Fast-forwards the match to terminal conclusion in a single batch request.
* **[ Change Match ]**: Opens the modal selector to choose any historical match.

---

## 6. API Integration Contract

The frontend connects to the backend using standard REST endpoints:

1. `GET /api/v1/health`: Checks backend health and count of available matches.
2. `GET /api/v1/matches?limit=50&search=...`: Retrieves historical matches for selection.
3. `POST /api/v1/simulations`: Creates a new simulation session for `match_id`, establishing the initial T0 pre-match anchor.
4. `POST /api/v1/simulations/{session_id}/next`: Steps forward by one delivery and returns updated score and probability.
5. `POST /api/v1/simulations/{session_id}/run?include_history=true`: Runs simulation to conclusion and returns final match state.
6. `GET /api/v1/simulations/{session_id}/probabilities`: Fetches complete probability points for the chart.

---

## 7. Verification Summary

* **Type Safety & Build**: Passed `tsc -b` and `vite build` without errors.
* **Backend API Compatibility**: Verified end-to-end against local live API (`uvicorn src.backend.main:app`).
* **Regression Suite**: Passed 57 / 57 unit tests in `tests/` with zero regressions.
