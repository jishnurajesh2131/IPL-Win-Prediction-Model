# PHASE 7 — FRONTEND / DASHBOARD IMPLEMENTATION REPORT

**Project**: Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System  
**Phase**: Phase 7 — Local Frontend / Dashboard  
**Status**: COMPLETE & VERIFIED  
**Date**: September 2026  

---

## 1. Executive Summary & Deliverables Overview

Phase 7 delivers a production-grade, responsive, dark-mode sports analytics web dashboard for the **Cloud-Based IPL Match Winning Probability Prediction & Live Match Analytics System**.

The frontend runs locally, connecting exclusively via REST HTTP to the Phase 6 FastAPI backend (`http://127.0.0.1:8000/api/v1`). It provides real-time ball-by-ball match simulation, live win probability calculation, historical probability trajectory graphing, active batter/bowler telemetry, and interactive simulation controls across 1,243 historical IPL matches.

### Key Milestones Delivered:
1. **Production React 18 + TypeScript Application**: Fully typed architecture without `any` types, built with Vite 6.
2. **Dynamic Data Visualization**: Smooth, interactive win probability trajectories using Recharts.
3. **Deep Stadium Floodlight Aesthetic**: Hand-crafted CSS design system featuring glassmorphic panels, neon team accents (Cyan `#38bdf8` and Crimson `#f43f5e`), and clean typography using Google Outfit and JetBrains Mono.
4. **Resilient State Architecture**: Custom React hooks (`useSimulation`, `useMatches`) managing delivery stepping, fast-forwarding, error recovery, and match switching.
5. **Zero ML / Model Alteration**: Zero changes made to frozen Phase 3 pre-match models, Phase 5 live XGBoost models, or preprocessed datasets.
6. **Zero AWS Dependency**: All functionality operates in a local development environment.

---

## 2. Technology Stack Selection Rationale

| Technology | Selected Tool | Rationale |
| :--- | :--- | :--- |
| **Framework** | **React 18.3.1** | Component-driven reactivity, widespread sports dashboard ecosystem, hooks-based state management. |
| **Language** | **TypeScript 5.7.3** | Strict compile-time validation guaranteeing alignment with FastAPI Pydantic schemas. |
| **Bundler** | **Vite 6.0.7** | Sub-second cold starts, instant Hot Module Replacement (HMR), and lightweight production bundles (574 kB JS, 20 kB CSS). |
| **Charting Engine** | **Recharts 2.15.0** | Declarative SVG charting supporting smooth path animations, responsive resizing, reference lines, and custom HTML tooltips. |
| **Iconography** | **Lucide React 0.475.0** | Tree-shakeable SVG icons providing clean visual cues for bowling, batting, venues, and status markers. |
| **Styling** | **Vanilla CSS (CSS3)** | High performance, zero runtime overhead, complete flexibility over custom CSS properties and micro-interactions without Tailwind abstraction. |

---

## 3. UI Design System & Aesthetic Architecture

The dashboard implements an immersive **Stadium Floodlight Dark Mode**:
* **Background Atmosphere**: Deep navy-blacks (`#080c14`, `#0d1527`, `#0f172a`) evoking night matches under floodlights.
* **Team Accents**: 
  - **Team 1 / Home**: Electric Cyan (`#38bdf8`, glow: `rgba(56, 189, 248, 0.3)`)
  - **Team 2 / Away**: Neon Crimson (`#f43f5e`, glow: `rgba(244, 63, 94, 0.3)`)
* **Glassmorphism**: Translucent card backgrounds (`rgba(15, 23, 42, 0.75)`) with subtle borders (`rgba(255, 255, 255, 0.08)`) and backdrop blur (`16px`).
* **Typography**:
  - **Headings & UI**: `Outfit` (sans-serif, geometric, contemporary).
  - **Telemetry & Numbers**: `JetBrains Mono` (tabular numbers preventing jitter during live ball-by-ball updates).

---

## 4. Frontend Directory Structure & Module Breakdown

```text
frontend/
├── index.html                   # HTML document root with web fonts
├── package.json                 # Dependency tree and build scripts
├── tsconfig.json                # TypeScript compiler configuration
├── tsconfig.node.json           # Node configuration for Vite
├── vite.config.ts               # Vite bundler configuration and API proxy
└── src/
    ├── main.tsx                 # Application DOM entry point
    ├── App.tsx                  # Master application orchestrator and layout grid
    ├── index.css                # CSS variables, utilities, glassmorphism, responsive grid
    ├── vite-env.d.ts            # Client environment type definitions
    ├── types/
    │   ├── match.ts             # MatchListItem, MatchDetail, MatchFilter
    │   ├── prediction.ts        # PredictionResponse, ProbabilityPoint, ProbabilityHistory
    │   └── simulation.ts       # DeliverySnapshot, MatchStateSnapshot, Simulation responses
    ├── api/
    │   ├── client.ts            # Typed HTTP client with fetch wrapper
    │   ├── matches.ts           # Match directory endpoints
    │   └── simulations.ts       # Simulation session and stepping endpoints
    ├── hooks/
    │   ├── useMatches.ts        # Match selection and filtering state hook
    │   └── useSimulation.ts     # Step-by-step match replay state machine hook
    └── components/
        ├── Header.tsx           # Stadium top navigation and API status
        ├── MatchHeader.tsx      # Match venue, toss, date, and team badge banner
        ├── WinProbability.tsx   # Hero probability percentage displays and duel bar
        ├── ProbabilityChart.tsx # Recharts live trajectory line chart with event tooltips
        ├── ScoreCard.tsx        # Inning score, wickets, overs, CRR, target, and RRR
        ├── DeliveryCard.tsx     # Animated last delivery event card with runs & extras
        ├── CurrentPlayers.tsx   # Striker, non-striker, and bowler tracking
        ├── SimulationControls.tsx # Next Ball, Run to End, and Change Match controls
        ├── MatchSelector.tsx    # Modal dialog for browsing and searching 1,243 matches
        └── LoadingState.tsx     # Skeleton loading state and error alert banners
```

---

## 5. TypeScript Interfaces & Data Contracts

All TypeScript interfaces mirror the Phase 6 FastAPI Pydantic models:

```typescript
// Prediction contract
export interface PredictionResponse {
  team_1: string;
  team_2: string;
  team_1_probability: number;
  team_2_probability: number;
  prediction_type: string;
}

// Match state telemetry contract
export interface MatchStateSnapshot {
  innings: number;
  score: number;
  wickets: number;
  legal_balls: number;
  overs: string;
  overs_completed: number;
  current_run_rate: number;
  target_runs: number | null;
  runs_needed: number | null;
  required_run_rate: number | null;
  rrr_crr_differential: number | null;
  striker: string;
  non_striker: string;
  bowler: string;
}
```

---

## 6. API Client & Service Layer Integration

The API service layer (`frontend/src/api/`) encapsulates all network I/O with standard error handling:
* **`client.ts`**: Provides a generic `request<T>()` wrapper with JSON headers, status verification, and detailed error messages.
* **`matches.ts`**: Provides `fetchMatches()` with search filtering and `fetchMatchDetail()`.
* **`simulations.ts`**: Implements `checkBackendHealth()`, `createSimulation()`, `advanceSimulation()`, `runSimulation()`, and `fetchProbabilityHistory()`.

---

## 7. Simulation State Management & Custom Hooks

The application state is encapsulated within custom React hooks:

### `useSimulation(matchId: string | null)`
Manages the simulation lifecycle:
* **Creation**: Automatically initializes a session on `matchId` change via `POST /simulations`.
* **Single-Step Advancement**: Dispatches `POST /simulations/{id}/next`, appends the new probability point to the trajectory history, and plays ball animation.
* **Run to Completion**: Dispatches `POST /simulations/{id}/run?include_history=true`, updating the full trajectory and marking the match as `completed`.
* **History Synchronization**: Ensures the trajectory chart reflects all ball-by-ball updates sequentially without re-fetching stale points.

### `useMatches()`
Manages the match catalog:
* Fetches the 1,243 historical match catalog with client-side text filtering by team or season.
* Provides quick selection functions to swap the active match.

---

## 8. Component Architecture: Header & System Health Indicator

* Displays the branding: **IPL LIVE WIN PREDICTOR**.
* Fetches `/api/v1/health` on mount and every 30 seconds.
* Green pulsing badge: `API Online (1,243 matches)` when the FastAPI backend is responsive.
* Red warning badge: `API Offline` with automated error alerting if the backend is stopped.
* Header includes a `Change Match` trigger button for convenient match switching.

---

## 9. Component Architecture: Match Selector Modal & Search Experience

* Implemented as a modal dialog with backdrop blur.
* Includes real-time search input for filtering by team name, season, or match ID.
* Displays match metadata cards (Teams, Date, Venue, Winner, Result).
* Highlights the currently active match.
* Clicking any match closes the modal and immediately initializes a new simulation session.

---

## 10. Component Architecture: Match Metadata Banner

* Displays the match context banner:
  - Season and Match Date.
  - Stadium / Venue name.
  - Toss Winner and Toss Decision ("elected to bat" / "elected to field").
  - Match Phase badge (`READY (T0)`, `IN PROGRESS`, or `COMPLETED`).
  - Winner announcement banner when match reaches completion.

---

## 11. Component Architecture: Win Probability Hero Meter & Bar

* High-contrast hero percentage displays:
  - **Team 1 Probability**: Large font in Cyan (`#38bdf8`).
  - **Team 2 Probability**: Large font in Crimson (`#f43f5e`).
* **Dynamic Probability Bar**: Visual indicator showing real-time balance of winning probability.
* **Prediction Engine Tag**: Displays the active inference mechanism (`pre_match` at T0, `in_play_xgb` during play, `terminal_team_a_win`/`terminal_team_b_win` at completion).

---

## 12. Component Architecture: Recharts Dynamic Probability Trajectory Graph

* Configured with `ResponsiveContainer` and smooth linear interpolation.
* **X-Axis**: Event sequence index (from ball 0 to ball 250+).
* **Y-Axis**: Probability scale (0% to 100%), formatted with percentage labels.
* **Trajectory Curves**:
  - Cyan line tracking Team 1 win probability.
  - Crimson line tracking Team 2 win probability.
* **Reference Lines**: Dashed vertical markers indicating Innings 1 and Innings 2 transitions.
* **Interactive Tooltip**: Displays ball number, score, wickets down, and exact win probabilities on hover.

---

## 13. Component Architecture: Live Score Card & CRR / RRR Telemetry

* **Score Header**: Large tabular typography showing current score, wickets, and overs (e.g., `187/6 (17.4 ov)`).
* **Current Run Rate (CRR)**: Dynamically calculated per over.
* **Chase Telemetry (Innings 2)**:
  - Target Score.
  - Runs Required.
  - Balls Remaining.
  - Required Run Rate (RRR).
  - CRR vs RRR differential badge.

---

## 14. Component Architecture: Delivery Event Card & Wicket Notifications

* Displays details for the latest ball:
  - Over and ball indicator badge (e.g., `Over 15.3`).
  - Runs scored off the bat.
  - Extras breakdown (wides, no-balls, leg-byes).
  - Wicket alert banner with dismissal type and dismissed batter name.

---

## 15. Component Architecture: Active Players On-Field Tracker

* **Striker**: Displays current striker name with active strike icon and runs/balls.
* **Non-Striker**: Displays non-striker partner name.
* **Bowler**: Displays active bowler name.

---

## 16. Component Architecture: Simulation Controls

* **`[ Next Ball ]`**: Advances simulation by one delivery with loading state. Disabled when match is complete or during stepping.
* **`[ Run to End ]`**: Fast-forwards the remaining match to conclusion in a single call.
* **`[ Change Match ]`**: Triggers the match selection modal.

---

## 17. Component Architecture: Loading Skeletons, Errors, and Empty States

* **Loading Skeletons**: Displayed during initial session creation.
* **Error Banner**: Displays clear error messages if an API call fails, with a "Retry" button.
* **Terminal Match Overlay**: Prevents invalid step actions when a match concludes.

---

## 18. End-to-End Simulation Flow & State Transitions

```mermaid
stateDiagram-v2
    [*] --> Idle: Mount App
    Idle --> LoadingSession: Select Match (e.g. 1426274)
    LoadingSession --> Ready_T0: POST /simulations (session_id created)
    Ready_T0 --> Stepping: Click [ Next Ball ]
    Stepping --> InProgress: POST /simulations/{id}/next
    InProgress --> Stepping: Click [ Next Ball ]
    InProgress --> RunningToEnd: Click [ Run to End ]
    RunningToEnd --> Completed: POST /simulations/{id}/run
    InProgress --> Completed: Last Ball Bowled / Wicket 10 / Target Reached
    Completed --> LoadingSession: Click [ Change Match ]
```

---

## 19. Edge Case Handling

1. **Innings Transition**: Correctly switches batting and bowling teams at event 125, resetting balls and setting target runs.
2. **Rain / Shortened Overs**: State machine respects total legal balls from Cricsheet without assuming fixed 120 balls.
3. **Wickets & Super Overs**: Accurately reflects terminal conditions (`terminal_team_a_win`, `terminal_team_b_win`, `terminal_tie`).
4. **Extras**: Handles wides and no-balls without incrementing legal ball counter.

---

## 20. Verification Methodology & Automated Test Suite Results

The full project regression test suite was executed:
```powershell
python -m unittest discover tests/ -v
```
**Results**:
* Total Tests Run: **57**
* Failures: **0**
* Errors: **0**
* Skipped: **0**
* Execution Time: **56.4 seconds**
* Status: **100% PASSING**

---

## 21. End-to-End REST Simulation Execution Proof

Verified via automated live client script against the running backend server (`http://127.0.0.1:8000`):

```text
Matches sample count: 3
Simulation created: session_id=b05a8431-7859-4b8d-9b4e-b515a19c02b7, status=ready, total_events=254
T0 prob: Kolkata Knight Riders=0.4956 | Royal Challengers Bengaluru=0.5044 (pre_match)
Next ball: event=1, ball=0.2, striker=PD Salt, score=0/0, Kolkata Knight Riders=0.4541
  -> event=2, ball=0.3, score=6/0, Kolkata Knight Riders=0.5756
  -> event=3, ball=0.4, score=10/0, Kolkata Knight Riders=0.5756
  -> event=4, ball=0.5, score=11/0, Kolkata Knight Riders=0.5756
Probabilities trajectory points: 5
Run to end: status=completed, winner=Kolkata Knight Riders, total_events=254
Final prob: Kolkata Knight Riders=1.0000 | Royal Challengers Bengaluru=0.0000 (terminal_team_a_win)
Final trajectory points count: 254

ALL FRONTEND REST CALLS SUCCEEDED PERFECTLY!
```

---

## 22. Local Deployment Guide & Developer Experience

### Prerequisites
* Python 3.10+ with project virtual environment installed
* Node.js 18+ and npm 9+

### Starting the Stack
1. **Backend**:
   ```powershell
   uvicorn src.backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Frontend**:
   ```powershell
   cd frontend
   npm run dev
   ```
3. Open `http://localhost:5173` in any modern web browser.

---

## 23. Phase 7 Sign-Off & Freeze Statement

Phase 7 (Local Frontend / Dashboard) is complete, fully functional, and verified.
* All UI components are operational.
* Live REST integration with Phase 6 FastAPI backend is validated.
* Recharts dynamic probability trajectories render correctly.
* Zero changes were made to ML models or datasets.
* Zero AWS cloud resources were created or required.

```text
PHASE 7 STATUS: LOCAL FRONTEND/DASHBOARD COMPLETE
```
