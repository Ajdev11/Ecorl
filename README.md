# Algorithmic Competition and Consumer Welfare – MVP

Minimal, research-first backend to simulate AI-driven pricing dynamics and measure competition and welfare impacts.

## Stack (MVP)
- FastAPI (HTTP API)
- Ray (background simulation jobs)
- NumPy/Pandas/PyArrow (data + Parquet)
- MLflow (local file-based tracking at `./mlruns`)

## Quickstart (Windows PowerShell)
1) Create and activate a virtual environment (Python 3.11+ recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies
```powershell
pip install --upgrade pip
pip install -r requiremeAnts.txt
```

3) Run the API (auto-reload)
```powershell
uvicorn backend.app.main:app --reload
```
It will start at `http://127.0.0.1:8000` with interactive docs at `/docs`.

## Key API endpoints
- POST `/scenarios` – create a scenario (market + demand + regulation)
- POST `/runs` – launch a simulation run for a scenario
- GET `/runs/{run_id}` – check run status
- GET `/results/{run_id}` – fetch summary metrics
- GET `/streams/{run_id}` – server‑sent events (live progress)

## Project layout
```
backend/
  app/
    main.py                # FastAPI app and routing
    schemas.py             # Pydantic models
    simulation/
      runner.py            # Simulation driver (Ray task)
      market.py            # Market and demand primitives
      agents.py            # Firm pricing policies (baselines)
      regulator.py         # Regulation hooks (stubs)
    utils/
      run_store.py         # Simple file-backed run registry
      io.py                # IO helpers for Parquet/JSON/logs
scripts/
  sweep_runs.py            # Launch small sweeps via API, write sweep CSV
  aggregate_runs.py        # Aggregate all runs to one metrics CSV
data/
  runs/                    # Per-run outputs (created at runtime)
  scenarios/               # Stored scenarios (JSON)
mlruns/                    # Local MLflow tracking (created at runtime)
```

## Notes
- MLflow uses a local file store: `mlflow.set_tracking_uri("file:./mlruns")` – no server needed.
- Results are stored under `data/runs/{run_id}/` including:
  - `panel.parquet` – time-firm panel of price, cost, quantity, profit
  - `summary.json` – high-level metrics
  - `status.json` – run state: queued/running/succeeded/failed
  - `stdout.log` – progress stream for `/streams/{run_id}`
  - `scenario.json` – exact scenario used for the run
  - `config.json` – run configuration (seed, log interval)

## Next steps
- Add richer agent policies (bandits, Q-learning, actor-critic)
- Switch metadata to Postgres when needed
- Add a Streamlit/Next.js UI to build scenarios and visualize results

## Batch usage (optional)
Install optional deps:
```powershell
python -m pip install requests pandas
```

Run a small sweep (myopic vs bandit, seeds 51–55):
```powershell
python scripts/sweep_runs.py
```

Aggregate coordination metrics across all runs:
```powershell
python scripts/aggregate_runs.py
```
Outputs:
- `data/aggregates/sweep_results.csv` – summaries for the sweep
- `data/aggregates/metrics.csv` – per-run metrics + coordination indicators

