from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import asyncio

try:
	import ray  # type: ignore[import-not-found]
except Exception:
	class _RayNoop:
		@staticmethod
		def is_initialized() -> bool:
			return False
		@staticmethod
		def init(*args, **kwargs) -> None:
			return None
		@staticmethod
		def shutdown() -> None:
			return None
	ray = _RayNoop()  # type: ignore[assignment]
	from fastapi import FastAPI, HTTPException  # type: ignore[import-not-found]
	from fastapi.responses import StreamingResponse  # type: ignore[import-not-found]

from .schemas import (
	Scenario,
	ScenarioCreate,
	RunCreate,
	RunStatus,
	RunSummary,
)
from .utils.io import ensure_dir, write_json, read_json
from .utils.run_store import status_file, get_status, set_status


BASE_DIR = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = BASE_DIR / "data" / "scenarios"
RUNS_DIR = BASE_DIR / "data" / "runs"
ensure_dir(SCENARIOS_DIR)
ensure_dir(RUNS_DIR)


app = FastAPI(title="Algorithmic Competition – MVP", version="0.1.0")


@app.on_event("startup")
def _on_startup() -> None:
	if not ray.is_initialized():
		ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)


@app.on_event("shutdown")
def _on_shutdown() -> None:
	if ray.is_initialized():
		ray.shutdown()


@app.post("/scenarios")
def create_scenario(payload: ScenarioCreate) -> Dict[str, str]:
	scenario_id = str(uuid.uuid4())
	scenario = Scenario(id=scenario_id, **payload.model_dump())
	path = SCENARIOS_DIR / f"{scenario_id}.json"
	write_json(path, scenario.model_dump())
	return {"scenario_id": scenario_id}


def _load_scenario(scenario_id: str) -> Scenario:
	path = SCENARIOS_DIR / f"{scenario_id}.json"
	if not path.exists():
		raise HTTPException(status_code=404, detail="Scenario not found")
	return Scenario(**read_json(path))


@app.post("/runs")
def create_run(payload: RunCreate) -> Dict[str, str]:
	if payload.scenario_id is None and payload.scenario is None:
		raise HTTPException(status_code=400, detail="Provide either scenario_id or scenario")
	if payload.scenario_id is not None and payload.scenario is not None:
		raise HTTPException(status_code=400, detail="Provide only one of scenario_id or scenario")

	if payload.scenario_id:
		scenario = _load_scenario(payload.scenario_id)
	else:
		scenario_id = str(uuid.uuid4())
		scenario = Scenario(id=scenario_id, **payload.scenario.model_dump())
		write_json(SCENARIOS_DIR / f"{scenario_id}.json", scenario.model_dump())

	run_id = str(uuid.uuid4())
	run_dir = RUNS_DIR / run_id
	ensure_dir(run_dir)
	set_status(status_file(run_dir), "queued", "Run created")

	# Fire-and-forget Ray task
	# Lazy import to avoid heavy deps at startup
	from .simulation.runner import ray_simulation_task  # type: ignore
	ray_simulation_task.remote(
		run_id=run_id,
		scenario_dict=scenario.model_dump(),
		run_config_dict=payload.config.model_dump(),
		base_dir_str=str(BASE_DIR),
	)
	return {"run_id": run_id}


@app.get("/runs/{run_id}", response_model=RunStatus)
def get_run_status(run_id: str) -> RunStatus:
	run_dir = RUNS_DIR / run_id
	status = get_status(status_file(run_dir))
	if status is None:
		raise HTTPException(status_code=404, detail="Run not found")
	return RunStatus(run_id=run_id, status=status["status"], message=status.get("message"))


@app.get("/results/{run_id}", response_model=RunSummary)
def get_run_results(run_id: str) -> RunSummary:
	run_dir = RUNS_DIR / run_id
	summary_path = run_dir / "summary.json"
	status = get_status(status_file(run_dir))
	if status is None:
		raise HTTPException(status_code=404, detail="Run not found")
	if status["status"] != "succeeded":
		raise HTTPException(status_code=409, detail=f"Run not completed: {status['status']}")
	if not summary_path.exists():
		raise HTTPException(status_code=500, detail="Summary not found")
	data = read_json(summary_path)
	return RunSummary(run_id=run_id, **data)


@app.get("/streams/{run_id}")
def stream_run_logs(run_id: str) -> StreamingResponse:
	run_dir = RUNS_DIR / run_id
	log_path = run_dir / "stdout.log"
	if not log_path.exists():
		raise HTTPException(status_code=404, detail="Run/log not found")

	async def event_stream() -> Any:
		with log_path.open("r", encoding="utf-8") as f:
			# Start from beginning; simple tail-like behavior
			while True:
				line = f.readline()
				if line:
					yield f"data: {line.strip()}\n\n"
				else:
					await asyncio.sleep(0.5)

	return StreamingResponse(event_stream(), media_type="text/event-stream")

