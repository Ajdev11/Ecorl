from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
from .io import ensure_dir, write_json, read_json


def status_file(run_dir: Path) -> Path:
	return run_dir / "status.json"


def set_status(path: Path, status: str, message: Optional[str] = None) -> None:
	data = {"status": status}
	if message is not None:
		data["message"] = message
	write_json(path, data)


def get_status(path: Path) -> Optional[Dict]:
	if not path.exists():
		return None
	return read_json(path)

