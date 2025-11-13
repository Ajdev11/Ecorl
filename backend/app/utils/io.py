from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

try:
	import orjson  # type: ignore[import-not-found]
except Exception:
	class _OrjsonCompat:
		OPT_INDENT_2 = 0
		@staticmethod
		def dumps(data: Dict[str, Any], option: int | None = None) -> bytes:
			# Fallback to stdlib json with UTF-8 bytes
			return json.dumps(data, indent=2).encode("utf-8")
		@staticmethod
		def loads(b: bytes | str) -> Dict[str, Any]:
			if isinstance(b, (bytes, bytearray)):
				return json.loads(b.decode("utf-8"))
			return json.loads(b)
	orjson = _OrjsonCompat()  # type: ignore[assignment]
import pandas as pd


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
	ensure_dir(path.parent)
	path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def read_json(path: Path) -> Dict[str, Any]:
	return orjson.loads(path.read_bytes())


def write_parquet(df: pd.DataFrame, path: Path) -> None:
	ensure_dir(path.parent)
	try:
		# Prefer pyarrow if available
		import pyarrow  # type: ignore # noqa: F401
		df.to_parquet(path, engine="pyarrow", index=False)
	except Exception:
		# Fallback to CSV to avoid hard dependency during quickstart
		csv_path = path.with_suffix(".csv")
		df.to_csv(csv_path, index=False)


def append_log_line(path: Path, line: str) -> None:
	ensure_dir(path.parent)
	with path.open("a", encoding="utf-8") as f:
		f.write(line.rstrip("\n") + "\n")

