from __future__ import annotations

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_toml(path_like: str | Path) -> dict:
    path = resolve_project_path(path_like)
    with open(path, "rb") as f:
        return tomllib.load(f)
