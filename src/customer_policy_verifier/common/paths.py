"""Repository locations and read-time resolution of frozen historical paths.

Moving an artifact must not rewrite its contents or invalidate its recorded
digest. The public migration map contains relative paths only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def project_root() -> Path:
    override = os.environ.get("CPV_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "configs" / "jev-training.json").is_file():
        return candidate
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "configs" / "jev-training.json").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


ROOT = project_root()


def migration_map(root: Path = ROOT) -> dict[str, str]:
    config = root / "configs" / "layout-migration.json"
    if not config.is_file():
        return {}
    return json.loads(config.read_text(encoding="utf-8"))["path_migrations"]


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve relocated project references without mutating frozen metadata.

    Paths outside this repository are left alone, so temporary fixtures and
    caller-supplied external data directories retain their normal semantics.
    References recorded on another machine can be rebased using the project
    directory name. Fresh output paths should use the current layout directly.
    """
    root = Path(root) if root is not None else ROOT
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            for name in ("jev-customer-policy-verifier", "customer-policy-verifier"):
                marker = f"/{name}/"
                if marker in path.as_posix():
                    relative = path.as_posix().split(marker, 1)[1]
                    break
            else:
                return path
    else:
        relative = path.as_posix()
    for old, new in sorted(migration_map(root).items(), key=lambda item: len(item[0]), reverse=True):
        if relative == old or relative.startswith(old + "/"):
            return root / (new + relative[len(old):])
    return root / relative
