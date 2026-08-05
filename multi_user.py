"""Multi-user assignment, authentication, and review-state helpers."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Mapping

ADMIN_USERNAME = "admin"
USER_IDS = tuple(f"user{i:02d}" for i in range(1, 21))

STATE_DIR_NAME = ".mesh_reviewer"
ASSIGNMENTS_FILE = "assignments.json"
REVIEWS_DIR = "reviews"
STORAGE_SECRET_FILE = "storage_secret"

_STATE_LOCK = threading.RLock()


def build_credentials(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the fixed credential map, allowing environment overrides."""
    env = os.environ if environ is None else environ
    common_user_password = env.get("MESH_REVIEWER_USER_PASSWORD")
    credentials = {
        user_id: env.get(
            f"MESH_REVIEWER_{user_id.upper()}_PASSWORD",
            common_user_password or user_id,
        )
        for user_id in USER_IDS
    }
    credentials[ADMIN_USERNAME] = env.get(
        "MESH_REVIEWER_ADMIN_PASSWORD", "qiuzhi2026"
    )
    return credentials


def authenticate(
    username: str, password: str, credentials: Mapping[str, str]
) -> str | None:
    """Return the account role for valid credentials, otherwise ``None``."""
    expected = credentials.get(username)
    if expected is None or not hmac.compare_digest(password, expected):
        return None
    return "admin" if username == ADMIN_USERNAME else "reviewer"


def ensure_state_dir(dataset_dir: Path) -> Path:
    state_dir = dataset_dir / STATE_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_storage_secret(state_dir: Path) -> str:
    """Return a stable NiceGUI storage secret without hard-coding one."""
    env_secret = os.environ.get("MESH_REVIEWER_STORAGE_SECRET")
    if env_secret:
        return env_secret

    path = state_dir / STORAGE_SECRET_FILE
    with _STATE_LOCK:
        if path.is_file():
            return path.read_text().strip()
        secret = secrets.token_urlsafe(48)
        path.write_text(secret)
        path.chmod(0o600)
        return secret


def _read_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return default.copy()
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON state file: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected a JSON object in state file: {path}")
    return data


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(fd, "w") as tmp_file:
            json.dump(data, tmp_file, indent=2, sort_keys=True)
            tmp_file.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def ensure_assignments(
    state_dir: Path,
    files: Iterable[Path],
    user_ids: tuple[str, ...] = USER_IDS,
) -> dict[str, str]:
    """Assign new files evenly while preserving every existing assignment."""
    if not user_ids:
        raise ValueError("At least one reviewer is required")

    file_names = sorted({path.name for path in files})
    assignment_path = state_dir / ASSIGNMENTS_FILE

    with _STATE_LOCK:
        payload = _read_json(assignment_path, {})
        raw_assignments = payload.get("assignments", {})
        if not isinstance(raw_assignments, dict):
            raise RuntimeError(
                f"Expected 'assignments' to be an object in {assignment_path}"
            )

        assignments = {
            str(file_name): str(user_id)
            for file_name, user_id in raw_assignments.items()
            if str(user_id) in user_ids
        }
        active_names = set(file_names)
        counts = {
            user_id: sum(
                1
                for file_name, assigned_user in assignments.items()
                if file_name in active_names and assigned_user == user_id
            )
            for user_id in user_ids
        }

        changed = payload.get("version") != 1 or payload.get("users") != list(
            user_ids
        )
        for file_name in file_names:
            if file_name in assignments:
                continue
            user_id = min(user_ids, key=lambda item: (counts[item], item))
            assignments[file_name] = user_id
            counts[user_id] += 1
            changed = True

        if changed or not assignment_path.is_file():
            _write_json_atomic(
                assignment_path,
                {
                    "version": 1,
                    "users": list(user_ids),
                    "assignments": assignments,
                },
            )

    return assignments


def files_for_user(
    files: Iterable[Path], assignments: Mapping[str, str], user_id: str
) -> list[Path]:
    if user_id not in USER_IDS:
        raise ValueError(f"Unknown reviewer: {user_id}")
    return sorted(path for path in files if assignments.get(path.name) == user_id)


def _review_path(state_dir: Path, user_id: str) -> Path:
    if user_id not in USER_IDS:
        raise ValueError(f"Unknown reviewer: {user_id}")
    return state_dir / REVIEWS_DIR / f"{user_id}.json"


def load_review_state(state_dir: Path, user_id: str) -> dict[str, str]:
    with _STATE_LOCK:
        state = _read_json(_review_path(state_dir, user_id), {})
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in state.items()
    ):
        raise RuntimeError(f"Invalid review state for {user_id}")
    return state


def set_review_status(
    state_dir: Path, user_id: str, file_key: str, status: str
) -> dict[str, str]:
    """Merge one status into the latest user state and persist it atomically."""
    path = _review_path(state_dir, user_id)
    with _STATE_LOCK:
        state = _read_json(path, {})
        state[file_key] = status
        _write_json_atomic(path, state)
        return state


def progress_rows(
    files: Iterable[Path], assignments: Mapping[str, str], state_dir: Path
) -> list[dict]:
    files_by_name = {path.name: path for path in files}
    rows: list[dict] = []
    for user_id in USER_IDS:
        user_files = [
            path
            for name, path in files_by_name.items()
            if assignments.get(name) == user_id
        ]
        state = load_review_state(state_dir, user_id)
        file_keys = [path.stem.removesuffix(".geom") for path in user_files]
        accepted = sum(state.get(key) == "accepted" for key in file_keys)
        denied = sum(state.get(key) == "denied" for key in file_keys)
        assigned = len(user_files)
        reviewed = accepted + denied
        rows.append(
            {
                "user": user_id,
                "assigned": assigned,
                "reviewed": reviewed,
                "accepted": accepted,
                "denied": denied,
                "remaining": assigned - reviewed,
                "progress": f"{(reviewed / assigned * 100):.0f}%" if assigned else "0%",
            }
        )
    return rows
