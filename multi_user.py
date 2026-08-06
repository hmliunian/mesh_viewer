"""Multi-user assignment, authentication, and review-state helpers."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

ADMIN_USERNAME = "admin"
USER_IDS = tuple(f"user{i:02d}" for i in range(1, 21))

STATE_DIR_NAME = ".mesh_reviewer"
ASSIGNMENTS_FILE = "assignments.json"
GLOBAL_REVIEWS_FILE = "reviewed.json"
REVIEWS_DIR = "reviews"
STORAGE_SECRET_FILE = "storage_secret"

VALID_REVIEW_STATUSES = frozenset({"accepted", "denied"})

_STATE_LOCK = threading.RLock()


class StaleAssignmentError(RuntimeError):
    """Raised when a reviewer submits work from an obsolete assignment page."""


@dataclass(frozen=True)
class AssignmentSnapshot:
    generation: int
    users: tuple[str, ...]
    assignments: dict[str, str]
    batch_mode: bool = False


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
    """Return the account role for valid credentials, otherwise None."""
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


def _validate_user_ids(user_ids: Iterable[str]) -> tuple[str, ...]:
    users = tuple(user_ids)
    if not users:
        raise ValueError("At least one reviewer is required")
    if len(set(users)) != len(users):
        raise ValueError("Reviewer IDs must be unique")
    unknown = [user_id for user_id in users if user_id not in USER_IDS]
    if unknown:
        raise ValueError(f"Unknown reviewer: {unknown[0]}")
    return users


def _users_from_payload(payload: Mapping) -> tuple[str, ...]:
    raw_users = payload.get("users")
    if not isinstance(raw_users, list):
        return USER_IDS
    users = tuple(
        str(user_id)
        for user_id in raw_users
        if str(user_id) in USER_IDS
    )
    if not users or len(set(users)) != len(users):
        return USER_IDS
    return users


def _load_assignment_snapshot_unlocked(state_dir: Path) -> AssignmentSnapshot:
    path = state_dir / ASSIGNMENTS_FILE
    payload = _read_json(path, {})
    users = _users_from_payload(payload)
    raw_assignments = payload.get("assignments", {})
    if not isinstance(raw_assignments, dict):
        raise RuntimeError(f"Expected 'assignments' to be an object in {path}")

    assignments = {
        str(file_name): str(user_id)
        for file_name, user_id in raw_assignments.items()
        if str(user_id) in users
    }
    raw_generation = payload.get("generation", 0)
    generation = (
        raw_generation
        if isinstance(raw_generation, int) and raw_generation >= 0
        else 0
    )
    return AssignmentSnapshot(
        generation=generation,
        users=users,
        assignments=assignments,
        batch_mode=payload.get("batch_mode") is True,
    )


def load_assignment_snapshot(state_dir: Path) -> AssignmentSnapshot:
    with _STATE_LOCK:
        snapshot = _load_assignment_snapshot_unlocked(state_dir)
    return AssignmentSnapshot(
        generation=snapshot.generation,
        users=snapshot.users,
        assignments=snapshot.assignments.copy(),
        batch_mode=snapshot.batch_mode,
    )


def _write_assignment_snapshot_unlocked(
    state_dir: Path, snapshot: AssignmentSnapshot, quotas: Mapping[str, int] | None = None
) -> None:
    payload: dict = {
        "version": 2,
        "generation": snapshot.generation,
        "users": list(snapshot.users),
        "batch_mode": snapshot.batch_mode,
        "assignments": snapshot.assignments,
    }
    if quotas is not None:
        payload["quotas"] = dict(quotas)
    _write_json_atomic(state_dir / ASSIGNMENTS_FILE, payload)


def _review_path(state_dir: Path, user_id: str) -> Path:
    if user_id not in USER_IDS:
        raise ValueError(f"Unknown reviewer: {user_id}")
    return state_dir / REVIEWS_DIR / f"{user_id}.json"


def _load_review_state_unlocked(state_dir: Path, user_id: str) -> dict[str, str]:
    state = _read_json(_review_path(state_dir, user_id), {})
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in state.items()
    ):
        raise RuntimeError(f"Invalid review state for {user_id}")
    return {str(key): str(value) for key, value in state.items()}


def load_review_state(state_dir: Path, user_id: str) -> dict[str, str]:
    with _STATE_LOCK:
        return _load_review_state_unlocked(state_dir, user_id)


def _global_reviews_path(state_dir: Path) -> Path:
    return state_dir / GLOBAL_REVIEWS_FILE


def _load_global_reviews_unlocked(state_dir: Path) -> dict[str, dict[str, str]]:
    path = _global_reviews_path(state_dir)
    payload = _read_json(path, {})
    raw_reviews = payload.get("reviews", {})
    if not isinstance(raw_reviews, dict):
        raise RuntimeError(f"Expected 'reviews' to be an object in {path}")

    reviews: dict[str, dict[str, str]] = {}
    for file_name, raw_record in raw_reviews.items():
        if isinstance(raw_record, str):
            status = raw_record
            reviewer = ""
        elif isinstance(raw_record, dict):
            status = raw_record.get("status")
            reviewer = raw_record.get("reviewed_by", "")
        else:
            raise RuntimeError(f"Invalid review record for {file_name} in {path}")
        if status not in VALID_REVIEW_STATUSES or not isinstance(reviewer, str):
            raise RuntimeError(f"Invalid review record for {file_name} in {path}")
        reviews[str(file_name)] = {
            "status": str(status),
            "reviewed_by": reviewer,
        }
    return reviews


def load_global_reviews(state_dir: Path) -> dict[str, dict[str, str]]:
    with _STATE_LOCK:
        reviews = _load_global_reviews_unlocked(state_dir)
    return {file_name: record.copy() for file_name, record in reviews.items()}


def _write_global_reviews_unlocked(
    state_dir: Path, reviews: Mapping[str, Mapping[str, str]]
) -> None:
    _write_json_atomic(
        _global_reviews_path(state_dir),
        {
            "version": 1,
            "reviews": {
                file_name: dict(record) for file_name, record in reviews.items()
            },
        },
    )


def _file_key(path: Path) -> str:
    return path.stem.removesuffix(".geom")


def ensure_assignments(
    state_dir: Path,
    files: Iterable[Path],
    user_ids: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Load assignments and initialize legacy/non-batch states evenly.

    Once the admin has created a batch, deliberately unassigned pending files stay
    unassigned until the next explicit redistribution.
    """
    file_names = sorted({path.name for path in files})
    active_names = set(file_names)
    assignment_path = state_dir / ASSIGNMENTS_FILE

    with _STATE_LOCK:
        existed = assignment_path.is_file()
        previous = _load_assignment_snapshot_unlocked(state_dir)
        users = _validate_user_ids(user_ids) if user_ids is not None else previous.users
        reviewed = _load_global_reviews_unlocked(state_dir)
        reviewed_names = {
            file_name
            for file_name, record in reviewed.items()
            if record.get("status") in VALID_REVIEW_STATUSES
        }
        assignments = {
            file_name: assigned_user
            for file_name, assigned_user in previous.assignments.items()
            if file_name in active_names
            and file_name not in reviewed_names
            and assigned_user in users
        }

        should_fill_unassigned = (
            not existed or not previous.batch_mode or user_ids is not None
        )
        if should_fill_unassigned:
            counts = {
                user_id: sum(assigned == user_id for assigned in assignments.values())
                for user_id in users
            }
            for file_name in file_names:
                if file_name in reviewed_names or file_name in assignments:
                    continue
                user_id = min(users, key=lambda item: (counts[item], item))
                assignments[file_name] = user_id
                counts[user_id] += 1

        changed = assignments != previous.assignments or users != previous.users
        generation = max(previous.generation, 1)
        if existed and changed:
            generation += 1
        snapshot = AssignmentSnapshot(
            generation=generation,
            users=users,
            assignments=assignments,
            batch_mode=previous.batch_mode if existed else False,
        )
        _write_assignment_snapshot_unlocked(state_dir, snapshot)

    return assignments.copy()


def files_for_user(
    files: Iterable[Path], assignments: Mapping[str, str], user_id: str
) -> list[Path]:
    if user_id not in USER_IDS:
        raise ValueError(f"Unknown reviewer: {user_id}")
    return sorted(path for path in files if assignments.get(path.name) == user_id)


def set_review_status(
    state_dir: Path,
    user_id: str,
    file_key: str,
    status: str,
    *,
    file_name: str | None = None,
    expected_generation: int | None = None,
) -> dict[str, str]:
    """Persist one status, optionally validating the current assignment batch."""
    if user_id not in USER_IDS:
        raise ValueError(f"Unknown reviewer: {user_id}")
    if status not in VALID_REVIEW_STATUSES:
        raise ValueError(f"Invalid review status: {status}")
    if expected_generation is not None and file_name is None:
        raise ValueError("file_name is required when validating a generation")

    path = _review_path(state_dir, user_id)
    with _STATE_LOCK:
        if file_name is not None:
            snapshot = _load_assignment_snapshot_unlocked(state_dir)
            if (
                expected_generation is not None
                and snapshot.generation != expected_generation
            ):
                raise StaleAssignmentError("The assignment batch has changed")
            if snapshot.assignments.get(file_name) != user_id:
                raise StaleAssignmentError("This file is no longer assigned to you")
            if file_name in _load_global_reviews_unlocked(state_dir):
                raise StaleAssignmentError("This file has already been consolidated")

        state = _load_review_state_unlocked(state_dir, user_id)
        state[file_key] = status
        _write_json_atomic(path, state)
        return state


def effective_review_results(
    files: Iterable[Path], assignments: Mapping[str, str], state_dir: Path
) -> dict[str, dict[str, str]]:
    """Return consolidated results plus current, not-yet-consolidated reviews."""
    files_by_name = {path.name: path for path in files}
    with _STATE_LOCK:
        global_reviews = _load_global_reviews_unlocked(state_dir)
        results = {
            file_name: record.copy()
            for file_name, record in global_reviews.items()
            if file_name in files_by_name
        }
        states: dict[str, dict[str, str]] = {}
        for file_name, user_id in assignments.items():
            if file_name not in files_by_name or file_name in results:
                continue
            if user_id not in USER_IDS:
                continue
            if user_id not in states:
                states[user_id] = _load_review_state_unlocked(state_dir, user_id)
            status = states[user_id].get(_file_key(files_by_name[file_name]))
            if status in VALID_REVIEW_STATUSES:
                results[file_name] = {
                    "status": status,
                    "reviewed_by": user_id,
                }
    return results


def balanced_quotas(total: int, user_ids: Iterable[str]) -> dict[str, int]:
    users = _validate_user_ids(user_ids)
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("Total must be a non-negative integer")
    base, remainder = divmod(total, len(users))
    return {
        user_id: base + (index < remainder)
        for index, user_id in enumerate(users)
    }


def redistribute_pending(
    state_dir: Path,
    files: Iterable[Path],
    user_ids: Iterable[str],
    quotas: Mapping[str, int] | None = None,
) -> dict:
    """Consolidate completed reviews and create a fresh pending-work batch."""
    users = _validate_user_ids(user_ids)
    files_by_name = {path.name: path for path in files}
    file_names = sorted(files_by_name)

    with _STATE_LOCK:
        previous = _load_assignment_snapshot_unlocked(state_dir)
        global_reviews = _load_global_reviews_unlocked(state_dir)
        before_reviewed = {
            name for name in files_by_name if name in global_reviews
        }

        user_states: dict[str, dict[str, str]] = {}
        for file_name, assigned_user in previous.assignments.items():
            path = files_by_name.get(file_name)
            if path is None or assigned_user not in USER_IDS:
                continue
            if file_name in global_reviews:
                continue
            if assigned_user not in user_states:
                user_states[assigned_user] = _load_review_state_unlocked(
                    state_dir, assigned_user
                )
            status = user_states[assigned_user].get(_file_key(path))
            if status in VALID_REVIEW_STATUSES:
                global_reviews[file_name] = {
                    "status": status,
                    "reviewed_by": assigned_user,
                }

        pending_names = [
            file_name for file_name in file_names if file_name not in global_reviews
        ]
        if quotas is None:
            allocation = balanced_quotas(len(pending_names), users)
        else:
            unexpected = set(quotas) - set(users)
            missing = set(users) - set(quotas)
            if unexpected or missing:
                raise ValueError("Custom quantities must include exactly the active users")
            allocation = {}
            for user_id in users:
                value = quotas[user_id]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Allocation quantities must be non-negative integers")
                allocation[user_id] = value
            if sum(allocation.values()) > len(pending_names):
                raise ValueError(
                    "Planned assignments exceed the current unreviewed file count"
                )

        assignments: dict[str, str] = {}
        name_index = 0
        for user_id in users:
            for _ in range(allocation[user_id]):
                assignments[pending_names[name_index]] = user_id
                name_index += 1

        # A pending file may have stale per-user state from an older batch. Remove
        # those entries before publishing the new assignment generation.
        pending_keys = {_file_key(files_by_name[name]) for name in pending_names}
        for user_id in USER_IDS:
            review_path = _review_path(state_dir, user_id)
            if not review_path.is_file():
                continue
            state = _load_review_state_unlocked(state_dir, user_id)
            cleaned = {
                key: value for key, value in state.items() if key not in pending_keys
            }
            if cleaned != state:
                _write_json_atomic(review_path, cleaned)

        _write_global_reviews_unlocked(state_dir, global_reviews)
        generation = max(previous.generation, 0) + 1
        snapshot = AssignmentSnapshot(
            generation=generation,
            users=users,
            assignments=assignments,
            batch_mode=True,
        )
        _write_assignment_snapshot_unlocked(state_dir, snapshot, allocation)

        active_results = {
            name: record for name, record in global_reviews.items() if name in files_by_name
        }
        accepted = sum(
            record["status"] == "accepted" for record in active_results.values()
        )
        denied = sum(
            record["status"] == "denied" for record in active_results.values()
        )
        newly_reviewed = len(set(active_results) - before_reviewed)

    return {
        "generation": generation,
        "users": users,
        "assignments": assignments.copy(),
        "quotas": allocation.copy(),
        "newly_reviewed": newly_reviewed,
        "reviewed": accepted + denied,
        "accepted": accepted,
        "denied": denied,
        "remaining": len(pending_names),
        "assigned": len(assignments),
        "unassigned": len(pending_names) - len(assignments),
    }


def progress_rows(
    files: Iterable[Path],
    assignments: Mapping[str, str],
    state_dir: Path,
    user_ids: Iterable[str] = USER_IDS,
) -> list[dict]:
    files_by_name = {path.name: path for path in files}
    users = _validate_user_ids(user_ids)
    rows: list[dict] = []
    for user_id in users:
        user_files = [
            path
            for name, path in files_by_name.items()
            if assignments.get(name) == user_id
        ]
        state = load_review_state(state_dir, user_id)
        file_keys = [_file_key(path) for path in user_files]
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
