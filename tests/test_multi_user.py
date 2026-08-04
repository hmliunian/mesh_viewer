from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_user import (
    ADMIN_USERNAME,
    USER_IDS,
    authenticate,
    build_credentials,
    ensure_assignments,
    files_for_user,
    load_review_state,
    progress_rows,
    set_review_status,
)


class AssignmentTests(unittest.TestCase):
    def test_eighty_files_are_split_four_per_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            files = [Path(f"{index:04d}.geom.npz") for index in range(80)]

            assignments = ensure_assignments(state_dir, files)

            self.assertEqual(set(assignments), {path.name for path in files})
            self.assertEqual(
                [len(files_for_user(files, assignments, user)) for user in USER_IDS],
                [4] * 20,
            )

    def test_existing_assignments_stay_stable_when_a_file_is_added(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            files = [Path(f"{index:04d}.geom.npz") for index in range(40)]
            original = ensure_assignments(state_dir, files)

            added_file = Path("new.geom.npz")
            updated = ensure_assignments(state_dir, [added_file, *files])

            self.assertEqual(
                {path.name: updated[path.name] for path in files}, original
            )
            self.assertEqual(updated[added_file.name], USER_IDS[0])

    def test_assignment_file_is_reused_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            files = [Path(f"{index:04d}.geom.npz") for index in range(25)]
            first = ensure_assignments(state_dir, files)

            payload = json.loads((state_dir / "assignments.json").read_text())
            second = ensure_assignments(state_dir, reversed(files))

            self.assertEqual(first, second)
            self.assertEqual(payload["assignments"], second)


class ReviewStateTests(unittest.TestCase):
    def test_user_states_are_isolated_and_progress_is_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            files = [Path("a.geom.npz"), Path("b.geom.npz")]
            assignments = {
                "a.geom.npz": "user01",
                "b.geom.npz": "user02",
            }

            set_review_status(state_dir, "user01", "a", "accepted")
            set_review_status(state_dir, "user02", "b", "denied")

            self.assertEqual(load_review_state(state_dir, "user01"), {"a": "accepted"})
            self.assertEqual(load_review_state(state_dir, "user02"), {"b": "denied"})
            rows = {
                row["user"]: row for row in progress_rows(files, assignments, state_dir)
            }
            self.assertEqual(rows["user01"]["accepted"], 1)
            self.assertEqual(rows["user02"]["denied"], 1)

    def test_status_update_merges_with_latest_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            set_review_status(state_dir, "user01", "a", "accepted")
            set_review_status(state_dir, "user01", "b", "denied")

            self.assertEqual(
                load_review_state(state_dir, "user01"),
                {"a": "accepted", "b": "denied"},
            )


class AuthenticationTests(unittest.TestCase):
    def test_default_credentials_use_account_names(self) -> None:
        credentials = build_credentials({})

        self.assertEqual(authenticate(ADMIN_USERNAME, "admin", credentials), "admin")
        self.assertEqual(authenticate("user01", "user01", credentials), "reviewer")
        self.assertIsNone(authenticate("user01", "wrong", credentials))

    def test_environment_can_override_passwords(self) -> None:
        credentials = build_credentials(
            {
                "MESH_REVIEWER_ADMIN_PASSWORD": "admin-secret",
                "MESH_REVIEWER_USER_PASSWORD": "shared-secret",
                "MESH_REVIEWER_USER02_PASSWORD": "user02-secret",
            }
        )

        self.assertEqual(
            authenticate(ADMIN_USERNAME, "admin-secret", credentials), "admin"
        )
        self.assertEqual(
            authenticate("user01", "shared-secret", credentials), "reviewer"
        )
        self.assertEqual(
            authenticate("user02", "user02-secret", credentials), "reviewer"
        )


if __name__ == "__main__":
    unittest.main()
