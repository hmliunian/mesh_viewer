from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_user import (
    ADMIN_USERNAME,
    StaleAssignmentError,
    USER_IDS,
    authenticate,
    balanced_quotas,
    build_credentials,
    effective_review_results,
    ensure_assignments,
    files_for_user,
    load_assignment_snapshot,
    load_global_reviews,
    load_review_state,
    progress_rows,
    redistribute_pending,
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


class RedistributionTests(unittest.TestCase):
    def test_balanced_quotas_differ_by_at_most_one(self) -> None:
        users = USER_IDS[:3]

        self.assertEqual(
            balanced_quotas(8, users),
            {
                "user01": 3,
                "user02": 3,
                "user03": 2,
            },
        )

    def test_refresh_consolidates_reviews_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            users = USER_IDS[:2]
            files = [Path(f"{index}.geom.npz") for index in range(5)]
            assignments = ensure_assignments(state_dir, files, users)
            reviewed_file = files[0]
            reviewer = assignments[reviewed_file.name]
            file_key = reviewed_file.stem.removesuffix(".geom")
            set_review_status(state_dir, reviewer, file_key, "accepted")

            self.assertEqual(load_global_reviews(state_dir), {})
            self.assertEqual(
                effective_review_results(files, assignments, state_dir)[
                    reviewed_file.name
                ]["status"],
                "accepted",
            )
            old_generation = load_assignment_snapshot(state_dir).generation

            first = redistribute_pending(state_dir, files, users)

            self.assertEqual(first["newly_reviewed"], 1)
            self.assertEqual(first["accepted"], 1)
            self.assertEqual(first["remaining"], 4)
            self.assertEqual(first["assigned"], 4)
            self.assertEqual(first["unassigned"], 0)
            snapshot = load_assignment_snapshot(state_dir)
            self.assertEqual(snapshot.generation, old_generation + 1)
            self.assertEqual(snapshot.users, users)
            self.assertNotIn(reviewed_file.name, snapshot.assignments)
            self.assertEqual(
                [
                    sum(owner == user_id for owner in snapshot.assignments.values())
                    for user_id in users
                ],
                [2, 2],
            )
            self.assertEqual(
                load_global_reviews(state_dir)[reviewed_file.name],
                {"status": "accepted", "reviewed_by": reviewer},
            )

            second = redistribute_pending(state_dir, files, users)

            self.assertEqual(second["newly_reviewed"], 0)
            self.assertEqual(second["accepted"], 1)
            self.assertEqual(len(load_global_reviews(state_dir)), 1)

    def test_custom_quantities_leave_a_persistent_unassigned_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            users = USER_IDS[:2]
            files = [Path(f"{index}.geom.npz") for index in range(7)]
            ensure_assignments(state_dir, files, users)

            result = redistribute_pending(
                state_dir,
                files,
                users,
                {"user01": 2, "user02": 1},
            )

            self.assertEqual(result["assigned"], 3)
            self.assertEqual(result["unassigned"], 4)
            self.assertEqual(result["quotas"], {"user01": 2, "user02": 1})
            snapshot = load_assignment_snapshot(state_dir)
            self.assertEqual(snapshot.users, users)
            self.assertTrue(snapshot.batch_mode)
            self.assertEqual(
                [
                    sum(owner == user_id for owner in snapshot.assignments.values())
                    for user_id in users
                ],
                [2, 1],
            )

            assignments_after_restart = ensure_assignments(state_dir, files)

            self.assertEqual(assignments_after_restart, snapshot.assignments)
            self.assertEqual(
                load_assignment_snapshot(state_dir).generation,
                snapshot.generation,
            )

    def test_submission_from_an_old_batch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            users = USER_IDS[:2]
            files = [Path(f"{index}.geom.npz") for index in range(4)]
            ensure_assignments(state_dir, files, users)
            old_snapshot = load_assignment_snapshot(state_dir)
            file_name, reviewer = next(iter(old_snapshot.assignments.items()))
            file_key = Path(file_name).stem.removesuffix(".geom")

            redistribute_pending(state_dir, files, users)

            with self.assertRaises(StaleAssignmentError):
                set_review_status(
                    state_dir,
                    reviewer,
                    file_key,
                    "denied",
                    file_name=file_name,
                    expected_generation=old_snapshot.generation,
                )

    def test_legacy_assignment_state_is_consolidated_on_first_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            files = [Path("legacy.geom.npz"), Path("pending.geom.npz")]
            (state_dir / "assignments.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "users": list(USER_IDS),
                        "assignments": {
                            "legacy.geom.npz": "user01",
                            "pending.geom.npz": "user02",
                        },
                    }
                )
            )
            set_review_status(state_dir, "user01", "legacy", "denied")

            result = redistribute_pending(state_dir, files, USER_IDS[:2])

            self.assertEqual(result["newly_reviewed"], 1)
            self.assertEqual(result["denied"], 1)
            self.assertEqual(
                load_global_reviews(state_dir)["legacy.geom.npz"],
                {"status": "denied", "reviewed_by": "user01"},
            )
            self.assertNotIn(
                "legacy.geom.npz",
                load_assignment_snapshot(state_dir).assignments,
            )


class AuthenticationTests(unittest.TestCase):
    def test_default_credentials_use_account_names(self) -> None:
        credentials = build_credentials({})

        self.assertEqual(
            authenticate(ADMIN_USERNAME, "qiuzhi2026", credentials), "admin"
        )
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
