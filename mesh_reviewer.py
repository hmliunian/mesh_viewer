#!/usr/bin/env python3
"""Mesh Reviewer GUI — review .geom.npz meshes and export accepted ones.

Usage:
    uv run python tools/mesh_reviewer.py \
        --dataset-dir /path/to/preprocess/timestamp \
        --export-dir /tmp/reviewed_export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from fastapi.responses import RedirectResponse
from nicegui import app, ui

from consts import (
    BADGE_OK,
    CONTENT_AREA,
    FOOTER,
    HEADER,
    PANEL,
    SCENE_BG,
    SECTION_HEADER,
    inject_styles,
)
from multi_user import (
    ADMIN_USERNAME,
    USER_IDS,
    authenticate,
    build_credentials,
    ensure_assignments,
    ensure_state_dir,
    files_for_user,
    get_storage_secret,
    load_review_state,
    progress_rows,
    set_review_status,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_ACCEPT = "accepted"
STATUS_DENY = "denied"

COLOR_ACCEPT = "#4ade80"
COLOR_DENY = "#f87171"
COLOR_PENDING = "#9ca3af"
COLOR_BBOX = "#facc15"

STATUS_ICONS = {
    STATUS_ACCEPT: ("check_circle", COLOR_ACCEPT),
    STATUS_DENY: ("cancel", COLOR_DENY),
}


# ---------------------------------------------------------------------------
# GLB cache
# ---------------------------------------------------------------------------

_glb_cache: dict[str, str] = {}  # file stem -> media URL
_glb_tmpdir = tempfile.mkdtemp(prefix="mesh_reviewer_")


def _get_glb_url(npz_path: Path) -> str:
    """Load .geom.npz, export as GLB, register with NiceGUI, return URL."""
    stem = npz_path.stem.removesuffix(".geom")
    if stem in _glb_cache:
        return _glb_cache[stem]

    with np.load(npz_path, allow_pickle=False) as a:
        vertices = np.asarray(a["vertices"], dtype=np.float64)
        faces = np.asarray(a["faces"], dtype=np.int64)
        vertex_normals = np.asarray(a["vertex_normals"], dtype=np.float64)

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_normals=vertex_normals, process=False
    )
    glb_path = Path(_glb_tmpdir) / f"{stem}.glb"
    mesh.export(glb_path, file_type="glb")

    digest = hashlib.sha1(str(npz_path).encode()).hexdigest()[:12]
    url = app.add_media_file(local_file=glb_path, url_path=f"/mesh_{digest}.glb")
    _glb_cache[stem] = url
    return url


# ---------------------------------------------------------------------------
# NPZ metadata
# ---------------------------------------------------------------------------


def _load_meta(npz_path: Path) -> dict:
    """Load bounding-box and geometry stats from a .geom.npz file."""
    with np.load(npz_path, allow_pickle=False) as a:
        bounds_min = np.asarray(a["bounds_min"], dtype=np.float64)
        bounds_max = np.asarray(a["bounds_max"], dtype=np.float64)
        n_verts = a["vertices"].shape[0]
        n_faces = a["faces"].shape[0]
        volume = float(a["volume"])
    dims = bounds_max - bounds_min
    return {
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "dims": dims,
        "n_verts": n_verts,
        "n_faces": n_faces,
        "volume": volume,
    }


# ---------------------------------------------------------------------------
# Bounding box lines
# ---------------------------------------------------------------------------


def _draw_bbox(scene_obj: ui.scene, bmin: np.ndarray, bmax: np.ndarray) -> None:
    """Draw 12 AABB edges in the 3D scene."""
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # verticals
    ]
    for a, b in edges:
        scene_obj.line(corners[a], corners[b]).material(COLOR_BBOX)


# ---------------------------------------------------------------------------
# Page builder
# ---------------------------------------------------------------------------


def _logout() -> None:
    app.storage.user.clear()
    ui.navigate.to("/login")


def build_page(
    dataset_dir: Path,
    export_dir: Path,
    state_dir: Path,
    username: str,
    files: list[Path],
) -> None:
    inject_styles(ui)

    if not files:
        with ui.header(fixed=False).classes(HEADER):
            with ui.row().classes("items-center gap-2"):
                ui.icon("view_in_ar").classes("text-white text-2xl")
                ui.label("Mesh Reviewer").classes("text-h6 font-bold text-white")
            with ui.row().classes("items-center gap-2"):
                ui.label(username).classes("text-sm text-gray-300")
                ui.button(icon="logout", on_click=_logout).props("flat round").tooltip(
                    "Log out"
                )
        with ui.column().classes(
            "w-full min-h-[calc(100vh-64px)] items-center justify-center gap-2"
        ):
            ui.icon("inbox", size="xl").classes("text-gray-500")
            ui.label("No meshes assigned").classes("text-lg text-gray-300")
        return

    state = load_review_state(state_dir, username)
    current_idx = {"value": 0}

    # --- pre-compute metadata for all files ---
    all_meta: list[dict] = [_load_meta(f) for f in files]

    # Filtered indices — indices into `files` that pass the current filter
    filtered_indices: list[int] = list(range(len(files)))

    # --- refs to updatable widgets ---
    refs: dict = {}

    def _counts() -> tuple[int, int, int]:
        statuses = [state.get(_stem(path)) for path in files]
        acc = sum(value == STATUS_ACCEPT for value in statuses)
        den = sum(value == STATUS_DENY for value in statuses)
        return acc, den, len(files) - acc - den

    def _stem(p: Path) -> str:
        return p.stem.removesuffix(".geom")

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(idx: int) -> None:
        idx = max(0, min(idx, len(files) - 1))
        current_idx["value"] = idx
        path = files[idx]
        stem = _stem(path)

        # 3D scene
        scene: ui.scene = refs["scene"]
        scene.clear()
        scene.spot_light(intensity=5.0).move(1, 2, 1)
        scene.spot_light(intensity=3.0).move(-1, -1, 2)
        glb_url = _get_glb_url(path)
        scene.gltf(glb_url)

        meta = all_meta[idx]
        _draw_bbox(scene, meta["bounds_min"], meta["bounds_max"])

        # info panel
        dims = meta["dims"]
        vol_str = (
            f"{meta['volume'] * 1e6:.2f} cm^3" if np.isfinite(meta["volume"]) else "N/A"
        )
        refs["info_name"].set_text(f"{stem}")
        refs["info_bbox"].set_text(
            f"Bounding Box: {dims[0] * 100:.2f} x {dims[1] * 100:.2f} x {dims[2] * 100:.2f} cm"
        )
        refs["info_geo"].set_text(
            f"Vertices: {meta['n_verts']:,}  |  Faces: {meta['n_faces']:,}  |  Volume: {vol_str}"
        )

        _update_list_highlight()
        _update_footer()

    def _update_footer() -> None:
        acc, den, rem = _counts()
        total_reviewed = acc + den
        showing = len(filtered_indices)
        if showing < len(files):
            footer_text = (
                f"Showing {showing}/{len(files)}  |  "
                f"Reviewed {total_reviewed}/{len(files)}  |  "
                f"Accepted: {acc}  Denied: {den}  Remaining: {rem}"
            )
        else:
            footer_text = (
                f"Reviewed {total_reviewed}/{len(files)}  |  "
                f"Accepted: {acc}  Denied: {den}  Remaining: {rem}"
            )
        refs["footer_label"].set_text(footer_text)
        refs["progress"].set_value(total_reviewed / len(files) if files else 0)
        refs["stat_acc"].set_text(f"Accepted: {acc}")
        refs["stat_den"].set_text(f"Denied: {den}")
        refs["stat_rem"].set_text(f"Remaining: {rem}")

    def _update_list_highlight() -> None:
        idx = current_idx["value"]
        for i, row in enumerate(refs["list_rows"]):
            stem = _stem(files[i])
            st = state.get(stem)
            hidden = " hidden" if i not in filtered_indices else ""
            if i == idx:
                row.classes(
                    replace=f"w-full cursor-pointer px-2 py-1 rounded bg-[#0277BD]/40{hidden}"
                )
            elif st == STATUS_ACCEPT:
                row.classes(
                    replace=f"w-full cursor-pointer px-2 py-1 rounded bg-[#4ade80]/10{hidden}"
                )
            elif st == STATUS_DENY:
                row.classes(
                    replace=f"w-full cursor-pointer px-2 py-1 rounded bg-[#f87171]/10{hidden}"
                )
            else:
                row.classes(
                    replace=f"w-full cursor-pointer px-2 py-1 rounded hover:bg-white/5{hidden}"
                )

    def _update_status_icons() -> None:
        for i, icon_el in enumerate(refs["list_icons"]):
            stem = _stem(files[i])
            st = state.get(stem)
            if st in STATUS_ICONS:
                icon_name, color = STATUS_ICONS[st]
                icon_el.props(f'name="{icon_name}" color="{color}"')
                icon_el.set_visibility(True)
            else:
                icon_el.set_visibility(False)

    # ------------------------------------------------------------------
    # Bounding-box filter
    # ------------------------------------------------------------------

    def _apply_filter() -> None:
        filtered_indices.clear()
        x_min = refs["filter_x_min"].value
        x_max = refs["filter_x_max"].value
        y_min = refs["filter_y_min"].value
        y_max = refs["filter_y_max"].value
        z_min = refs["filter_z_min"].value
        z_max = refs["filter_z_max"].value

        for i, meta in enumerate(all_meta):
            dims_cm = meta["dims"] * 100
            if (
                (x_min is None or dims_cm[0] >= x_min)
                and (x_max is None or dims_cm[0] <= x_max)
                and (y_min is None or dims_cm[1] >= y_min)
                and (y_max is None or dims_cm[1] <= y_max)
                and (z_min is None or dims_cm[2] >= z_min)
                and (z_max is None or dims_cm[2] <= z_max)
            ):
                filtered_indices.append(i)

        # Update matched count
        refs["filter_count"].set_text(
            f"Matched: {len(filtered_indices)}/{len(files)}"
        )
        refs["objects_header"].set_text(
            f"Objects ({len(filtered_indices)})"
        )

        # Show/hide list rows
        for i, row in enumerate(refs["list_rows"]):
            if i in filtered_indices:
                row.classes(remove="hidden")
            else:
                row.classes(add="hidden")

        # If current selection is hidden, auto-select first visible
        if current_idx["value"] not in filtered_indices and filtered_indices:
            select(filtered_indices[0])
        else:
            _update_footer()

    def _reset_filter() -> None:
        refs["filter_x_min"].set_value(None)
        refs["filter_x_max"].set_value(None)
        refs["filter_y_min"].set_value(None)
        refs["filter_y_max"].set_value(None)
        refs["filter_z_min"].set_value(None)
        refs["filter_z_max"].set_value(None)
        _apply_filter()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _set_status(status: str) -> None:
        idx = current_idx["value"]
        stem = _stem(files[idx])
        latest_state = set_review_status(state_dir, username, stem, status)
        state.clear()
        state.update(latest_state)
        _update_status_icons()
        # auto-advance to next visible item
        if idx in filtered_indices:
            pos = filtered_indices.index(idx)
            if pos + 1 < len(filtered_indices):
                select(filtered_indices[pos + 1])
            else:
                _update_list_highlight()
                _update_footer()
        else:
            _update_list_highlight()
            _update_footer()

    def _accept() -> None:
        _set_status(STATUS_ACCEPT)

    def _deny() -> None:
        _set_status(STATUS_DENY)

    def _prev() -> None:
        idx = current_idx["value"]
        if idx in filtered_indices:
            pos = filtered_indices.index(idx)
            if pos > 0:
                select(filtered_indices[pos - 1])
        elif filtered_indices:
            # Current item not visible; jump to nearest visible before it
            for fi in reversed(filtered_indices):
                if fi < idx:
                    select(fi)
                    return
            select(filtered_indices[0])

    def _next() -> None:
        idx = current_idx["value"]
        if idx in filtered_indices:
            pos = filtered_indices.index(idx)
            if pos + 1 < len(filtered_indices):
                select(filtered_indices[pos + 1])
        elif filtered_indices:
            # Current item not visible; jump to nearest visible after it
            for fi in filtered_indices:
                if fi > idx:
                    select(fi)
                    return
            select(filtered_indices[-1])

    async def _export() -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        accepted = [f for f in files if state.get(_stem(f)) == STATUS_ACCEPT]
        if not accepted:
            ui.notify("No accepted meshes to export.", type="warning")
            return

        manifest = []
        for f in accepted:
            shutil.copy2(f, export_dir / f.name)
            meta = _load_meta(f)
            dims = meta["dims"]
            manifest.append(
                {
                    "file": f.name,
                    "bbox_cm": [round(d * 100, 3) for d in dims],
                    "reviewed_by": username,
                    "n_verts": meta["n_verts"],
                    "n_faces": meta["n_faces"],
                    "volume_cm3": round(float(meta["volume"]) * 1e6, 4)
                    if np.isfinite(meta["volume"])
                    else None,
                }
            )

        (export_dir / f"manifest_{username}.json").write_text(
            json.dumps(manifest, indent=2)
        )
        ui.notify(
            f"Exported {len(accepted)} meshes to {export_dir}",
            type="positive",
        )

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _on_key(e) -> None:
        if not e.action.keydown:
            return
        has_mod = (
            e.modifiers.alt or e.modifiers.ctrl or e.modifiers.meta or e.modifiers.shift
        )
        if e.key == "a" and not has_mod:
            _accept()
        elif e.key == "d" and not has_mod:
            _deny()
        elif e.key.arrow_left:
            _prev()
        elif e.key.arrow_right:
            _next()

    ui.keyboard(on_key=_on_key)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    # Header
    with ui.header(fixed=False).classes(HEADER):
        with ui.row().classes("items-center gap-2"):
            ui.icon("view_in_ar").classes("text-white text-2xl")
            ui.label("Mesh Reviewer").classes(
                "text-h6 font-bold text-white tracking-wide"
            )
            ui.label(username).classes(
                "text-xs font-medium text-gray-300 border border-gray-600 "
                "px-2 py-1 rounded"
            )
        with ui.row().classes("items-center gap-2"):
            ui.button("Export Accepted", icon="file_download", on_click=_export).props(
                "flat text-color=white"
            )
            ui.button(icon="logout", on_click=_logout).props("flat round").tooltip(
                "Log out"
            )

    # Main content
    with ui.splitter(value=25).classes(CONTENT_AREA) as splitter:
        # ----- left sidebar -----
        with splitter.before:
            with ui.column().classes(PANEL):
                # Stats
                ui.label("Progress").classes(SECTION_HEADER)
                acc, den, rem = _counts()
                with ui.column().classes("gap-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("check_circle", color=COLOR_ACCEPT, size="xs")
                        refs["stat_acc"] = ui.label(f"Accepted: {acc}").classes(
                            "text-sm text-gray-300"
                        )
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("cancel", color=COLOR_DENY, size="xs")
                        refs["stat_den"] = ui.label(f"Denied: {den}").classes(
                            "text-sm text-gray-300"
                        )
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("pending", color=COLOR_PENDING, size="xs")
                        refs["stat_rem"] = ui.label(f"Remaining: {rem}").classes(
                            "text-sm text-gray-300"
                        )

                ui.separator()

                # Bounding-box filter
                ui.label("Filter (bbox cm)").classes(SECTION_HEADER)
                axis_labels = ["X", "Y", "Z"]
                axis_keys = ["x", "y", "z"]
                for ax_i, (label, key) in enumerate(zip(axis_labels, axis_keys)):
                    with ui.row(wrap=False).classes(
                        "w-full items-center gap-1"
                    ):
                        ui.label(label).classes(
                            "text-xs text-gray-400 w-3 flex-shrink-0"
                        )
                        refs[f"filter_{key}_min"] = (
                            ui.number(
                                label="Min",
                                value=None,
                                format="%.1f",
                                on_change=lambda _: _apply_filter(),
                            )
                            .props("dense outlined dark")
                            .classes("flex-grow")
                            .style("min-width: 0")
                        )
                        refs[f"filter_{key}_max"] = (
                            ui.number(
                                label="Max",
                                value=None,
                                format="%.1f",
                                on_change=lambda _: _apply_filter(),
                            )
                            .props("dense outlined dark")
                            .classes("flex-grow")
                            .style("min-width: 0")
                        )
                with ui.row(wrap=False).classes("w-full items-center justify-between"):
                    refs["filter_count"] = ui.label(
                        f"Matched: {len(files)}/{len(files)}"
                    ).classes("text-xs text-gray-400")
                    ui.button(
                        "Reset", icon="restart_alt", on_click=_reset_filter
                    ).props("flat dense size=sm").classes("text-gray-400")

                ui.separator()

                # Object list
                refs["objects_header"] = ui.label(
                    f"Objects ({len(files)})"
                ).classes(SECTION_HEADER)
                refs["list_rows"] = []
                refs["list_icons"] = []
                with ui.scroll_area().classes("w-full flex-grow"):
                    for i, f in enumerate(files):
                        stem = _stem(f)
                        idx_val = i  # capture

                        with ui.row(wrap=False).classes(
                            "w-full cursor-pointer px-2 py-1 rounded hover:bg-white/5"
                        ) as row:
                            row.on("click", lambda _, idx=idx_val: select(idx))
                            icon_el = ui.icon("check_circle", size="xs").classes(
                                "flex-shrink-0"
                            )
                            icon_el.set_visibility(False)
                            ui.label(stem).classes(
                                "text-xs text-gray-300 truncate flex-grow"
                            ).style("max-width: 200px")
                            refs["list_rows"].append(row)
                            refs["list_icons"].append(icon_el)

                ui.separator()

                # Paths
                ui.label("Paths").classes(SECTION_HEADER)
                ui.label(f"Dataset: {dataset_dir}").classes(
                    "text-xs text-gray-500 break-all"
                )
                ui.label(f"Export: {export_dir}").classes(
                    "text-xs text-gray-500 break-all"
                )
                ui.label(f"Reviewer: {username}").classes(
                    "text-xs text-gray-500 break-all"
                )

        # ----- right content -----
        with splitter.after:
            with ui.column().classes("w-full h-full p-3 gap-3"):
                # 3D scene
                refs["scene"] = ui.scene(
                    grid=False,
                    background_color=SCENE_BG,
                ).classes("w-full flex-grow rounded-2xl overflow-hidden min-h-0")

                # Info panel
                with ui.card().classes(
                    "w-full !bg-[#1e1e2e] border border-gray-700 rounded-xl p-4"
                ):
                    refs["info_name"] = ui.label("—").classes(
                        "text-lg font-bold text-white"
                    )
                    refs["info_bbox"] = ui.label("Bounding Box: —").classes(
                        "text-sm text-gray-300"
                    )
                    refs["info_geo"] = ui.label(
                        "Vertices: — | Faces: — | Volume: —"
                    ).classes("text-sm text-gray-400")

                # Action buttons
                with ui.row().classes("w-full justify-center gap-4"):
                    ui.button(
                        "Accept", icon="check", color="green", on_click=_accept
                    ).props("rounded").classes("px-6")
                    ui.button("Deny", icon="close", color="red", on_click=_deny).props(
                        "rounded"
                    ).classes("px-6")
                    ui.separator().props("vertical")
                    ui.button("Prev", icon="arrow_back", on_click=_prev).props(
                        "flat rounded"
                    )
                    ui.button("Next", icon="arrow_forward", on_click=_next).props(
                        "flat rounded"
                    )

    # Footer
    with ui.footer(fixed=False).classes(FOOTER):
        with ui.row().classes("w-full items-center gap-4"):
            with ui.row().classes(BADGE_OK):
                refs["footer_label"] = ui.label(f"Reviewed 0/{len(files)}").classes(
                    "text-sm text-white/90"
                )
            refs["progress"] = ui.linear_progress(value=0, show_value=False).classes(
                "flex-grow"
            )

    # Initialize status icons from saved state and select first item
    _update_status_icons()
    # Find first un-reviewed item, or fall back to 0
    start_idx = 0
    for i, f in enumerate(files):
        if _stem(f) not in state:
            start_idx = i
            break
    select(start_idx)


# ---------------------------------------------------------------------------
# Authentication and admin pages
# ---------------------------------------------------------------------------

def build_login_page(credentials: dict[str, str]) -> None:
    inject_styles(ui)

    def _sign_in() -> None:
        username = str(username_input.value or "")
        password = str(password_input.value or "")
        role = authenticate(username, password, credentials)
        if role is None:
            password_input.set_value("")
            ui.notify("Invalid username or password.", type="negative")
            return

        app.storage.user.clear()
        app.storage.user.update(
            {
                "authenticated": True,
                "username": username,
                "role": role,
            }
        )
        ui.navigate.to("/admin" if role == "admin" else "/")

    with ui.column().classes(
        "w-full min-h-screen items-center justify-center px-4 bg-[#11111b]"
    ):
        with ui.card().classes(
            "w-full max-w-sm !bg-[#1e1e2e] border border-gray-700 rounded-lg p-6"
        ):
            with ui.row().classes("items-center gap-3 mb-3"):
                ui.icon("view_in_ar").classes("text-white text-3xl")
                ui.label("Mesh Reviewer").classes("text-h5 font-bold text-white")
            username_input = (
                ui.select(
                    options=[ADMIN_USERNAME, *USER_IDS],
                    label="Username",
                )
                .props("outlined dark")
                .classes("w-full")
            )
            password_input = (
                ui.input(
                    label="Password",
                    password=True,
                    password_toggle_button=True,
                )
                .props("outlined dark")
                .classes("w-full")
            )
            password_input.on("keydown.enter", lambda _: _sign_in())
            ui.button("Sign in", icon="login", on_click=_sign_in).classes(
                "w-full mt-2"
            )


def build_admin_page(
    dataset_dir: Path,
    export_dir: Path,
    state_dir: Path,
    files: list[Path],
    assignments: dict[str, str],
) -> None:
    inject_styles(ui)
    refs: dict = {}

    async def _export_all() -> None:
        states = {
            user_id: load_review_state(state_dir, user_id) for user_id in USER_IDS
        }
        accepted: list[tuple[Path, str]] = []
        for path in files:
            user_id = assignments.get(path.name)
            if user_id is None:
                continue
            stem = path.stem.removesuffix(".geom")
            if states[user_id].get(stem) == STATUS_ACCEPT:
                accepted.append((path, user_id))

        if not accepted:
            ui.notify("No accepted meshes to export.", type="warning")
            return

        export_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for path, user_id in accepted:
            shutil.copy2(path, export_dir / path.name)
            meta = _load_meta(path)
            dims = meta["dims"]
            manifest.append(
                {
                    "file": path.name,
                    "reviewed_by": user_id,
                    "bbox_cm": [round(d * 100, 3) for d in dims],
                    "n_verts": meta["n_verts"],
                    "n_faces": meta["n_faces"],
                    "volume_cm3": round(float(meta["volume"]) * 1e6, 4)
                    if np.isfinite(meta["volume"])
                    else None,
                }
            )

        (export_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        ui.notify(
            f"Exported {len(accepted)} meshes to {export_dir}",
            type="positive",
        )

    def _refresh() -> None:
        rows = progress_rows(files, assignments, state_dir)
        accepted = sum(row["accepted"] for row in rows)
        denied = sum(row["denied"] for row in rows)
        reviewed = accepted + denied
        remaining = len(files) - reviewed

        refs["total"].set_text(str(len(files)))
        refs["reviewed"].set_text(str(reviewed))
        refs["accepted"].set_text(str(accepted))
        refs["denied"].set_text(str(denied))
        refs["remaining"].set_text(str(remaining))
        refs["progress"].set_value(reviewed / len(files) if files else 0)
        refs["table"].rows = rows
        refs["table"].update()

    with ui.header(fixed=False).classes(HEADER):
        with ui.row().classes("items-center gap-2"):
            ui.icon("view_in_ar").classes("text-white text-2xl")
            ui.label("Mesh Reviewer").classes("text-h6 font-bold text-white")
            ui.label("Admin").classes(
                "text-xs font-medium text-gray-300 border border-gray-600 "
                "px-2 py-1 rounded"
            )
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Export Accepted", icon="file_download", on_click=_export_all
            ).props("flat text-color=white")
            ui.button(icon="logout", on_click=_logout).props("flat round").tooltip(
                "Log out"
            )

    with ui.column().classes("w-full max-w-7xl mx-auto p-6 gap-5"):
        ui.label("Review Progress").classes("text-h5 font-bold text-white")
        with ui.row().classes(
            "w-full items-stretch justify-between gap-6 py-4 border-y border-gray-700"
        ):
            for label, key in (
                ("Total", "total"),
                ("Reviewed", "reviewed"),
                ("Accepted", "accepted"),
                ("Denied", "denied"),
                ("Remaining", "remaining"),
            ):
                with ui.column().classes("gap-1 min-w-24"):
                    ui.label(label).classes(SECTION_HEADER)
                    refs[key] = ui.label("0").classes(
                        "text-2xl font-semibold text-white"
                    )
        refs["progress"] = ui.linear_progress(value=0, show_value=False).classes(
            "w-full"
        )
        columns = [
            {"name": "user", "label": "Reviewer", "field": "user", "align": "left"},
            {"name": "assigned", "label": "Assigned", "field": "assigned"},
            {"name": "reviewed", "label": "Reviewed", "field": "reviewed"},
            {"name": "accepted", "label": "Accepted", "field": "accepted"},
            {"name": "denied", "label": "Denied", "field": "denied"},
            {"name": "remaining", "label": "Remaining", "field": "remaining"},
            {"name": "progress", "label": "Progress", "field": "progress"},
        ]
        refs["table"] = (
            ui.table(rows=[], columns=columns, row_key="user", pagination=20)
            .props("flat bordered dense dark")
            .classes("w-full")
        )
        ui.label(f"Dataset: {dataset_dir}").classes(
            "text-xs text-gray-500 break-all"
        )

    _refresh()
    ui.timer(2.0, _refresh)

# Main
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Review .geom.npz meshes")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing .geom.npz files",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./reviewed_export",
        help="Directory to export accepted meshes (default: ./reviewed_export)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port to run the server on (default: 8090)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    export_dir = Path(args.export_dir).resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")

    files = sorted(dataset_dir.glob("*.geom.npz"))
    state_dir = ensure_state_dir(dataset_dir)
    assignments = ensure_assignments(state_dir, files)
    credentials = build_credentials()
    storage_secret = get_storage_secret(state_dir)

    @ui.page("/login")
    def login():
        if app.storage.user.get("authenticated"):
            target = "/admin" if app.storage.user.get("role") == "admin" else "/"
            return RedirectResponse(target)
        build_login_page(credentials)

    @ui.page("/")
    def index():
        if not app.storage.user.get("authenticated"):
            return RedirectResponse("/login")
        if app.storage.user.get("role") == "admin":
            return RedirectResponse("/admin")

        username = app.storage.user.get("username")
        if username not in USER_IDS:
            app.storage.user.clear()
            return RedirectResponse("/login")

        user_files = files_for_user(files, assignments, username)
        build_page(
            dataset_dir,
            export_dir,
            state_dir,
            username,
            user_files,
        )

    @ui.page("/admin")
    def admin():
        if not app.storage.user.get("authenticated"):
            return RedirectResponse("/login")
        if app.storage.user.get("role") != "admin":
            return RedirectResponse("/")
        build_admin_page(
            dataset_dir,
            export_dir,
            state_dir,
            files,
            assignments,
        )

    ui.run(
        port=args.port,
        title="Mesh Reviewer",
        storage_secret=storage_secret,
        reload=False,
    )


if __name__ == "__main__":
    main()
