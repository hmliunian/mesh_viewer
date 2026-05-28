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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_ACCEPT = "accepted"
STATUS_DENY = "denied"
STATE_FILE = "_review_state.json"

COLOR_ACCEPT = "#4ade80"
COLOR_DENY = "#f87171"
COLOR_PENDING = "#9ca3af"
COLOR_BBOX = "#facc15"

STATUS_ICONS = {
    STATUS_ACCEPT: ("check_circle", COLOR_ACCEPT),
    STATUS_DENY: ("cancel", COLOR_DENY),
}


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_state(dataset_dir: Path) -> dict[str, str]:
    state_path = dataset_dir / STATE_FILE
    if state_path.is_file():
        return json.loads(state_path.read_text())
    return {}


def _save_state(dataset_dir: Path, state: dict[str, str]) -> None:
    state_path = dataset_dir / STATE_FILE
    state_path.write_text(json.dumps(state, indent=2))


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


def build_page(dataset_dir: Path, export_dir: Path) -> None:
    inject_styles(ui)

    # --- data ---
    files = sorted(dataset_dir.glob("*.geom.npz"))
    if not files:
        ui.label("No .geom.npz files found in dataset directory.").classes(
            "text-xl text-red-400 m-8"
        )
        return

    state = _load_state(dataset_dir)
    current_idx = {"value": 0}

    # --- refs to updatable widgets ---
    refs: dict = {}

    def _counts() -> tuple[int, int, int]:
        acc = sum(1 for v in state.values() if v == STATUS_ACCEPT)
        den = sum(1 for v in state.values() if v == STATUS_DENY)
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

        meta = _load_meta(path)
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
        refs["footer_label"].set_text(
            f"Reviewed {total_reviewed}/{len(files)}  |  Accepted: {acc}  Denied: {den}  Remaining: {rem}"
        )
        refs["progress"].set_value(total_reviewed / len(files) if files else 0)
        refs["stat_acc"].set_text(f"Accepted: {acc}")
        refs["stat_den"].set_text(f"Denied: {den}")
        refs["stat_rem"].set_text(f"Remaining: {rem}")

    def _update_list_highlight() -> None:
        idx = current_idx["value"]
        for i, row in enumerate(refs["list_rows"]):
            stem = _stem(files[i])
            st = state.get(stem)
            if i == idx:
                row.classes(
                    replace="w-full cursor-pointer px-2 py-1 rounded bg-[#0277BD]/40"
                )
            elif st == STATUS_ACCEPT:
                row.classes(
                    replace="w-full cursor-pointer px-2 py-1 rounded bg-[#4ade80]/10"
                )
            elif st == STATUS_DENY:
                row.classes(
                    replace="w-full cursor-pointer px-2 py-1 rounded bg-[#f87171]/10"
                )
            else:
                row.classes(
                    replace="w-full cursor-pointer px-2 py-1 rounded hover:bg-white/5"
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
    # Actions
    # ------------------------------------------------------------------

    def _set_status(status: str) -> None:
        idx = current_idx["value"]
        stem = _stem(files[idx])
        state[stem] = status
        _save_state(dataset_dir, state)
        _update_status_icons()
        # auto-advance
        if idx < len(files) - 1:
            select(idx + 1)
        else:
            _update_list_highlight()
            _update_footer()

    def _accept() -> None:
        _set_status(STATUS_ACCEPT)

    def _deny() -> None:
        _set_status(STATUS_DENY)

    def _prev() -> None:
        select(current_idx["value"] - 1)

    def _next() -> None:
        select(current_idx["value"] + 1)

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
        with ui.row().classes("items-center gap-2"):
            ui.button("Export Accepted", icon="file_download", on_click=_export).props(
                "flat text-color=white"
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

                # Object list
                ui.label("Objects").classes(SECTION_HEADER)
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
# Main
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

    @ui.page("/")
    def index():
        build_page(dataset_dir, export_dir)

    ui.run(port=args.port, title="Mesh Reviewer", reload=False)


if __name__ == "__main__":
    main()
