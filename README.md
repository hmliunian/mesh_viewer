# Mesh Viewer

A web-based GUI for reviewing `.geom.npz` mesh files. Built with [NiceGUI](https://nicegui.io/) and [trimesh](https://trimesh.org/).

## Features

- 3D preview of meshes with bounding box visualization
- Accept / Deny workflow with review state persistence
- Keyboard shortcuts: `A` accept, `D` deny, arrow keys to navigate
- Export accepted meshes with metadata manifest
- Configurable reviewer count and per-user pending-work quantities
- Admin dashboard with cumulative and current-batch progress

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- [just](https://just.systems/) (optional, for convenience commands)

## Setup

```bash
just install
```

Or manually:

```bash
uv sync
```

## Usage

```bash
just run
```

Or directly:

```bash
uv run python mesh_reviewer.py
```

### CLI Options

| Flag             | Default              | Description                              |
|------------------|----------------------|------------------------------------------|
| `--dataset-dir`  | `./input`            | Directory containing `.geom.npz` files   |
| `--export-dir`   | `./reviewed_export`  | Directory to export accepted meshes      |
| `--port`         | `8090`               | Port for the web server                  |

The dataset directory is scanned for `.geom.npz` files only.

## Convert meshes to `.geom.npz`

Convert every supported mesh in a directory (`.glb`, `.gltf`, `.obj`, `.off`,
`.ply`, or `.stl`):

```bash
uv run python mesh_to_geom.py ./input
```

The command also accepts a single mesh file. Existing outputs are skipped; pass
`--overwrite` to replace them. The viewer interprets coordinates as metres, so
use `--scale 0.001` when source coordinates are in millimetres.

Each output is named `<mesh-name>.geom.npz` and contains:

| Key              | Shape    | Description                                |
|------------------|----------|--------------------------------------------|
| `vertices`       | `(N, 3)` | Vertex positions (`float32`)               |
| `faces`          | `(M, 3)` | Triangle vertex indices (`int32`)          |
| `vertex_normals` | `(N, 3)` | Per-vertex normals (`float32`)             |
| `bounds_min`     | `(3,)`   | Axis-aligned bounding-box minimum          |
| `bounds_max`     | `(3,)`   | Axis-aligned bounding-box maximum          |
| `volume`         | scalar   | Enclosed volume, or `NaN` if not reliable |

## Accounts

The application has one admin and 20 available reviewer accounts:

- `admin`
- `user01` through `user20`

The default admin password is `qiuzhi2026`. Reviewer passwords are the same as
their usernames. Configure passwords before exposing the server outside a trusted
network:

```bash
export MESH_REVIEWER_ADMIN_PASSWORD='replace-admin-password'
export MESH_REVIEWER_USER_PASSWORD='shared-reviewer-password'
```

An individual reviewer password can override the shared password:

```bash
export MESH_REVIEWER_USER01_PASSWORD='user01-password'
```

Initial assignments are created evenly on first startup and stored under
`<dataset-dir>/.mesh_reviewer/assignments.json`. On the admin page, the Assignment
panel can:

- select between 1 and 20 active users (`user01` through `userNN`);
- edit the number of unreviewed files assigned to each active user;
- restore the default even split with **Equal Split**; and
- consolidate completed work and create a fresh batch with
  **Refresh & Redistribute**.

Custom quantities may leave files in the unassigned pool, and those files remain
there until the next redistribution. Reviewer pages automatically reload when a
new assignment batch is published.

Current per-user state is stored in `<dataset-dir>/.mesh_reviewer/reviews/`.
Consolidated, deduplicated results are stored in
`<dataset-dir>/.mesh_reviewer/reviewed.json` and are never assigned again.

After login, reviewers see only their assigned files while retaining the existing
Accept, Deny, filtering, navigation, and export workflow. The admin account is
redirected to `/admin`, which displays cumulative and current-batch progress and
exports accepted meshes from both consolidated and in-progress results.

## Keyboard Shortcuts


| Key   | Action          |
|-------|-----------------|
| `A`   | Accept mesh     |
| `D`   | Deny mesh       |
| `<-`  | Previous mesh   |
| `->`  | Next mesh       |
