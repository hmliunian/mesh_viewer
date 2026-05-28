# Mesh Viewer

A web-based GUI for reviewing `.geom.npz` mesh files. Built with [NiceGUI](https://nicegui.io/) and [trimesh](https://trimesh.org/).

## Features

- 3D preview of meshes with bounding box visualization
- Accept / Deny workflow with review state persistence
- Keyboard shortcuts: `A` accept, `D` deny, arrow keys to navigate
- Export accepted meshes with metadata manifest

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
just run --dataset-dir /path/to/geom_npz_files --export-dir /tmp/reviewed_export
```

Or directly:

```bash
uv run python mesh_reviewer.py --dataset-dir /path/to/geom_npz_files
```

### CLI Options

| Flag             | Default              | Description                              |
|------------------|----------------------|------------------------------------------|
| `--dataset-dir`  | *(required)*         | Directory containing `.geom.npz` files   |
| `--export-dir`   | `./reviewed_export`  | Directory to export accepted meshes      |
| `--port`         | `8090`               | Port for the web server                  |

## Keyboard Shortcuts

| Key   | Action          |
|-------|-----------------|
| `A`   | Accept mesh     |
| `D`   | Deny mesh       |
| `<-`  | Previous mesh   |
| `->`  | Next mesh       |
