#!/usr/bin/env python3
"""Convert mesh files to the ``*.geom.npz`` format used by Mesh Viewer."""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".off", ".ply", ".stl"}


@dataclass(frozen=True)
class GeometryData:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    volume: np.ndarray


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh and bake all scene-node transforms into one mesh."""
    loaded = trimesh.load(path, force="scene", process=False)

    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"expected a mesh scene, got {type(loaded).__name__}")
    if not loaded.geometry:
        raise ValueError("the file contains no geometry")

    non_mesh = [
        name
        for name, geometry in loaded.geometry.items()
        if not isinstance(geometry, trimesh.Trimesh)
    ]
    if non_mesh:
        names = ", ".join(map(str, non_mesh[:3]))
        suffix = " ..." if len(non_mesh) > 3 else ""
        raise ValueError(f"the scene contains non-mesh geometry: {names}{suffix}")

    mesh = loaded.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"could not combine the scene into a mesh: {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("the mesh has no vertices or triangular faces")
    return mesh


def compute_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray, chunk_size: int = 250_000
) -> np.ndarray:
    """Compute area-weighted vertex normals without optional SciPy."""
    normals = np.zeros_like(vertices, dtype=np.float64)
    for start in range(0, len(faces), chunk_size):
        face_chunk = faces[start : start + chunk_size]
        face_normals = np.cross(
            vertices[face_chunk[:, 1]] - vertices[face_chunk[:, 0]],
            vertices[face_chunk[:, 2]] - vertices[face_chunk[:, 0]],
        )
        for corner in range(3):
            np.add.at(normals, face_chunk[:, corner], face_normals)

    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, np.newaxis]
    return normals


def make_geometry_data(mesh: trimesh.Trimesh, scale: float) -> GeometryData:
    """Create validated arrays matching the viewer's NPZ schema."""
    vertices64 = np.asarray(mesh.vertices, dtype=np.float64) * scale
    faces64 = np.asarray(mesh.faces, dtype=np.int64)

    if vertices64.ndim != 2 or vertices64.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices64.shape}")
    if faces64.ndim != 2 or faces64.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces64.shape}")
    if not np.isfinite(vertices64).all():
        raise ValueError("vertices contain NaN or infinite values")
    if faces64.min() < 0 or faces64.max() >= len(vertices64):
        raise ValueError("faces contain an out-of-range vertex index")
    if faces64.max() > np.iinfo(np.int32).max:
        raise ValueError("the mesh has too many vertices for int32 face indices")

    normals64 = compute_vertex_normals(vertices64, faces64)
    if not np.isfinite(normals64).all():
        raise ValueError("calculated vertex normals contain NaN or infinite values")

    # A signed volume from an open or inconsistently wound surface is not a
    # meaningful enclosed volume. The viewer treats NaN as "N/A".
    volume = np.nan
    if mesh.is_watertight and mesh.is_winding_consistent:
        candidate = abs(float(mesh.volume)) * scale**3
        if np.isfinite(candidate):
            volume = candidate

    return GeometryData(
        vertices=np.ascontiguousarray(vertices64, dtype=np.float32),
        faces=np.ascontiguousarray(faces64, dtype=np.int32),
        vertex_normals=np.ascontiguousarray(normals64, dtype=np.float32),
        bounds_min=np.asarray(vertices64.min(axis=0), dtype=np.float32),
        bounds_max=np.asarray(vertices64.max(axis=0), dtype=np.float32),
        volume=np.asarray(volume, dtype=np.float64),
    )


def save_geometry(path: Path, data: GeometryData) -> None:
    """Atomically write a compressed NPZ so interrupted runs leave no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                vertices=data.vertices,
                faces=data.faces,
                vertex_normals=data.vertex_normals,
                bounds_min=data.bounds_min,
                bounds_max=data.bounds_max,
                volume=data.volume,
            )
        temporary_path.replace(path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def discover_meshes(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise ValueError(f"input path does not exist: {source}")

    meshes = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not meshes:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"no supported mesh files found in {source} ({extensions})")
    return meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one mesh or a directory of meshes to *.geom.npz"
    )
    parser.add_argument("source", type=Path, help="mesh file or directory")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="output directory; defaults to the source directory",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply vertex coordinates by this value (default: 1.0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing *.geom.npz files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not np.isfinite(args.scale) or args.scale <= 0:
        print("error: --scale must be a finite number greater than zero", file=sys.stderr)
        return 2

    try:
        sources = discover_meshes(source)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = source if source.is_dir() else source.parent

    targets: dict[Path, Path] = {}
    for mesh_path in sources:
        target = output_dir / f"{mesh_path.stem}.geom.npz"
        if target in targets:
            print(
                f"error: {targets[target]} and {mesh_path} map to the same output {target}",
                file=sys.stderr,
            )
            return 2
        targets[target] = mesh_path

    converted = 0
    skipped = 0
    failed = 0
    for target, mesh_path in targets.items():
        if target.exists() and not args.overwrite:
            print(f"[skip] {target} already exists")
            skipped += 1
            continue

        try:
            mesh = load_mesh(mesh_path)
            data = make_geometry_data(mesh, args.scale)
            save_geometry(target, data)
        except Exception as error:
            print(f"[fail] {mesh_path}: {error}", file=sys.stderr)
            failed += 1
            continue

        volume_text = "N/A" if np.isnan(data.volume) else f"{float(data.volume):.6g}"
        print(
            f"[ok] {mesh_path.name} -> {target} "
            f"({len(data.vertices):,} vertices, {len(data.faces):,} faces, "
            f"volume={volume_text})"
        )
        converted += 1

    print(f"done: {converted} converted, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
