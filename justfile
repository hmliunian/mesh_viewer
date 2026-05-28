# Mesh Viewer project commands

# Install dependencies via uv
install:
    uv sync

# Run the mesh reviewer
run *ARGS:
    uv run python mesh_reviewer.py {{ARGS}}

# Run with example arguments
review dataset_dir export_dir="./reviewed_export" port="8090":
    uv run python mesh_reviewer.py --dataset-dir {{dataset_dir}} --export-dir {{export_dir}} --port {{port}}
