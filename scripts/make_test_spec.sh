#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
python_executable="${PYTHON:-python}"

PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_executable}" - "${project_root}" <<'PY'
import os
import sys

from fem2d.input_source import generate_geo_with_topology

project_root = sys.argv[1]
geo_path = os.path.join(project_root, "models", "test_spec.geo")
output_path = os.path.join(project_root, "models", "test_spec.msh")
published, _import = generate_geo_with_topology(
    geo_path, output_path=output_path)
if published is None:
    raise SystemExit(
        "Gmsh is unavailable or mesh generation failed. Set GMSH_PATH to "
        "a working executable and retry.")
print(f"Regenerated {published}")
PY
