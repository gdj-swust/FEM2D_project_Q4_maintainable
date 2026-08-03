"""Tests for source-safe and topology-checked Gmsh generation."""
import os
import subprocess
import tempfile
from unittest.mock import patch

from scripts.geo_spec import generate_geo
from scripts.gmsh_runner import (
    build_gmsh_command,
    run_gmsh,
)


def _write(path, content):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(content)


def test_quad_command_is_deterministic_and_source_safe():
    with tempfile.TemporaryDirectory() as folder:
        geo = os.path.join(folder, "plate.geo")
        msh = os.path.join(folder, "plate.msh")
        source = "lc = 0.1;\nMesh 2;\n"
        _write(geo, source)

        command = build_gmsh_command("gmsh", geo, msh, quad=True)

        assert command[:9] == [
            "gmsh", "-v", "2",
            "-setnumber", "Mesh.RecombineAll", "1",
            "-setnumber", "Mesh.Algorithm", "8",
        ]
        assert command[-4:] == ["-o", os.path.abspath(msh), "-format", "msh"]
        with open(geo, "r", encoding="utf-8") as stream:
            assert stream.read() == source


def test_defer_publish_failure_cleans_temporary_files():
    """defer_publish=True 且 Gmsh 写出部分 .msh 后非零退出 — 临时文件必须清理.

    评审失败注入场景: 曾因 finally 只在非 defer 模式清理, 部分写入的
    .fem2d-gmsh-*.msh 会残留.
    """
    with tempfile.TemporaryDirectory() as folder:
        geo = os.path.join(folder, "plate.geo")
        output = os.path.join(folder, "plate.msh")
        _write(geo, "Mesh 2;\n")
        _write(output, "known-good mesh\n")

        def fake_run(command, **kwargs):
            del kwargs
            temporary_output = command[command.index("-o") + 1]
            _write(temporary_output, "$MeshFormat\n4.1 0 8\n")  # 部分写出
            return subprocess.CompletedProcess(command, 1, "", "boom")

        with patch("scripts.gmsh_runner.subprocess.run", side_effect=fake_run):
            published = run_gmsh(
                geo, quad=True, output_path=output, gmsh_exe="gmsh",
                defer_publish=True)

        assert published is None
        # 旧网格保留
        with open(output, "r", encoding="utf-8") as stream:
            assert stream.read() == "known-good mesh\n"
        # 临时文件全部清理 (曾残留)
        leftovers = [
            name for name in os.listdir(folder)
            if name.startswith(".fem2d-gmsh-")
        ]
        assert leftovers == [], f"临时文件残留: {leftovers}"


def test_generated_quad_geo_places_options_before_meshing():
    with tempfile.TemporaryDirectory() as folder:
        geo = os.path.join(folder, "plate.geo")
        spec = {
            "type": "rect",
            "params": {"width": 2.0, "height": 1.0},
            "holes": [],
            "mesh_size": 0.25,
            "boundaries": [],
            "body_force": None,
        }
        generate_geo(spec, geo, quad=True)
        with open(geo, "r", encoding="utf-8") as stream:
            content = stream.read()

        assert content.index("Mesh.RecombineAll = 1;") < content.index(
            "Mesh 2;")
        assert "\nSave " not in content


def test_run_gmsh_publishes_valid_mesh_atomically():
    with tempfile.TemporaryDirectory() as folder:
        geo = os.path.join(folder, "plate.geo")
        output = os.path.join(folder, "plate.msh")
        source = 'lc = 0.1;\nMesh 2;\nSave "plate.msh";\n'
        _write(geo, source)
        _write(output, "old mesh\n")

        def fake_run(command, **kwargs):
            del kwargs
            geometry = next(
                item for item in command if item.endswith(".geo"))
            with open(geometry, "r", encoding="utf-8") as stream:
                prepared_source = stream.read()
                assert "\nSave " not in prepared_source
                assert prepared_source.rstrip().endswith(
                    "Mesh.SaveAll = 1;")
            temporary_output = command[command.index("-o") + 1]
            _write(temporary_output, "$MeshFormat\n4.1 0 8\n")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("scripts.gmsh_runner.subprocess.run", side_effect=fake_run):
            published = run_gmsh(
                geo, quad=True, output_path=output, gmsh_exe="gmsh")

        assert published == os.path.abspath(output)
        with open(output, "r", encoding="utf-8") as stream:
            assert stream.read().startswith("$MeshFormat")
        with open(geo, "r", encoding="utf-8") as stream:
            assert stream.read() == source
        assert not any(
            name.startswith(".fem2d-gmsh-") for name in os.listdir(folder))


def test_run_gmsh_preserves_previous_mesh_after_failure():
    """gmsh 失败 (非零退出) 时保留旧 .msh — 拓扑验证已移至主进程 import_msh."""
    with tempfile.TemporaryDirectory() as folder:
        geo = os.path.join(folder, "plate.geo")
        output = os.path.join(folder, "plate.msh")
        _write(geo, "Mesh 2;\n")
        _write(output, "known-good mesh\n")

        def fake_run(command, **kwargs):
            del kwargs
            return subprocess.CompletedProcess(command, 1, "", "boom")

        with patch("scripts.gmsh_runner.subprocess.run", side_effect=fake_run):
            published = run_gmsh(
                geo, quad=True, output_path=output, gmsh_exe="gmsh")

        assert published is None
        with open(output, "r", encoding="utf-8") as stream:
            assert stream.read() == "known-good mesh\n"
