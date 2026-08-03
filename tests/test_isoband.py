"""Isoband input validation tests"""
import numpy as np
from fem2d import Mesh, solve
from fem2d.visualize import plot_contour
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest


def _make_test_mesh():
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    elems = np.array([[0,1,2],[1,3,2]], dtype=int)
    return Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)


def test_isoband_rejects_non_increasing():
    mesh = _make_test_mesh()
    mesh.fix_node(0,'both'); mesh.fix_node(1,'both'); mesh.add_force(3,1e6,0)
    r = solve(mesh)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='strictly increasing'):
        plot_contour(mesh, r['vm_stress'], '', ax=ax, shading='isoband',
                    location='element', levels=[0, 2, 1, 3])
    plt.close('all')


def test_isoband_rejects_non_uniform():
    mesh = _make_test_mesh()
    mesh.fix_node(0,'both'); mesh.fix_node(1,'both'); mesh.add_force(3,1e6,0)
    r = solve(mesh)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='uniformly spaced'):
        plot_contour(mesh, r['vm_stress'], '', ax=ax, shading='isoband',
                    location='element', levels=[0, 1, 3, 6])
    plt.close('all')


def test_isoband_rejects_nan():
    mesh = _make_test_mesh()
    mesh.fix_node(0,'both'); mesh.fix_node(1,'both'); mesh.add_force(3,1e6,0)
    r = solve(mesh)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='NaN'):
        plot_contour(mesh, r['vm_stress'], '', ax=ax, shading='isoband',
                    location='element', levels=[0, np.nan, 2])
    plt.close('all')


def test_isoband_constant_stress_ok():
    mesh = _make_test_mesh()
    mesh.fix_node(0,'both'); mesh.fix_node(1,'both'); mesh.add_force(3,1e6,0)
    solve(mesh)
    fig, ax = plt.subplots()
    plot_contour(mesh, np.array([1.0, 1.0]), '', ax=ax, shading='isoband',
                location='element', n=12)
    plt.close('all')
