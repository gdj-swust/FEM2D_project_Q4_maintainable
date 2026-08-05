"""覆盖轮 C1 — element/{q4,q4r,q4i,cst}.py 缺口行 (内核冻结区: 只测不改).

全部为防御分支 (奇异/负 Jacobian 拒绝、缓存失效重算、验证失败路径),
用退化几何或材料指纹变更触发 — 不触碰任何冻结公式的数值结果.
"""
import numpy as np
import pytest

from fem2d.element.q4 import B_matrix, element_stiffness as q4_stiffness
from fem2d.element.q4 import Q4Element
from fem2d.element.q4r import element_stiffness as q4r_stiffness
from fem2d.element.q4r import Q4RElement
from fem2d.element.q4i import element_operators, element_stiffness as q4i_stiffness
from fem2d.element.q4i import Q4IElement
from fem2d.element.cst import CSTElement
from fem2d.mesh import Mesh

_U = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])   # CCW 单位方
_R = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])   # 反向 (CW)
_C = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])   # 共线退化


def _mesh(coords=_U, elem_type="CPS4"):
    m = Mesh(nodes=coords, elements=np.array([[0, 1, 2, 3]]),
             E=1e6, nu=0.3, thickness=1.0, elem_type=elem_type)
    m.build_connectivity()      # 按 element_kernel 填充几何缓存 (q4_*/q4r_*/q4i_*)
    return m


# ── Q4: 奇异/负 Jacobian 拒绝 ───────────────────────────────────────────────

def test_q4_b_matrix_singular_rejected():
    """共线四边 → detJ≈0 → ValueError (line 52)."""
    with pytest.raises(ValueError, match="singular"):
        B_matrix(_C, 0.0, 0.0)


def test_q4_element_stiffness_negative_jacobian():
    """反向单元 → 负 detJ → ValueError (line 74)."""
    with pytest.raises(ValueError, match="non-positive"):
        q4_stiffness(_R, 1e6, 0.3)


def test_q4_stiffness_batch_negative_jacobian():
    """批量路径 Gauss 点负 detJ → ValueError (line 214)."""
    mesh = _mesh(_R)
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        Q4Element().stiffness_batch(mesh)


def test_q4_shape_values_at_outside_returns_none():
    """域外远点 → 牛顿迭代后残差超限 → None (line 320)."""
    assert Q4Element().shape_values_at(_U, 100.0, 100.0) is None


def test_q4_verify_mesh_bad_jacobian_fails(capsys):
    """反向单元网格 → Jacobian 检查失败 → verify_mesh False."""
    mesh = _mesh(_R)
    assert Q4Element().verify_mesh(mesh, verbose=True) is False
    assert "[FAIL] Q4 Jacobian check" in capsys.readouterr().out


# ── Q4R: 形状校验 / 奇异投影 / 缓存失效 / 负能量告警 ───────────────────────

def test_q4r_element_stiffness_bad_shape():
    """坐标形状 ≠ (4,2) → ValueError (line 109)."""
    with pytest.raises(ValueError, match="shape \\(4, 2\\)"):
        q4r_stiffness(np.zeros((3, 2)), 1e6, 0.3)


def test_q4r_affine_complement_singular():
    """共线四边 → 仿射补空间零模 → ValueError (line 85)."""
    from fem2d.element.q4r import _affine_complement
    with pytest.raises(ValueError, match="singular"):
        _affine_complement(_C[None, :, :])


def test_q4r_element_stiffness_negative_jacobian():
    """反向单元 → 负 detJ → ValueError (line 118)."""
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        q4r_stiffness(_R, 1e6, 0.3)


def test_q4r_hourglass_cache_recompute_on_material_change():
    """材料指纹变更后 hourglass_energy → 按当前材料重算 (line 252)."""
    k = Q4RElement()
    mesh = _mesh(elem_type="CPS4R")
    k.stiffness_batch(mesh)                     # 建立缓存
    mesh.E = 2e6                                 # 变更材料 → 指纹不匹配
    energy = k.hourglass_energy(mesh, np.zeros((1, 8)))
    assert energy.shape == (1,) and np.isfinite(energy).all()
    assert energy[0] == 0.0                      # 零位移 → 零稳定能


def test_q4r_hourglass_negative_energy_warns():
    """负沙漏系数 (绕过校验直接赋值) → RuntimeWarning (line 267)."""
    k = Q4RElement()
    mesh = _mesh(elem_type="CPS4R")
    k.stiffness_batch(mesh)                      # 缓存正系数稳定项
    k.hourglass_coefficient = -1.0               # 绕过 validate 注入负系数
    u = np.array([[1.0, 0.5, -1.0, 0.5, 0.0, 0.5, 0.5, -0.5]])
    with pytest.warns(RuntimeWarning, match="显著为负"):
        k.hourglass_energy(mesh, u)


def test_q4r_verify_mesh_bad_jacobian_fails(capsys):
    mesh = _mesh(_R, elem_type="CPS4R")
    assert Q4RElement().verify_mesh(mesh, verbose=True) is False
    assert "[FAIL] Q4R Jacobian check" in capsys.readouterr().out


# ── Q4I: 奇异/负 Jacobian 拒绝 ──────────────────────────────────────────────

def test_q4i_centre_jacobian_singular():
    """共线四边 → 中心 Jacobian 奇异 → ValueError (line 90)."""
    with pytest.raises(ValueError, match="centre Jacobian is singular"):
        element_operators(_C)


def test_q4i_element_stiffness_negative_jacobian():
    """反向单元 → 负 detJ → ValueError (line 211)."""
    with pytest.raises(ValueError, match="non-positive"):
        q4i_stiffness(_R, 1e6, 0.3)


def test_q4i_stiffness_batch_negative_jacobian():
    """批量路径 Gauss 点负 detJ → ValueError (line 188)."""
    mesh = _mesh(_R, elem_type="CPS4I")
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        Q4IElement().stiffness_batch(mesh)


def test_q4i_verify_mesh_bad_jacobian_fails(capsys):
    mesh = _mesh(_R, elem_type="CPS4I")
    assert Q4IElement().verify_mesh(mesh, verbose=True) is False
    assert "[FAIL] Q4I Jacobian check" in capsys.readouterr().out


# ── CST: 逐单元体力积分 / 退化 shape_values_at ──────────────────────────────

def test_cst_body_force_vector_integral():
    """Hammer 积分累加 → 有限且与解析一致 (lines 320-322)."""
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]),
             E=1e6, nu=0.3, thickness=1.0)
    m.build_connectivity()
    fe = CSTElement().body_force_vector(m, 0, (0.0, -1.0))
    assert fe.shape == (6,) and np.all(np.isfinite(fe))
    # 常量体力: f_e = t·A·(bx,by)/3 分布到每个节点
    assert fe[1::2] == pytest.approx(-1.0 * 0.5 / 3.0)


def test_cst_shape_values_at_degenerate_returns_none():
    """反向 (CW) 三角形 → detJ≤0 → None (line 344)."""
    cw = np.array([[0., 0.], [0., 1.], [1., 0.]])
    assert CSTElement().shape_values_at(cw, 0.2, 0.2) is None
