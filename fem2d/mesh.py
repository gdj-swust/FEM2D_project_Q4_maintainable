"""Mesh 数据结构 — Bathe §4.2.1: 有限元节点/单元/载荷数据容器

数据结构：
  nodes:     (n_nodes, 2) — 节点坐标 [x, y]
  elements:  (n_elem, nnode_e) — 单元节点索引 (边界按 CCW 排列)
  DOF 编号:  节点 i → [2i (x), 2i+1 (y)]  (Bathe Eq 4.18)
"""
import copy
import numbers
import warnings
from dataclasses import dataclass, field

import numpy as np

from .checks import (
    require_dof_index_array,
    require_finite_positive,
    require_finite_scalar,
)
# 载荷 schema (形状/分量校验) 唯一实现在 loads_schema.py — 契约清账阶段 3
# 从本模块纯搬移 (行为不变), bc_apply 等调用方直接引用该模块
from .loads_schema import _check_load_pair, _check_load_scalar
from .topology_core import (
    ElementLocator,
    build_edge_table,
    element_neighbor_table,
    node_element_table,
)


@dataclass(init=False)
class Mesh:
    """二维、同构单元网格

    Bathe §4.2.1: 位移有限元法的完整网格数据。

    Attributes
    ----------
    nodes : (n_nodes, 2) ndarray
        节点坐标 (只读数组; 构造后修改必须走
        :meth:`replace_nodes` — 直接重绑 ``mesh.nodes = new`` 也会
        自动路由到 replace_nodes, 保证邻接/几何缓存一致失效)
    elements : (n_elem, nnode_e) ndarray
        单元节点索引 (边界按 CCW 方向; 只读, 修改走
        :meth:`replace_elements`)
    elem_type : str
        已注册的单元类型或别名；默认 ``"CPS3"``。
    thickness : float
        厚度 [m] (平面应力/应变)
    E : float
        杨氏模量 [Pa]
    nu : float
        泊松比
    plane_type : str
        "stress" (平面应力) 或 "strain" (平面应变)
        Bathe Table 4.3 / §6.3.4
    fixed_dofs : ndarray
        受约束的 DOF 索引
    prescribed_vals : dict
        DOF索引 → 指定位移值
    body_force : tuple or callable or None
        (bx, by) [N/m³] 体力分量或函数 f(x,y)→(bx,by)
    surface_tractions : list
        面力边列表 [{"nodes":(ni,nj), "traction":(tx,ty)}, ...]
    concentrated_forces : list
        集中力列表 [{"node":nid, "force":(fx,fy)}, ...]

    Computed (__post_init__):
    -------------------------
    node_to_elems : list[list[int]]
        node_to_elems[i] = 包含节点 i 的所有单元索引
    elem_neighbors : list[list[int]]
        elem_neighbors[e] = 与单元 e 共享边的相邻单元索引 (最多3个)
    boundary_edges : list[tuple[int,int]]
        仅属于一个单元的边 (边界边)
    """
    # ── 私有几何存储: nodes/elements 由 property 暴露, setter 路由到
    # replace_nodes/replace_elements — 直接赋值也正确失效缓存。
    # 构造期由 __init__ 直接写 _nodes/_elements, __post_init__ 校验后
    # 再经 setter 锁定。
    _nodes: np.ndarray = field(init=False, repr=False)
    _elements: np.ndarray = field(init=False, repr=False)
    _fixed_dofs: np.ndarray = field(init=False, repr=False)
    _fixed_set: set | None = field(default=None, init=False, repr=False)
    _fixed_dirty: bool = field(default=False, init=False, repr=False)
    # 刚体模态检查结果缓存: (fixed_dofs 内容快照拷贝, 结果 list) | None —
    # 见 check_rigid_body_constraints 缓存注释 (BC 未变免重算拓扑)
    _rigid_cache: tuple | None = field(default=None, init=False, repr=False)
    thickness: float = 1.0
    E: float = 210e9
    nu: float = 0.3
    plane_type: str = "stress"
    prescribed_vals: dict = field(default_factory=dict)
    body_force: object = None   # tuple | callable | None
    surface_tractions: list = field(default_factory=list)
    concentrated_forces: list = field(default_factory=list)
    # elem_type 是只读 property (构造后赋值须重建 Mesh) — 不做 dataclass
    # 字段声明, 避免与 property 同名触发 F811

    # ── 预计算拓扑 (Bathe §4.3.6: 应力恢复需要邻接关系) ──
    node_to_elems: list = field(default=None, repr=False)
    elem_neighbors: list = field(default=None, repr=False)
    boundary_edges: list = field(default=None, repr=False)
    edge_to_elems: dict = field(default=None, repr=False)  # (a,b)→[elem_ids]
    internal_edge_data: np.ndarray = field(default=None, repr=False)
    locator: object = field(default=None, init=False, repr=False)
    areas: np.ndarray = field(default=None, repr=False)        # (n_elem,) 单元面积 (正值)
    signed_areas: np.ndarray = field(default=None, repr=False) # (n_elem,) 有向面积 (CCW>0)
    element_kernel: object = field(default=None, init=False, repr=False)

    def __init__(self, nodes, elements, thickness=1.0, E=210e9, nu=0.3,
                 plane_type="stress", fixed_dofs=None, prescribed_vals=None,
                 body_force=None, surface_tractions=None,
                 concentrated_forces=None, elem_type="CPS3"):
        """签名与 dataclass 自动 __init__ 完全兼容 (只读 property 需 init=False)."""
        self.thickness = thickness
        self.E = E
        self.nu = nu
        self.plane_type = plane_type
        self.fixed_dofs = (np.array([], dtype=int) if fixed_dofs is None
                           else fixed_dofs)
        self.prescribed_vals = ({} if prescribed_vals is None
                                else prescribed_vals)
        self.body_force = body_force
        self.surface_tractions = ([] if surface_tractions is None
                                  else surface_tractions)
        self.concentrated_forces = ([] if concentrated_forces is None
                                    else concentrated_forces)
        self._elem_type = elem_type
        # 缓存字段 (areas 等) 由 field(default=None) 类属性兜底, 无需赋 None
        _reject_complex_nodes(nodes, "nodes")
        self._nodes = np.asarray(nodes, dtype=float)
        self._elements = np.asarray(elements)
        self.__post_init__()

    # ── 几何属性: setter 保证缓存一致性 ──

    @property
    def elem_type(self) -> str:
        """单元类型名 (只读) — 构造后赋值曾只改字符串不改 element_kernel,
        显示 CPS4I/CPS4R 实际仍按原单元计算; 换类型必须重建 Mesh."""
        return self._elem_type

    @elem_type.setter
    def elem_type(self, value):
        raise AttributeError(
            "elem_type is read-only after construction — rebuild the Mesh "
            f"with elem_type={value!r} to change element type")

    @property
    def nodes(self) -> np.ndarray:
        """节点坐标 (n_nodes, 2) — 只读数组, 修改走 replace_nodes."""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        # 构造后任何赋值路由到 replace_nodes (校验/只读/失效缓存);
        # 构造期由 __init__ 直接写 _nodes, __post_init__ 经本 setter 锁定
        self.replace_nodes(value)

    @property
    def elements(self) -> np.ndarray:
        """单元节点索引 (n_elem, nnode_e) — 只读数组, 修改走 replace_elements."""
        return self._elements

    @elements.setter
    def elements(self, value):
        self.replace_elements(value)

    @property
    def fixed_dofs(self) -> np.ndarray:
        """受约束 DOF 索引 — 只读数组, 修改走 fix_node/fix_nodes_func.

        延迟落盘: fix_node 只更新内部 set (O(1)), 数组在首次读取时
        一次性重建 (O(n log n)) — 整边固支 (bc_apply 逐节点 fix_node)
        曾每次全量 set + sorted 重建 (O(n² log n), 10 万节点 ≈ 10¹⁰
        次操作)。内容恒为排序去重 int64, 与历史 np.unique 语义一致。
        """
        if self._fixed_dirty:
            assert self._fixed_set is not None  # dirty 仅由 fix_node 置位, 彼时 set 已建
            arr = np.array(sorted(self._fixed_set), dtype=int)
            arr.setflags(write=False)
            self._fixed_dofs = arr
            self._fixed_dirty = False
        return self._fixed_dofs

    @fixed_dofs.setter
    def fixed_dofs(self, value):
        # __post_init__ 校验后经本 setter 写入; fix_node 不走 setter
        # (直接改 set + 标记 dirty — 见 fix_node 延迟落盘注释)
        self._fixed_dofs = value
        self._fixed_set = None      # 惰性: 首次 fix_node 时从数组重建
        self._fixed_dirty = False
        # BC 重赋值 → 刚体模态检查结果失效。只清 _rigid_cache, 不调
        # invalidate_cache: 连接/几何缓存与 BC 无关, 全清会让参数研究
        # 中改 BC 后白白重建全部预处理 (fix_node 不经 setter 写 set +
        # dirty, 由检查时的内容快照比较兜底 — 见 check_rigid_body_constraints)
        self._rigid_cache = None

    def __post_init__(self):
        """初始化后不自动计算拓扑 — 延迟到首次访问时 (lazy evaluation)"""
        self._connectivity_built = False

        # Import here to keep Mesh lightweight and avoid an import cycle through
        # fem2d.__init__.  Importing element registers the built-in kernels.
        from .element import get_element_kernel
        # 浅拷贝: 注册表缓存单例 kernel, 而 Q4R 的 hourglass_coefficient 等
        # 是可变类属性 — 共享单例会令一个 Mesh 的修改污染进程内所有同型
        # 网格 (批量/多模型脚本中静默改变刚度)。
        self.element_kernel = copy.copy(get_element_kernel(self.elem_type))
        self._elem_type = str(self._elem_type).strip().upper()

        # ── 类型转换 & 基本校验 (先于所有检查) ──
        nodes = np.asarray(self.nodes, dtype=float)
        elems_raw = np.asarray(self.elements)

        # 标量/非数组输入会裸 IndexError (0 维数组 shape[0] 越界) — 先验维度
        if nodes.ndim == 0:
            raise ValueError(
                f"nodes must be a 2-D array, got scalar {self.nodes!r}")
        if elems_raw.ndim == 0:
            raise ValueError(
                f"elements must be a 2-D array, got scalar {self.elements!r}")
        if nodes.shape[0] == 0:
            raise ValueError("Mesh must contain at least one node")
        if elems_raw.shape[0] == 0:
            raise ValueError("Mesh must contain at least one element")
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"nodes must be (n_nodes, 2), got {nodes.shape}")
        expected_npe = self.element_kernel.nodes_per_element
        if elems_raw.ndim != 2 or elems_raw.shape[1] != expected_npe:
            raise ValueError(
                f"{self.elem_type} elements must be "
                f"(n_elem, {expected_npe}), got {elems_raw.shape}")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("nodes contain NaN or Inf")
        # 布尔单元索引: True/False 会被 rint 静默转 1/0, 构造出重复节点
        # 退化单元 — 与 _validate_node_id (mesh.py) / require_dof_index_array
        # (checks.py) 的 bool 拒绝策略同族, 此处补漏网
        if elems_raw.dtype.kind == "b":
            raise ValueError(
                "Element node indices must be integers — boolean arrays "
                "are rejected (True/False silently become 1/0)")
        if elems_raw.dtype.kind not in ("i", "u", "f"):
            # str/object 会在 np.isfinite 冒裸 TypeError — 非数值 dtype
            # 带参数名拒绝 (与 require_finite_scalar 模式对齐)
            raise TypeError(
                f"elements must be a numeric array of node indices, "
                f"got dtype {elems_raw.dtype}")
        if not np.all(np.isfinite(elems_raw)):
            raise ValueError("elements contain NaN or Inf")
        # 拒绝非整数节点索引 (浮点索引会被静默截断, 非常危险)
        if not np.issubdtype(elems_raw.dtype, np.integer):
            bad = elems_raw != np.rint(elems_raw)
            if np.any(bad):
                raise ValueError(
                    "Element node indices must be integers — "
                    f"non-integer value found: {elems_raw[bad].flat[0]}. "
                    "Floating-point indices can silently change mesh topology.")
            elems_raw = np.rint(elems_raw)

        elements = elems_raw.astype(np.int64, copy=False)
        if np.any(elements < 0):
            raise ValueError("Element node indices must be ≥ 0")
        if np.any(elements >= nodes.shape[0]):
            bad = elements[(elements < 0) | (elements >= nodes.shape[0])]
            raise ValueError(
                f"Element node indices out of bounds [0, {nodes.shape[0]-1}]: "
                f"{np.unique(bad).tolist()}")

        # ── 固定自由度校验 (共享 DOF helper: 布尔掩码 TypeError / 其余 ValueError) ──
        n_dof = 2 * nodes.shape[0]
        fixed = require_dof_index_array(
            self.fixed_dofs, "fixed_dofs", n_dof=n_dof)
        fixed = np.unique(fixed)
        # 校验 prescribed_vals 的键都在 fixed_dofs 中
        if not isinstance(self.prescribed_vals, dict):
            raise ValueError(
                f"prescribed_vals must be a dict mapping DOF -> value, "
                f"got {type(self.prescribed_vals).__name__}")
        extra_keys = set(self.prescribed_vals.keys()) - set(fixed.tolist())
        if extra_keys:
            raise ValueError(
                f"prescribed_vals keys {extra_keys} are not in fixed_dofs")
        self.fixed_dofs = fixed
        self.fixed_dofs.setflags(write=False)

        # ── 材料参数校验 (共享标量 helper: 非数值类型 → TypeError) ──
        require_finite_scalar(self.E, "E")
        require_finite_scalar(self.nu, "nu")
        require_finite_positive(self.thickness, "thickness")

        # ── 锁定 & 存储 ──
        # 经 property setter 写入: 校验 → 复制 → 只读 → 缓存失效。
        # (__init__ 已写入原始值, 此处完成最终锁定; _nodes/_elements
        # 只读后即使绕过 API 直接写私有字段也会被 numpy 拒绝。)
        self.nodes = nodes.copy()
        self.elements = elements.copy()

        # 重复单元检测 (排序后的连接数组相同 = 重复)
        sorted_conn = np.sort(self.elements, axis=1)
        vals, counts = np.unique(sorted_conn, axis=0, return_counts=True)
        if np.any(counts > 1):
            dup = vals[np.where(counts > 1)[0][0]].tolist()
            raise ValueError(
                f"Duplicate element: nodes {dup}. "
                f"Remove duplicate elements before solving."
            )

    def invalidate_cache(self):
        """清除所有惰性缓存 — 在通过 replace_nodes/replace_elements
        修改网格后由这些方法调用.

        ``nodes``/``elements`` 是只读数组 (原地写入会抛 ValueError),
        property setter 把任何重绑赋值路由到 :meth:`replace_nodes` /
        :meth:`replace_elements` — 它们内部完成校验、只读锁定并清除
        缓存。直接绕过 API 改数组会导致面积/形函数系数/邻接关系
        静默过期, 产生错误结果。

        同时清除刚体模态检查结果缓存 — 节点坐标/单元连接直接决定
        连通分量分解与 R 约束矩阵, 网格变更后旧结果必然过期。
        """
        self._connectivity_built = False
        self.node_to_elems = None
        self.elem_neighbors = None
        self.boundary_edges = None
        self.edge_to_elems = None
        self.internal_edge_data = None
        self.locator = None
        for name in getattr(self, "_kernel_cache_names", ()):
            setattr(self, name, None)
        self._kernel_cache_names = ()
        self.areas = None
        self.signed_areas = None
        self.b_coeffs = None
        self.c_coeffs = None
        self.element_dofs = None
        self.centroids = None
        # 内核材料/几何指纹缓存 (Q4R 沙漏 _q4r_*, Q4I 缩聚 _q4i_*) —
        # 几何变更后必须失效, 否则 hourglass_energy / enhanced_amplitudes
        # 会用旧坐标算出的缓存 (B4 同类问题)。
        self._q4r_hourglass_material = None
        self._q4i_enhancement = None
        self._q4i_enhancement_material = None
        # 刚体模态检查结果缓存 (键含网格拓扑) — 网格变更必须失效
        self._rigid_cache = None

    def replace_nodes(self, new_nodes):
        """原子替换节点坐标 — 显式修改网格的唯一正规途径.

        ``nodes``/``elements`` 是只读数组 (直接写会抛 ValueError);
        需要改坐标时调用本方法 (或直接赋值 ``mesh.nodes = new`` —
        property setter 自动路由到这里): 校验新数组 (形状/有限性),
        重建节点数组 (保持只读), 并清除全部惰性缓存。单元连接不变。
        """
        _reject_complex_nodes(new_nodes, "replace_nodes: new_nodes")
        nodes = np.asarray(new_nodes, dtype=float)
        if nodes.shape != self._nodes.shape:
            raise ValueError(
                f"replace_nodes: 形状必须为 {self._nodes.shape} "
                f"(保持节点数), 得到 {nodes.shape}")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("replace_nodes: 坐标包含 NaN/Inf")
        self._nodes = nodes.copy()
        self._nodes.setflags(write=False)
        self.invalidate_cache()

    def replace_elements(self, new_elements):
        """原子替换单元连接 — 显式修改网格的唯一正规途径.

        校验与 ``__post_init__`` 相同的约束 (非空/整数索引/边界/单元
        节点数/重复单元), 重建单元数组 (保持只读) 并清除全部惰性缓存。
        节点坐标不变; 拓扑变化后既有 BC/载荷按新编号解释, 调用方自行
        负责一致性.
        """
        elems_raw = np.asarray(new_elements)
        expected_npe = self.element_kernel.nodes_per_element
        if elems_raw.ndim != 2 or elems_raw.shape[1] != expected_npe:
            raise ValueError(
                f"replace_elements: 需要 (n_elem, {expected_npe}), "
                f"得到 {elems_raw.shape}")
        if elems_raw.shape[0] == 0:
            raise ValueError("replace_elements: 单元集不能为空")
        # 与 __post_init__ 同族守卫: 布尔掩码会被 rint 静默转 1/0,
        # 构造出重复节点退化单元; str/object 在 ufunc 冒裸 TypeError
        if elems_raw.dtype.kind == "b":
            raise ValueError(
                "replace_elements: 单元节点索引必须是整数 — "
                "布尔数组被拒绝 (True/False 会静默转 1/0)")
        if elems_raw.dtype.kind not in ("i", "u", "f"):
            raise TypeError(
                f"replace_elements: 单元节点索引必须是数值数组, "
                f"got dtype {elems_raw.dtype}")
        if not np.issubdtype(elems_raw.dtype, np.integer):
            bad = elems_raw != np.rint(elems_raw)
            if np.any(bad):
                raise ValueError(
                    "replace_elements: 单元节点索引必须是整数 — "
                    f"非整数值: {elems_raw[bad].flat[0]}")
            elems_raw = np.rint(elems_raw)
        elements = elems_raw.astype(np.int64, copy=False)
        if np.any((elements < 0) | (elements >= self._nodes.shape[0])):
            raise ValueError("replace_elements: 节点索引越界")
        # 重复单元检测 (与 __post_init__ 一致 — 重复单元会使刚度和体力
        # 重复计算, 静默改变结果)
        sorted_conn = np.sort(elements, axis=1)
        vals, counts = np.unique(sorted_conn, axis=0, return_counts=True)
        if np.any(counts > 1):
            dup = vals[np.where(counts > 1)[0][0]].tolist()
            raise ValueError(
                f"replace_elements: 重复单元: nodes {dup}. "
                f"Remove duplicate elements before solving.")
        self._elements = elements.copy()
        self._elements.setflags(write=False)
        self.invalidate_cache()

    def validate_state(self):
        """求解/装配入口的完整状态校验.

        构造后字段可被重写 (如 ``mesh.thickness = -1.0`` 会静默返回
        负位移, 残差却接近机器精度), 求解前重新检查: 材料参数、
        平面类型、节点/单元、BC/载荷合法性、几何缓存一致性。

        返回 ``self`` 便于链式调用。
        """
        # 单元类型与 kernel 一致性: elem_type 只读后, 任何分叉都意味着
        # 外部直接破坏 kernel — 求解前拒绝, 防止"显示 CPS4I 实际按 Q4 算"
        if not self.element_kernel.matches(self._elem_type):
            raise ValueError(
                f"element_kernel ({self.element_kernel.name}) 与 elem_type "
                f"({self._elem_type}) 不一致 — 单元类型构造后不可修改, "
                f"请重建 Mesh")
        self._validate_material_and_mesh()
        self._validate_bc_state()
        self._validate_loads_state()
        # 几何缓存一致性: 缺失/被破坏时强制重建 (双保险 — 正常路径下
        # property setter 已保证一致, 这里兜底外部直接破坏缓存的情况;
        # element_dofs 是 build_connectivity 的动态属性, 用 getattr 查)
        if self.areas is None or self.centroids is None \
                or getattr(self, "element_dofs", None) is None:
            self.invalidate_cache()
            self.build_connectivity()
        return self

    def _validate_material_and_mesh(self):
        # 共享 helper 负责类型 + NaN/Inf; 消息保持历史格式 (既有测试锁定)
        require_finite_scalar(self.E, "E")
        require_finite_scalar(self.nu, "nu")
        require_finite_scalar(self.thickness, "thickness")
        if self.E <= 0.0:
            raise ValueError(
                f"E = {self.E} — must be finite and > 0")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                f"nu = {self.nu} — must be in (-1, 0.5) "
                f"for isotropic material stability")
        if self.thickness <= 0.0:
            raise ValueError(
                f"thickness = {self.thickness} — must be finite and > 0")
        if self.plane_type not in ("stress", "strain"):
            raise ValueError(
                f"plane_type = {self.plane_type!r} — must be 'stress' or 'strain'")
        if self._nodes.shape[0] == 0 or self._elements.shape[0] == 0:
            raise ValueError("Mesh must contain at least one node and one element")
        if not np.all(np.isfinite(self._nodes)):
            raise ValueError("nodes contain NaN or Inf")
        if not np.all(np.isfinite(self._elements)):
            raise ValueError("elements contain NaN or Inf")

    def _validate_bc_state(self):
        # ── BC 状态 (构造后可被重写; 共享 DOF helper) ──
        n_dof = 2 * self._nodes.shape[0]
        fixed = require_dof_index_array(
            self.fixed_dofs, "fixed_dofs", n_dof=n_dof)
        _u, _c = np.unique(fixed, return_counts=True)
        dup = _u[_c > 1]
        if len(dup):
            # 重复约束: 罚函数 RHS 重复累加 → 静默错误位移
            raise ValueError(
                f"fixed_dofs 含重复 DOF: {dup.tolist()} — "
                "同一 DOF 只能约束一次")
        fixed_set = set(fixed.tolist())
        if not isinstance(self.prescribed_vals, dict):
            raise ValueError(
                f"prescribed_vals must be a dict mapping DOF -> value, "
                f"got {type(self.prescribed_vals).__name__}")
        extra_keys = set(self.prescribed_vals.keys()) - fixed_set
        if extra_keys:
            raise ValueError(
                f"prescribed_vals keys {extra_keys} are not in fixed_dofs")
        for d, v in self.prescribed_vals.items():
            require_finite_scalar(v, f"prescribed_vals[{d}]")

    def _validate_loads_state(self):
        # 载荷 schema 校验 (P2-4): 形状错误在求解前响亮失败 — 否则
        # 多余分量静默忽略 / 单分量裸 IndexError/TypeError。
        for i, cf in enumerate(self.concentrated_forces):
            if not isinstance(cf, dict):
                raise ValueError(
                    f"concentrated_forces[{i}] must be a dict, "
                    f"got {type(cf).__name__}: {cf!r}")
            missing = {"node", "force"} - set(cf)
            if missing:
                raise ValueError(
                    f"concentrated_forces[{i}] is missing key(s) "
                    f"{sorted(missing)} — full record: {cf!r}")
            # 构造函数直传的载荷不走 add_* API — 节点号校验收敛到
            # _validate_node_id (整数值浮点 2.0 规范化**写回** — 只转
            # 局部变量验证的话, 原始记录仍是 2.0, 组装时 IndexError)
            nid = self._validate_node_id(cf["node"])
            if not (0 <= nid < self._nodes.shape[0]):
                raise ValueError(
                    f"concentrated force node {nid} out of range "
                    f"[0, {self._nodes.shape[0]-1}]")
            cf["node"] = nid   # 规范化写回 (整数)
            _check_load_pair(
                cf["force"], f"concentrated_forces[{i}]['force']",
                allow_callable=False)
        for i, st in enumerate(self.surface_tractions):
            if not isinstance(st, dict):
                raise ValueError(
                    f"surface_tractions[{i}] must be a dict, "
                    f"got {type(st).__name__}: {st!r}")
            missing = {"nodes", "traction"} - set(st)
            if missing:
                raise ValueError(
                    f"surface_tractions[{i}] is missing key(s) "
                    f"{sorted(missing)} — full record: {st!r}")
            nodes_pair = st["nodes"]
            if isinstance(nodes_pair, np.ndarray):
                is_pair = nodes_pair.ndim == 1 and nodes_pair.shape[0] == 2
            elif isinstance(nodes_pair, (tuple, list)):
                is_pair = len(nodes_pair) == 2
            else:
                is_pair = False
            if not is_pair:
                raise ValueError(
                    f"surface_tractions[{i}]['nodes'] must be exactly a "
                    f"node pair (ni, nj), got: {nodes_pair!r}")
            ni, nj = nodes_pair
            ni = self._validate_node_id(ni)
            nj = self._validate_node_id(nj)
            if not (0 <= ni < self._nodes.shape[0]
                    and 0 <= nj < self._nodes.shape[0]):
                raise ValueError(
                    f"surface traction nodes ({ni},{nj}) out of range "
                    f"[0, {self._nodes.shape[0]-1}]")
            # 内部边载荷必须在此拒绝 — 绕过 add_traction 的边界检查会让
            # solve 成功而误差估计崩溃
            self._validate_boundary_edge(ni, nj)
            st["nodes"] = (ni, nj)   # 规范化写回 (整数)
            if st.get("is_pressure"):
                # 压力: 恰好 1 个标量 (数值或 callable) — 标量/1 元组
                # 规范化写回 1 元组 (消费方统一 trac[0])
                st["traction"] = _check_load_scalar(
                    st["traction"], f"surface_tractions[{i}]['traction']")
            else:
                _check_load_pair(
                    st["traction"], f"surface_tractions[{i}]['traction']")
        if self.body_force is not None and not callable(self.body_force):
            _check_load_pair(self.body_force, "body_force")

    def build_connectivity(self):
        """预计算 node→elements, element→neighbors, boundary edges + 几何缓存

        使用向量化 topology_core (NumPy argsort) 替代 Python 循环,
        10万节点网格预处理从秒级降到毫秒级.
        """
        if self._connectivity_built:
            return

        n_nodes = self.n_nodes
        n_elem = self.n_elements

        # (1) node_to_elems: CSR 格式, 惰性列表
        self.node_to_elems = node_element_table(self.elements, n_nodes)

        # (2) 边表 + 邻接 + 边界: 纯 NumPy argsort
        edge_table = build_edge_table(
            self.elements, self.element_kernel.local_edges, n_nodes)
        self.elem_neighbors = element_neighbor_table(edge_table, n_elem)
        self.boundary_edges = [
            (int(lo), int(hi)) for lo, hi in zip(
                edge_table.lo[edge_table.boundary_mask()],
                edge_table.hi[edge_table.boundary_mask()])
        ]
        self.edge_to_elems = edge_table.as_mapping()
        self.locator = ElementLocator(self.nodes, self.elements)

        # 内部边数据: [(a,b,e1,e2), ...]
        internal_mask = edge_table.counts == 2
        self.internal_edge_data = np.column_stack([
            edge_table.lo[internal_mask],
            edge_table.hi[internal_mask],
            edge_table.owners[internal_mask, 0],
            edge_table.owners[internal_mask, 1],
        ]).astype(np.int64)

        # (3) 单元专属几何缓存。标准键为 areas/centroids；内核可附加
        # b_coeffs、Gauss Jacobians 等私有缓存。
        geometry = self.element_kernel.build_geometry(
            self.nodes, self.elements)
        self._kernel_cache_names = tuple(geometry)
        for name, value in geometry.items():
            setattr(self, name, value)
        if getattr(self, "areas", None) is None:
            raise RuntimeError(
                f"{self.element_kernel.name} geometry did not provide 'areas'.")
        if getattr(self, "centroids", None) is None:
            raise RuntimeError(
                f"{self.element_kernel.name} geometry did not provide "
                f"'centroids'.")

        # (4) 通用单元 DOF 索引: (ne, 2*nnode_e)
        dof_base = 2 * self.elements
        n_local_dof = self.element_kernel.dofs_per_element
        self.element_dofs = np.empty(
            (self.n_elements, n_local_dof), dtype=np.int32)
        self.element_dofs[:, 0::2] = dof_base
        self.element_dofs[:, 1::2] = dof_base + 1

        self._connectivity_built = True

    # ── 基本属性 ──

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_elements(self):
        return self.elements.shape[0]

    @property
    def n_dof(self):
        """总自由度数 = 2 × 节点数 (Bathe Eq 4.18)"""
        return 2 * self.n_nodes

    @staticmethod
    def _validate_node_id(nid):
        """节点编号必须为整数 — 曾接受 1.5 静默约束错误 DOF (fix_node
        1.5 → 节点 1 Y + 节点 2 X) / 载荷接口组装时崩溃。

        兼容"有限且恰为整数"的浮点 (如 1.0); 拒绝 bool (True==1 陷阱)。
        """
        if isinstance(nid, bool):
            raise TypeError(f"node id must be an integer, got bool {nid}")
        if isinstance(nid, float) and float(nid).is_integer():
            nid = int(nid)
        if not isinstance(nid, (int, np.integer)):
            raise TypeError(
                f"node id must be an integer, got {type(nid).__name__} {nid!r}")
        return int(nid)

    def fix_node(self, nid, dof="both", value=0.0):
        """固定单个节点并指定位移值 (Bathe §4.2.2)

        参数
        ----
        nid : int — 节点索引 (0-based)
        dof : str — "x", "y", 或 "both" (默认)
        value : float — 指定位移值 [m]
        """
        nid = self._validate_node_id(nid)
        if not (0 <= nid < self.n_nodes):
            raise ValueError(f"fix_node: nid={nid} out of range [0, {self.n_nodes-1}]")
        if dof not in ("x", "y", "both"):
            raise ValueError(
                f"fix_node: dof='{dof}' — must be 'x', 'y', or 'both'")
        require_finite_scalar(value, "fix_node: prescribed displacement value")
        dofs = []
        if dof in ("x", "both"):
            dofs.append(2 * nid)
        if dof in ("y", "both"):
            dofs.append(2 * nid + 1)

        # set 判重 O(1)/次 — 每次全量重建 (set + sorted) 让整边固支
        # (bc_apply 逐节点调用) 呈 O(n² log n) 结构性超线性; 延迟落盘:
        # 数组在首次读取时一次性重建, 见 fixed_dofs property
        if self._fixed_set is None:
            self._fixed_set = set(self.fixed_dofs.tolist())
        existing = self._fixed_set
        for d in dofs:
            if d in existing:
                old_val = self.prescribed_vals.get(d, 0.0)
                # 绝对 1e-12 会让微尺度位移 (1e-13 vs 2e-13) 覆盖静默无警告
                #
                if abs(old_val - value) > 1e-12 * max(
                        abs(old_val), abs(value), np.finfo(float).tiny):
                    warnings.warn(f"fix_node: DOF {d} already constrained to "
                                  f"{old_val:.6e}, overwriting to {value:.6e}")
            existing.add(d)
            self.prescribed_vals[d] = value
        self._fixed_dirty = True

    def fix_nodes_func(self, node_list, func):
        """函数形式的给定位移 (Bathe §4.2.2: 非零位移约束)

        参数
        ----
        node_list : list[int]
            受约束节点索引列表
        func : callable or float
            func(x, y) → (ux, uy) 或 常数 value
            例: lambda x,y: (x/r, y/r) → 径向单位位移

        Bathe Eq 4.43: 非零位移 U_b 需修正右端项 R_a' = R_a - K_ab·U_b
        """
        if isinstance(node_list, str):
            # 空串迭代零次 → 静默 no-op (fuzz 收紧后暴露: 用户拼错类型
            # 时约束消失无提示, 属载荷静默错误族)
            raise ValueError(
                f"fix_nodes_func: node_list 必须是节点索引列表, "
                f"got 字符串 {node_list!r}")
        if isinstance(node_list, (int, np.integer)):
            # 单个节点号会被 for-in 迭代成裸 TypeError — 明示期望列表
            raise ValueError(
                f"fix_nodes_func: node_list 必须是节点索引列表, "
                f"got 单个节点 {node_list!r}")
        for nid in node_list:
            nid = self._validate_node_id(nid)
            # 范围检查先于索引 — 越界裸 IndexError (与 fix_node 一致)
            if not (0 <= nid < self.n_nodes):
                raise ValueError(
                    f"fix_nodes_func: nid={nid} out of range "
                    f"[0, {self.n_nodes-1}]")
            x, y = self.nodes[nid]
            if callable(func):
                result = func(x, y)
                if isinstance(result, numbers.Real):
                    ux = uy = result
                else:
                    try:
                        result = tuple(result)
                    except TypeError:
                        raise TypeError(
                            f"fix_nodes_func: func({x:.4g},{y:.4g}) 返回值"
                            f"类型非法: {result!r} — 需要 (ux, uy) 二元组"
                            "或标量") from None
                    if len(result) != 2:
                        # 多余分量静默忽略 — 载荷静默错误
                        raise ValueError(
                            f"fix_nodes_func: func({x:.4g},{y:.4g}) 返回 "
                            f"{len(result)} 个分量 — 需要 (ux, uy) "
                            "恰好 2 个")
                    ux, uy = result[0], result[1]
            else:
                ux = uy = func
            self.fix_node(nid, "x", ux)
            self.fix_node(nid, "y", uy)

    def add_force(self, nid, fx=0.0, fy=0.0):
        """添加节点集中力 [N] (Bathe: R_c 向量)

        参数
        ----
        nid : int — 节点索引 (0-based)
        fx, fy : float — x, y 方向集中力分量
        """
        nid = self._validate_node_id(nid)
        if not (0 <= nid < self.n_nodes):
            raise ValueError(f"add_force: nid={nid} out of range [0, {self.n_nodes-1}]")
        require_finite_scalar(fx, "add_force: fx")
        require_finite_scalar(fy, "add_force: fy")
        if fx != 0.0 or fy != 0.0:
            # abs>1e-30 阈值会把微尺度模型合法小载荷 (1e-31 N) 静默
            # 丢弃, 求解"成功"但位移全零
            self.concentrated_forces.append({"node": nid, "force": (fx, fy)})

    def _get_edge_elements(self, ni, nj):
        """返回包含边 (ni,nj) 的单元索引列表; 边不存在则 ValueError."""
        self.build_connectivity()
        key = (min(int(ni), int(nj)), max(int(ni), int(nj)))
        eids = self.edge_to_elems.get(key, [])
        if len(eids) == 0:
            raise ValueError(
                f"Edge ({ni},{nj}) is not a mesh edge — "
                f"nodes must be connected by an element side.")
        return eids

    def _validate_boundary_edge(self, ni, nj):
        """验证 (ni,nj) 是网格中的边界边 (恰好属于一个单元)."""
        eids = self._get_edge_elements(ni, nj)
        if len(eids) > 1:
            raise ValueError(
                f"Edge ({ni},{nj}) is an interior edge shared by {len(eids)} elements. "
                f"Surface tractions only supported on boundary edges.")
        return eids[0]

    def boundary_outward_normal(self, ni, nj):
        """返回边界边 (ni,nj) 的单位外法向 — 由相邻单元 CCW 方向确定

        算法: 找到包含此边的唯一单元, 取单元局部 CCW 边 (a→b) 的方向,
        则外法向 n = (dy/L, -dx/L) (Bathe §5.3.2: CCW 域外法向 = 切向顺时针90°).

        add_pressure 内部自动调用此方法, 因此 add_pressure(ni,nj,p) 和
        add_pressure(nj,ni,p) 结果完全相同。

        返回
        ----
        (nx, ny) : tuple[float, float] — 单位外法向分量
        """
        ni = self._validate_node_id(ni)
        nj = self._validate_node_id(nj)
        eid = self._validate_boundary_edge(ni, nj)
        conn = self.elements[eid]

        # 找到单元中与 (ni,nj) 匹配的局部 CCW 边
        for ia, ib in self.element_kernel.local_edges:
            a, b = conn[ia], conn[ib]
            if {int(a), int(b)} == {int(ni), int(nj)}:
                xa, ya = self.nodes[a]
                xb, yb = self.nodes[b]
                dx, dy = xb - xa, yb - ya
                L = float(np.hypot(dx, dy))
                edge_ulp = 64.0 * np.finfo(float).eps * max(
                    float(max(abs(xa), abs(xb), abs(ya), abs(yb))),
                    np.finfo(float).tiny)
                if L <= edge_ulp:
                    raise ValueError(
                        f"Zero-length edge ({ni},{nj}) "
                        f"(L={L:.3e} <= ULP {edge_ulp:.3e}).")
                # CCW 单元局部边 (a→b): 外法向 = 切向顺时针90°
                return float(dy / L), float(-dx / L)

        raise RuntimeError(
            f"Boundary edge ({ni},{nj}) not found in adjacent element {eid}. "
            f"This should not happen — check mesh consistency.")

    def add_traction(self, ni, nj, tx, ty):
        """添加边上的均布面力 [Pa] (Bathe §4.2.1: R_s 向量).

        仅支持边界边 (恰好属于1个单元). 内部边应使用独立的界面模型.
        """
        ni = self._validate_node_id(ni)
        nj = self._validate_node_id(nj)
        for n in (ni, nj):
            if not (0 <= n < self.n_nodes):
                raise ValueError(f"add_traction: node {n} out of range [0, {self.n_nodes-1}]")
        for name, val in (("tx", tx), ("ty", ty)):
            if not callable(val):
                require_finite_scalar(val, f"add_traction: {name}")
        eids = self._get_edge_elements(ni, nj)
        if len(eids) != 1:
            raise ValueError(
                f"Edge ({ni},{nj}) is shared by {len(eids)} elements; "
                f"surface tractions require a boundary edge (exactly 1 element).")
        self._append_traction({"nodes": (ni, nj), "traction": (tx, ty)})

    def add_pressure(self, ni, nj, p):
        """添加边上的法向压力 [Pa] — 自动从单元方向计算 t = -p·n (Bathe §4.2.1)

        外法向由相邻单元的 CCW 局部边方向确定, 不依赖调用者传入的节点顺序。
        add_pressure(ni,nj,p) 和 add_pressure(nj,ni,p) 结果完全相同。

        对直线、圆弧、椭圆、样条离散边均适用, 自动处理外边界和孔洞方向。

        参数
        ----
        ni, nj : int — 边两端节点索引 (0-based, 顺序无关)
        p : float or callable — 压力幅值 [Pa] (正值=压缩, 指向域内)
        """
        ni = self._validate_node_id(ni)
        nj = self._validate_node_id(nj)
        for n in (ni, nj):
            if not (0 <= n < self.n_nodes):
                raise ValueError(f"add_pressure: node {n} out of range [0, {self.n_nodes-1}]")
        if not callable(p):
            require_finite_scalar(p, "add_pressure: p")
        # 不缓存外法向: 组装时由当前几何重新计算 (boundary_outward_normal),
        # 这样 replace_nodes/replace_elements 改变几何后载荷自动跟随 —
        # 缓存法向会导致改几何后压力沿用旧方向。
        self._append_traction({
            "nodes": (ni, nj),
            "traction": (p,),
            "is_pressure": True,
        })

    def _append_traction(self, record):
        """追加面力记录 — 同一边界边重复施加曾静默双倍载荷 (位移精确
        ×2, 审查实测比值 2.0000)。

        保持累加语义 (载荷拆分是既有锁定契约:
        test_error_indicator_invariant_to_load_splitting), 因此
        重复施加 → 响亮警告而非去重; 交互路径 (bc_apply 段选择去重)
        在段选择层防重, API 路径在施加层提醒。节点对按无序 (min,max)
        归一 — add_pressure(ni,nj) 与 add_pressure(nj,ni) 是同一载荷。
        """
        ni, nj = record["nodes"]
        key = (min(ni, nj), max(ni, nj))
        is_pressure = bool(record.get("is_pressure", False))
        for existing in self.surface_tractions:
            a, b = existing["nodes"]
            if (min(a, b), max(a, b)) == key \
                    and bool(existing.get("is_pressure", False)) == is_pressure:
                if _same_traction(existing["traction"], record["traction"]):
                    warnings.warn(
                        f"边 {key} 已施加完全相同面力 "
                        f"{existing['traction']!r} — 重复施加将使载荷翻倍 "
                        "(×2)。若为误操作请移除重复记录。",
                        UserWarning, stacklevel=3)
                else:
                    warnings.warn(
                        f"边 {key} 已施加不同面力 {existing['traction']!r} — "
                        f"新面力 {record['traction']!r} 将线性叠加。"
                        "若为误操作请移除重复记录。",
                        UserWarning, stacklevel=3)
        self.surface_tractions.append(record)

    def nodes_on_edge(self, axis, edge, tol=None):
        """查找边界框特定边上的节点.

        axis: 'x' 或 'y'. edge: 'min' 或 'max'.
        tol=None → 自动: span × 1e-8 (覆盖 μm~km 尺度).
        tol=0  → 严格相等.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")
        if edge not in ("min", "max"):
            raise ValueError(f"edge must be 'min' or 'max', got '{edge}'")
        if tol is not None:
            tol = require_finite_scalar(tol, "nodes_on_edge: tol")
            if tol < 0:
                raise ValueError(
                    f"nodes_on_edge: tol={tol!r} — must be non-negative")
        col = 0 if axis == "x" else 1
        val = self.nodes[:, col].min() if edge == "min" else self.nodes[:, col].max()
        span = max(np.ptp(self.nodes[:, 0]), np.ptp(self.nodes[:, 1]))
        # span<1e-30 → 1.0 绝对地板会让子 1e-30 跨度模型 tol=1e-8 覆盖
        # 全部节点, 边界查询静默返回整条边
        if span < 64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(self.nodes))), np.finfo(float).tiny):
            span = np.finfo(float).tiny
        effective_tol = tol if tol is not None else span * 1e-8
        return np.where(np.abs(self.nodes[:, col] - val) <= effective_tol)[0]

    def check_jacobian(self):
        """验证所有单元采样点的 Jacobian 行列式为正值。

        返回
        ----
        ok : bool
            所有单元 Jacobian 为正
        bad : list[int]
            Jacobian ≤ 0 的单元索引列表
        """
        self.build_connectivity()
        report = self.element_kernel.jacobian_report(self)
        return report.ok, report.bad.tolist()

    def check_rigid_body_constraints(self):
        """逐连通分量检查是否消除了全部刚体模态 (Bathe §4.2.2)

        每个 2D 连通分量需要至少 3 个独立约束消除:
          x-平动, y-平动, 绕原点转动

        仅计数不足时会漏掉: 3 个共线约束无法阻止转动, 或约束在断开的零件上。

        结果缓存: BC 与网格未变时直接复用上次结果, 免重算
        connected_components (O(n) 图遍历) — 参数研究/收敛研究的
        重复 solve 是热路径。缓存键 = fixed_dofs 内容快照拷贝
        (np.array_equal 比较, 覆盖 fix_node/setter/原地修改全部写
        入口) + 网格变更经 invalidate_cache 失效 (replace_nodes /
        replace_elements)。返回的 list 只读复用 — 调用方不得原地修改。

        返回
        ----
        list[dict]: 每个有问题的分量 {"component": int, "nodes": list, "issue": str}
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        fixed = self.fixed_dofs
        if self._rigid_cache is not None:
            snapshot, cached = self._rigid_cache
            if np.array_equal(fixed, snapshot):
                return cached

        self.build_connectivity()
        n_nodes = self.n_nodes

        # ── 1. 连通分量分解 (复用向量化边表 — 取代 Python 双循环) ──
        edge_table = build_edge_table(
            self.elements, self.element_kernel.local_edges, n_nodes)
        row = np.concatenate([edge_table.lo, edge_table.hi])
        col = np.concatenate([edge_table.hi, edge_table.lo])
        data = np.ones(len(row), dtype=int)
        adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
        n_comp, labels = connected_components(adj, directed=False)

        # ── 2. 逐分量检查 ──
        issues = []

        for comp in range(n_comp):
            comp_nodes = np.where(labels == comp)[0]
            if len(comp_nodes) < 2:
                # 孤立节点: 无单元连接
                issues.append({
                    "component": comp,
                    "nodes": comp_nodes.tolist(),
                    "issue": f"孤立分量 ({len(comp_nodes)} 节点, 无单元连接)"
                })
                continue

            # 该分量中被约束的 DOF — 向量化: 全分量 DOF 一次 np.isin
            # (旧实现逐节点 Python 循环 + set 成员查询, 100k 节点级
            # 网格是热路径)。2n 与 2n+1 互异且 comp_nodes 无重复 →
            # 结果 set 的元素集与旧循环逐元素一致 (len 相等)。
            comp_dofs_all = np.concatenate([2 * comp_nodes, 2 * comp_nodes + 1])
            comp_dofs = set(comp_dofs_all[np.isin(comp_dofs_all, fixed)].tolist())

            if len(comp_dofs) == 0:
                issues.append({
                    "component": comp,
                    "nodes": comp_nodes.tolist(),
                    "issue": "无任何约束 — 保留 3 个刚体模态"
                })
                continue

            # ── 刚体模态约束矩阵 (中心化 + 归一化, 与坐标系原点无关) ──
            # R_i = [[1, 0, -y_i], [0, 1, x_i]]  (2×3)
            # 被约束行 → R_constrained (n_fixed × 3)
            xy = self.nodes[comp_nodes]
            origin = xy.mean(axis=0)
            # 分量特征尺寸: 局部跨度 + 坐标 ULP — 固定 1.0 下限在极小
            # 尺寸模型 (1e-16) 下破坏刚体模态检查的尺度不变性, 合法
            # 约束被误报"仍有转动模态" (内层 1.0 残留
            # 在坐标<1 且跨度<~3e-29 时仍把旋转列缩到 SVD 阈值下)
            scl = max(np.ptp(xy, axis=0).max(),
                      64.0 * np.finfo(float).eps * max(
                          float(np.max(np.abs(xy))), np.finfo(float).tiny))
            constrained_nodes = sorted(set(d // 2 for d in comp_dofs))
            R_rows = []
            for n in constrained_nodes:
                x = (self.nodes[n, 0] - origin[0]) / scl
                y = (self.nodes[n, 1] - origin[1]) / scl
                if (2*n) in comp_dofs:
                    R_rows.append([1.0, 0.0, -y])   # x-平动 + 转动
                if (2*n + 1) in comp_dofs:
                    R_rows.append([0.0, 1.0,  x])   # y-平动 + 转动

            R = np.array(R_rows)  # (n_constrained, 3)
            rank = np.linalg.matrix_rank(R)

            if rank < 3:
                missing = []
                if rank < 2:
                    # 检查缺少哪个平动
                    _, _, vt = np.linalg.svd(R)
                    null = vt[2] if rank < 2 else vt[1]
                    if abs(null[0]) > 0.5:
                        missing.append("x-平动")
                    if abs(null[1]) > 0.5:
                        missing.append("y-平动")
                    if abs(null[2]) > 0.5:
                        missing.append("转动")
                elif rank == 2:
                    # 零空间向量指示真正缺失的模态 — 一律报"转动"会掩盖
                    # 三个仅 y 约束的节点缺失的是 x 平动
                    _, _, vt = np.linalg.svd(R)
                    null = vt[2]
                    if abs(null[0]) > 0.5:
                        missing.append("x-平动")
                    elif abs(null[1]) > 0.5:
                        missing.append("y-平动")
                    else:
                        missing.append("转动 (约束共线或等价)")

                issues.append({
                    "component": comp,
                    "nodes": comp_nodes.tolist(),
                    "issue": (f"约束不足 ({len(constrained_nodes)} 节点约束, "
                              f"rank={rank}/3) — 残留: {', '.join(missing) if missing else '未知'}"),
                    "constrained_nodes": constrained_nodes,
                })

        # 快照必须拷贝: 原地修改 fixed_dofs 数组 (setflags(write=True)
        # 绕过 setter/fix_node) 时, 内容比较才能发现变化 — 存引用会
        # 把数组与自己比较, 永远命中陈旧结果
        self._rigid_cache = (fixed.copy(), issues)
        return issues

    def info(self):
        """返回网格摘要字符串

        包含: 节点数、单元数、DOF 数、Jacobian 状态、边界边数
        """
        ok, bad = self.check_jacobian()
        self.build_connectivity()  # 确保拓扑已构建
        n_bdy = len(self.boundary_edges) if self.boundary_edges else 0
        return (f"Mesh: {self.n_nodes} nodes, {self.n_elements} elements  "
                f"{self.elem_type}/{self.element_kernel.name}  "
                f"{self.n_dof} DOFs  Jacobian: {'OK' if ok else f'{len(bad)} BAD'}  "
                f"boundary: {n_bdy} edges")


def _reject_complex_nodes(nodes, arg_name):
    """复数节点坐标 → TypeError (与 elements 拒绝族对齐).

    复数数组 `np.asarray(..., dtype=float)` 只发 ComplexWarning 并把
    虚部静默丢弃 — 坐标 1+5j 变 1.0, 几何被静默改变 (fuzz 已找到
    ComplexWarning 但未判 problem)。全库其余位置 (elements 索引 /
    principal_stresses / von_mises / nodal_average) 均显式拒绝复数,
    nodes 是漏网之鱼 — 在此补齐, 错误类型与消息风格对齐 elements
    的既有拒绝 (TypeError + dtype 上下文)。
    """
    arr = np.asarray(nodes)
    if np.iscomplexobj(arr):
        raise TypeError(
            f"{arg_name} must be a real-valued (x, y) coordinate array, "
            f"got dtype {arr.dtype} — complex coordinates are rejected "
            f"(imaginary parts would be silently discarded)")


def _same_traction(a, b):
    """两条面力记录的值是否完全相同 (callable 按对象同一性比较).

    add_traction/add_pressure 已验证分量有限, 此处只需数值相等。
    callable (空间分布面力) 无法按值比较, 同对象引用视为同一载荷。
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if callable(x) or callable(y):
            if x is not y:
                return False
        elif x != y:
            return False
    return True
