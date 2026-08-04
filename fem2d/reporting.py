"""求解结果的中文报告层 — 原 run.py 的摘要/警告打印逻辑。

纯输出职责: 计算结果摘要、网格质量摘要、物理建议警告 (体积自锁、
CST 弯曲刚度、误差未收敛)。不修改任何求解状态。
"""
import numpy as np


def displacement_scale(mesh, u):
    """云图变形放大系数 — 使最大位移在显示上约为特征尺寸的 10%.

    与 plot 层共享, 报告中也展示.
    """
    u2 = u.reshape(-1, 2)
    mag = np.sqrt(u2[:, 0] ** 2 + u2[:, 1] ** 2)
    span = (mesh.nodes[:, 0].max() - mesh.nodes[:, 0].min()
            + mesh.nodes[:, 1].max() - mesh.nodes[:, 1].min()) / 2
    if mag.max() == 0.0:
        # 零位移 (全约束/零载荷): 放大系数无意义 — 曾 1e-30 分母给出
        # 误导性的 1e6 上限
        return 1.0
    return max(0.0, min(span / mag.max() * 0.1, 1e6))


def bending_heuristics(mesh):
    """CST 弯曲方向的单元层数启发式 (厚度方向 < 6 时 CST 偏硬)."""
    x_span = mesh.nodes[:, 0].max() - mesh.nodes[:, 0].min()
    y_span = mesh.nodes[:, 1].max() - mesh.nodes[:, 1].min()
    h_char = np.sqrt(x_span * y_span / mesh.n_elements)  # 特征单元尺寸
    n_through_x = x_span / (h_char * 2)  # CST 弯曲方向启发式
    n_through_y = y_span / (h_char * 2)
    bending_stiff = (
        mesh.element_kernel.recovery_family == "cst"
        and min(n_through_x, n_through_y) < 6)
    return bending_stiff, n_through_x, n_through_y


def print_result_summary(config, mesh, result, z2, q, scale,
                         bending_stiff, n_through_x, n_through_y):
    """打印计算结果中文摘要 (位移 / 应力 / Z2 误差 / 网格质量)."""
    u = result['u']
    u2 = u.reshape(-1, 2)
    mag = np.sqrt(u2[:, 0] ** 2 + u2[:, 1] ** 2)
    vm_max = result['vm_stress'].max()
    vm_idx = int(np.argmax(result['vm_stress']))
    vm_x, vm_y = mesh.centroids[vm_idx]
    cond_info = result.get('condition_info')

    print(f"\n{'='*55}")
    print("  [Results] 计算结果摘要")
    print(f"{'='*55}")
    print("  求解状态:      [OK] 成功")
    # 标签与 solver 实际公式对齐: 分子只含自由 DOF 残差, penalty 路径是
    # 修改系统残差 — 曾标签声称全系统公式
    print(f"  求解系统残差 (后向误差) = {result['residual']:.2e}  "
          f"{'[OK] 正常' if result['residual'] < 1e-8 else '[WARN] 偏大'}")
    if cond_info and cond_info.get('condition_number') is not None:
        digits = cond_info.get('digits_lost', 0)
        print(f"  条件数 k(K)  = {cond_info['condition_number']:.2e}  "
              f"-> 约损失 {digits:.1f} 位有效数字  "
              f"{'[OK] 精度充足' if digits < 12 else '[WARN] 接近双精度极限'}")
    elif cond_info is not None:
        if cond_info.get("status") == "SINGULAR?":
            print("  条件数:        [FAIL] 特征值求解失败 — 刚度矩阵疑似奇异"
                  " (约束不足/孤立自由度)")
            print(f"                {cond_info.get('error', '')}")
        else:
            print("  条件数:        估计失败或问题规模过大, 已跳过")
    else:
        print("  条件数:        未计算 (solve 默认关闭, 用 --check-cond 开启)")
    print()
    print(f"  [Mesh] 网格质量  |  等级: {q['grade']}  "
          f"{'[OK]' if q['grade'] in ('A','B') else '[WARN]'}")
    print(f"    面积: {q['area_min']:.3e} ~ {q['area_max']:.3e} m^2  "
          f"均值 {q['area_mean']:.3e}  CV={q.get('area_cv',0):.1%}")
    print(f"    长宽比: <={q['ratio_max']:.1f}  "
          f"(<3: {q['ratio_ok']} / 3-5: {q['ratio_warn']} / >5: {q['ratio_bad']})")
    print(f"    角度: {q['angle_min']:.0f}~{q['angle_max']:.0f} deg  "
          f"(ok: {q['angle_ok']} / warn: {q['angle_warn']} / bad: {q['angle_bad']})")
    print(f"    Jacobian: {'[OK] 全部为正' if q['jacobian_neg'] == 0 else '[FAIL] 存在非正'}")
    print()
    print("  [Disp] 位移")
    print(f"    最大总位移  = {mag.max():.6e} m")
    print(f"    X向: {u2[:,0].min():.6e} ~ {u2[:,0].max():.6e} m  "
          f"(|max|={max(abs(u2[:,0].min()), abs(u2[:,0].max())):.6e})")
    print(f"    Y向: {u2[:,1].min():.6e} ~ {u2[:,1].max():.6e} m  "
          f"(|max|={max(abs(u2[:,1].min()), abs(u2[:,1].max())):.6e})")
    if mag.max() == 0.0:
        # 曾显示误导性的 "变形放大 1000000x" (零位移时 1e-30 分母)
        #
        print("    (零位移 — 无变形放大)")
    else:
        print(f"    (变形放大 {scale:.0f}x 用于云图显示)")
    print()
    print("  [Stress] 应力")
    print(f"    von Mises 最大值: {vm_max:.3e} Pa  ({vm_max*1e-6:.2f} MPa)")
    print(f"    位置: ({vm_x:.4f}, {vm_y:.4f}) m  [单元 #{vm_idx}]")
    print(f"    von Mises 最小值: {result['vm_stress'].min():.3e} Pa")
    print()
    print("  [Z2] Z2 能量误差估计 (Bathe 4.3.6)")
    print(f"    能量误差:  {z2['total_error']:.3e}")
    print(f"    应变能范数: {z2['energy_norm']:.3e}")
    print(f"    相对误差 eta: {z2['eta']:.1f}%  "
          f"{'[OK] 恢复型误差指标较低' if z2['eta'] < 10 else '[NOTE] 建议局部加密' if z2['eta'] < 20 else '[WARN] 需全局加密'}")
    print(f"    最差单元:   #{z2['worst_elem']} (贡献 {z2['elem_contrib'][z2['worst_elem']]:.1f}%)")
    print(f"    应力跳跃:   均值 {z2['stress_jumps']['avg_jump']:.3e}  "
          f"最大 {z2['stress_jumps']['max_jump']:.3e}")
    print()

    # ── 警告区 ──
    warnings = build_warnings(config, mesh, z2, bending_stiff,
                              n_through_x, n_through_y)
    if warnings:
        print(f"  [WARN] 警告 ({len(warnings)} 条):")
        for i, w in enumerate(warnings):
            print(f"  [{i+1}] {w}")
        print()

    print(f"{'='*55}")


def build_warnings(config, mesh, z2, bending_stiff, n_through_x, n_through_y):
    """按求解状态生成物理建议警告列表."""
    warnings = []

    # 体积自锁 (平面应变 + ν → 0.5)
    if config.plane == 'strain' and config.nu > 0.40:
        K_bulk = config.E / (3 * (1 - 2 * config.nu))
        warnings.append(
            f"  [WARN] 体积自锁风险 (Bathe 4.4.4):\n"
            f"    平面应变 + ν={config.nu} → 接近不可压缩\n"
            f"    体积模量 K = {K_bulk:.3e} Pa → ∞\n"
            f"    全积分位移单元可能过度刚硬\n"
            f"    建议: ν≤0.4 或换用混合/选择性积分格式单元")

    # Q4R 专用提示: compact 公式限制 (非编码错误, 见 q4r.py) —
    # 建议 Q4I 交叉验证。判据用 kernel.name: --elem-type Q4R 覆写后
    # elem_type=="Q4R" (kernel.name), CPS4R 白名单判断曾使警告静默消失
    kernel = getattr(mesh, "element_kernel", None)
    if getattr(kernel, "name", None) == "Q4R":
        warnings.append(
            "  [INFO] Q4R 为专用可选单元 (规则网格/长宽比<10/膜主导):"
            "\n    稳定性不如 Q4, 弯曲性能不如 Q4I —"
            "\n    若结果异常请用 Q4I (CPS4I) 交叉验证 (默认推荐)")

    # 弯曲刚度 (CST 弯曲时单元数太少)
    if bending_stiff:
        warnings.append(
            f"  [WARN] CST 弯曲刚度偏大 (Bathe 5.3.3):\n"
            f"    X向约 {n_through_x:.0f} 单元 / Y向约 {n_through_y:.0f} 单元\n"
            f"    CST 常应变特性导致弯曲变形被低估\n"
            f"    单元应变能中虚假剪切占比较高\n"
            f"    建议: 厚度方向至少 8-10 个单元 或换高阶单元")

    # 应力集中未收敛
    if z2['eta'] > 15:
        worst_elem = z2['worst_elem']
        warnings.append(
            f"  [WARN] 能量误差 eta={z2['eta']:.1f}% > 15%:\n"
            f"    网格在应力集中区可能不足\n"
            f"    建议: 在最差单元 #{worst_elem} 附近加密网格")

    return warnings
