"""死代码检测 — 基于 AST 的静态分析, 可复用脚本.

检测维度 (每个输出候选需人工核实, 自动排除已知合理模式):
  1. 未使用导入 (按文件内引用统计; 排除 ``__all__`` re-export 与跨模块 from-import)
  2. 模块级函数/类全项目零引用 (AST 精确计数; 顶层 fem2d 导出列入保护名单)
  3. 函数参数未使用 (排除 self/cls/下划线/闭包 callback/abstractmethod/保护函数)
  4. 模块级常量/变量赋值后未读
  5. 不可达语句 (return/raise/break/continue 后的语句)
  6. 类方法零调用 (类名限定计数 + 类内 self 调用; 顶层导出类的方法受保护)

v3 (2026-08): 修复 v2 的两个系统性漏报 (评审反馈):
  - 未使用导入按全项目变量名统计 → 其他文件用一次 os 就让所有文件的
    ``import os`` 逃检. v3 改为按文件内 Name 引用判定.
  - 方法按方法名统计 → 任意对象调用一次 .summary() 就让所有类的
    summary() 视为已使用. v3 改为类名限定计数 (ClassName.method() 或
    类内 self.method()).
  - 新增顶层导出保护名单: fem2d/__init__.py 的 __all__ 名字及其类方法
    默认不报 — 公共 API 不能仅凭仓库内部零调用判定为死代码
    (曾误删 ElementKernel.matches / RegionRegistry.summary /
     BoundaryDiagnostics.summary 三个公开方法).

用法: python scripts/check_dead_code.py [目录...]
"""
import ast
import os
import sys
from collections import Counter


def collect_files(roots):
    files = []
    for root in roots:
        if os.path.isfile(root):
            if root.endswith(".py"):
                files.append(root.replace("\\", "/"))
            else:
                print(f"  [WARN] 参数 '{root}' 不是 .py 文件 — 已跳过")
            continue
        if not os.path.isdir(root):
            # 不存在的路径曾静默产出"0 候选", 用户以为检查通过
            # (审计 2026-08-03)
            print(f"  [WARN] 目录不存在: '{root}' — 已跳过")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in filenames:
                if f.endswith(".py"):
                    files.append(os.path.join(dirpath, f).replace("\\", "/"))
    return files


def _read(path):
    try:
        return open(path, encoding="utf-8").read()
    except OSError:
        return None


def _norm(path):
    """统一分隔符并去掉 Windows 盘符 (支持绝对路径 roots)."""
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = p[2:].lstrip("/")
    return p


def _self_module(path):
    """文件路径 → 自身模块名 ("fem2d/loads.py" → "fem2d.loads")."""
    parts = _norm(path)[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _from_module_target(path, level, module):
    """解析 from-import 的完整目标模块名 (处理相对导入 level)."""
    if level == 0:
        return module or ""
    parts = _norm(path)[:-3].split("/")
    base = parts[:-1]  # 去掉文件自身
    if level > 1:
        base = base[:-(level - 1)]
    return ".".join(base + (module.split(".") if module else []))


def _collect_attr_types(tree):
    """收集可静态归因的类型信息.

    返回 (class_attr_types, func_ctx):
    - class_attr_types: {类名: {属性名: 类型名}} — ``self.attr = SomeClass(...)``
      或 ``self.attr = param`` (param 有类型标注, 如 __init__(self, table: EdgeTable))
    - func_ctx: {函数名: {"params": {参数名: 类型}, "locals": {变量名: 类型},
                          "class": 所属类名或 None}}
      局部变量归因仅限构造赋值 ``x = SomeClass(...)`` / ``x = SomeClass``
    """
    class_attr_types = {}
    func_ctx = {}

    def _annotated_params(fn):
        params = {}
        for a in fn.args.posonlyargs + fn.args.args:
            ann = a.annotation
            if ann is None:
                continue
            if isinstance(ann, ast.Name):
                params[a.arg] = ann.id
            elif isinstance(ann, ast.Constant) and isinstance(ann.value, str):
                params[a.arg] = ann.value  # "Mesh" 前向引用字符串标注
        return params

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls_attrs = {}
            methods = {}
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    methods[sub.name] = _annotated_params(sub)
                    for st in ast.walk(sub):
                        if not isinstance(st, ast.Assign):
                            continue
                        for t in st.targets:
                            if not (isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"):
                                continue
                            val = st.value
                            if isinstance(val, ast.Call) and isinstance(
                                    val.func, ast.Name):
                                cls_attrs[t.attr] = val.func.id
                            elif isinstance(val, ast.Name):
                                pmap = methods.get(sub.name, {})
                                if val.id in pmap:
                                    cls_attrs[t.attr] = pmap[val.id]
            class_attr_types[node.name] = cls_attrs
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    locals_ = {}
                    for st in ast.walk(sub):
                        if isinstance(st, ast.Assign):
                            for t in st.targets:
                                if not isinstance(t, ast.Name):
                                    continue
                                val = st.value
                                if isinstance(val, ast.Call) and isinstance(
                                        val.func, ast.Name):
                                    locals_[t.id] = val.func.id
                                elif isinstance(val, ast.Name):
                                    locals_[t.id] = val.id
                    func_ctx[sub.name] = {
                        "params": _annotated_params(sub),
                        "locals": locals_,
                        "class": node.name,
                    }
        elif isinstance(node, ast.FunctionDef):
            locals_ = {}
            for st in ast.walk(node):
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        if not isinstance(t, ast.Name):
                            continue
                        val = st.value
                        if isinstance(val, ast.Call) and isinstance(
                                val.func, ast.Name):
                            locals_[t.id] = val.func.id
                        elif isinstance(val, ast.Name):
                            locals_[t.id] = val.id
            func_ctx[node.name] = {
                "params": _annotated_params(node),
                "locals": locals_,
                "class": None,
            }
    return class_attr_types, func_ctx


def _resolve_call_type(val, func_name, class_attr_types, func_ctx):
    """解析调用链 base 的静态类型 — 返回候选类型名列表.

    支持:
      - 参数类型标注  ``edges: EdgeTable`` → edges.method()
      - 局部构造赋值  ``builder = BoundarySegmentBuilder(...)`` → builder.method()
      - self 属性     ``self._table`` (类内属性赋值类型)
      - 链式属性     ``mesh.locator`` (mesh: Mesh → Mesh.locator 的类型)
    """
    if isinstance(val, ast.Name):
        name = val.id
        results = []
        ctx = func_ctx.get(func_name)
        if ctx:
            if name in ctx["locals"]:
                results.append(ctx["locals"][name])
            if name in ctx["params"]:
                results.append(ctx["params"][name])
        if name == "self" and ctx and ctx["class"]:
            results.append(ctx["class"])
        return results
    if isinstance(val, ast.Attribute):
        # 链式: mesh.locator / self._table
        parts = []
        v = val
        while isinstance(v, ast.Attribute):
            parts.append(v.attr)
            v = v.value
        if not isinstance(v, ast.Name):
            return []
        root = v.id
        parts.reverse()
        types = []
        ctx = func_ctx.get(func_name)
        if ctx:
            if root in ctx["locals"]:
                types.append(ctx["locals"][root])
            if root in ctx["params"]:
                types.append(ctx["params"][root])
        if root == "self" and ctx and ctx["class"]:
            types.append(ctx["class"])
        # 逐层属性传递: Type.attr1.attr2...
        for attr in parts:
            resolved = []
            for t in types:
                resolved.append(class_attr_types.get(t, {}).get(attr))
            types = [t for t in resolved if t]
            if not types:
                return []
        return types
    if isinstance(val, ast.Call) and isinstance(val.func, ast.Name):
        # SomeClass(...).method() — 构造链
        return [val.func.id]
    return []


def _collect_file_refs(path, tree, name_uses, attr_calls, from_imports,
                       class_self_calls, class_names, immune_attrs,
                       from_external, global_attr_types=None):
    """单文件全项目引用统计.

    - name_uses : Counter — Name 读取/调用计数 (定义处不产生 Name 节点;
       模块级赋值 target 计入 mod_target_nodes 被排除 — 否则"未使用常量"
       永远命中不了)
    - attr_calls : Counter[(base, attr)] — 属性形式调用, base 为限定名的
       首段 (ClassName.method() → (ClassName, method); 实例变量/链式调用
       x.method() → (None, method), 不归因任何类)
    - from_imports : set[(完整模块名, 名字)] — ``from M import Y`` 的目标
       (评审: 只按名字统计会让所有 from-import 互相免疫, 改为按来源模块
       精确匹配 — 只有"从本文件模块 import 了 Y"才构成 re-export 证据)
    - class_self_calls : dict[类名, {attr}] — 类内 self.method() 调用
    - class_names : set — 全项目类名 (用于模糊调用归因提示)
    - immune_attrs : set — tests/scripts 中的实例形式调用方法名:
        被测试调用过 = 活代码, 不报 (评审场景: 生产代码 x.summary() 不免疫)
    - from_external : bool — 本文件是否属于 tests/scripts
    """
    class_attr_types, func_ctx = _collect_attr_types(tree)
    if global_attr_types:
        # 跨文件链式归因 (如 base.py 的 mesh: Mesh → Mesh.locator 类型
        # 定义在 mesh.py) — 全局表优先, 本文件表补充
        merged = dict(global_attr_types)
        merged.update(class_attr_types)
        class_attr_types = merged

    def _walk(node, func_name):
        """带函数上下文遍历 — Call 归因需要知道当前函数 (参数/局部类型)."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk(child, child.name)
                continue
            if isinstance(child, ast.ClassDef):
                _walk(child, func_name)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                # 只统计 ast.Load (读取/调用) 作为使用 — 赋值 target 是
                # Store, 天然不保护同名模块常量
                name_uses[child.id] += 1
            elif isinstance(child, ast.Call) and isinstance(
                    child.func, ast.Attribute):
                val = child.func.value
                base = None
                if isinstance(val, ast.Name):
                    base = val.id
                # 静态归因: 参数标注 / 局部构造赋值 / self 属性 / 链式
                resolved = _resolve_call_type(
                    val, func_name, class_attr_types, func_ctx)
                if resolved:
                    for t in resolved:
                        if t:
                            attr_calls[(t, child.func.attr)] += 1
                    if from_external and resolved[0]:
                        immune_attrs.add(child.func.attr)
                else:
                    if isinstance(val, ast.Name):
                        base = val.id
                    elif isinstance(val, ast.Call) and isinstance(
                            val.func, ast.Name):
                        # SomeClass(...).method() 构造链式调用
                        base = val.func.id
                    attr_calls[(base, child.func.attr)] += 1
                    if from_external and base is not None:
                        immune_attrs.add(child.func.attr)
            elif isinstance(child, ast.ImportFrom):
                for a in child.names:
                    target = _from_module_target(path, child.level,
                                                 child.module)
                    # 存源符号名 (a.name), 非本地别名 (asname) — 否则
                    # "from A import x as y" 会让 A 的 x 被误判未使用
                    from_imports.add((target, a.name))
            _walk(child, func_name)

    _walk(tree, None)
    # 注意: import X 的模块名不计入 from_imports — 本文件自己 import
    # os 不应让"其他文件的 import os 未使用"免疫 (v2 曾系统性漏报)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            self_calls = set()
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "self"):
                    self_calls.add(sub.func.attr)
            class_self_calls[node.name] = self_calls


def _module_all_names(tree):
    """收集 ``__all__ = [...]`` 中的字符串名 (re-export 声明)."""
    all_names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                all_names.add(elt.value)
    return all_names


def _imported_names(tree):
    """返回 [(name, lineno, level, module)] 按文件内出现顺序.

    ImportFrom 携带 level/module (module=None 时为 import X 语句, 无来源匹配).
    跳过 ``if TYPE_CHECKING:`` 块内的导入 — 运行时无引用, 是类型检查标准模式.
    """
    type_checking_ids = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            type_checking_ids.update(id(n) for n in ast.walk(node))

    imported = []
    for node in ast.walk(tree):
        if id(node) in type_checking_ids:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.append((a.asname or a.name.split(".")[0],
                                 node.lineno, None, None))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue  # from __future__ import annotations 无运行时引用
            for a in node.names:
                imported.append((a.asname or a.name, node.lineno,
                                 node.level, node.module))
    return imported


def _all_params(fn):
    """所有参数名 (含 posonly/kwonly/vararg/kwarg)."""
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _is_abstract(fn):
    return any(
        (isinstance(d, ast.Name) and d.id == "abstractmethod")
        or (isinstance(d, ast.Attribute) and d.attr == "abstractmethod")
        for d in fn.decorator_list)


def analyze(path, src, name_uses, attr_calls, from_imports,
            class_self_calls, protected, fuzzy_attrs, immune_attrs):
    """返回 {dimension: [(lineno, name/desc, reason)]}"""
    tree = ast.parse(src)
    findings = {k: [] for k in ("import", "unused_def", "unused_param",
                                "unused_const", "unreachable", "unused_method")}

    # 1. 未使用导入 — 按本文件内 Name 引用判定 (v3: 不再用全项目统计,
    #    否则其他文件用一次 os 会让所有文件的 import os 逃检);
    #    排除 __all__ 声明 与 被其他模块从本文件模块 from-import 的 re-export
    #    (v3.1: 按 (来源模块, 名字) 精确匹配 — 不再让所有 from-import 互相免疫)
    file_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            file_names.add(node.id)
    all_names = _module_all_names(tree)
    self_module = _self_module(path)
    for name, line, level, module in _imported_names(tree):
        if name == "*":
            continue  # 星号导入无法静态判定, 脚本常见模式
        used = name in file_names or name in all_names
        if not used and module is not None:  # ImportFrom — re-export 证据
            used = (self_module, name) in from_imports
        if not used:
            findings["import"].append((line, name, "未使用导入"))

    # 3. 未使用参数 (每个函数)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_abstract(node):
                continue  # 协议桩签名 — 参数由子类实现
            if node.name in protected:
                continue  # 顶层导出的公共函数 — 签名是 API, 不能删
            args = _all_params(node)
            body_names = set()
            has_del = False
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    body_names.add(sub.id)
                if isinstance(sub, ast.Delete):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            has_del = True
            if has_del:
                continue  # 显式 del 模式 (如 recovery_shape_matrix)
            for a in args:
                if a in ("self", "cls"):
                    continue
                if a.startswith("_"):
                    continue
                if a not in body_names:
                    findings["unused_param"].append(
                        (node.lineno, f"{node.name}->{a}", "未使用参数"))

    # 4. 模块级常量赋值后未读 (全项目 Name 读取计数; 赋值 target 自身已从
    #    name_uses 排除 — v3.1 修复: 此前 UNUSED = 1 的目标计入 name_uses,
    #    "未使用常量"检测永远无法命中)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                if isinstance(t, ast.Name) and not t.id.startswith("__"):
                    if (name_uses[t.id] == 0
                            and (self_module, t.id) not in from_imports):
                        findings["unused_const"].append(
                            (node.lineno, t.id, "赋值后未读"))

    # 5. 不可达语句 (return/raise/break/continue 后, 无 if/else 汇合)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            for i, stmt in enumerate(body[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise)):
                    nxt = body[i + 1]
                    if not isinstance(nxt, (ast.FunctionDef, ast.ClassDef)):
                        findings["unreachable"].append(
                            (nxt.lineno, "return/raise 后的语句", "不可达"))

    # 2. 模块级函数/类零引用 (全项目 Name 计数 + 模块精确 from-import; 保护跳过)
    #    模块限定调用 (import M; M.foo()) 只产生 attr_calls 不产生 Name 读取 —
    #    曾漏计导致已使用的模块级函数被误报死代码 (审计 2026-08-03)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name.startswith("__"):
                continue
            if node.name in protected or node.name in all_names:
                continue  # 导出 API — 外部用户可能正在使用
            if (name_uses[node.name] == 0
                    and (self_module, node.name) not in from_imports
                    and not any(
                        attr == node.name for _base, attr in attr_calls)):
                findings["unused_def"].append(
                    (node.lineno, node.name, "全项目零引用"))

    # 6. 类方法零调用 — v3 类名限定计数: 仅当出现 ClassName.method() 或
    #    该类内 self.method() 才算使用; 实例变量形式 x.method() 不归因
    #    (无法证明类型, 宁报候补不误删公共方法; 存在模糊调用时给出提示).
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name in protected or node.name in all_names:
                continue  # 导出类 — 其方法全部受保护
            self_calls = class_self_calls.get(node.name, set())
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and not sub.name.startswith("_"):
                    if any(
                            isinstance(d, ast.Name) and d.id == "property"
                            for d in sub.decorator_list):
                        continue  # @property 以属性形式访问 (无括号)
                    if ((node.name, sub.name) in attr_calls
                            or sub.name in self_calls
                            or sub.name in immune_attrs):
                        continue
                    hint = ("  [提示: 存在实例形式 .method() 调用, 无法归因"
                            "类名, 可能为误报]" if sub.name in fuzzy_attrs else "")
                    findings["unused_method"].append(
                        (sub.lineno, f"{node.name}.{sub.name}{hint}",
                         "方法零调用 (类名限定)"))

    return findings


def main():
    # Windows 控制台输出 UTF-8 (GBK 控制台会产生乱码)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    roots = sys.argv[1:] or ["fem2d"]
    files = collect_files(roots)

    # 引用统计覆盖全部源码: fem2d + tests + scripts + 入口 (被测试调用的公共
    # API 不算死代码; 入口只统计不输出, 避免转发表/测试文件自身的误报刷屏)
    reference_files = list(files)
    for d in ("tests", "scripts"):
        if os.path.isdir(d):
            reference_files.extend(collect_files([d]))
    for f in ("run.py", "run_demo.py"):
        if os.path.exists(f):
            reference_files.append(f)

    # 统一相对化到工作目录 — 绝对路径 roots 也能正确解析模块名
    # (_self_module / from_external 判定依赖相对路径形态)
    def _to_rel(f):
        try:
            return os.path.relpath(f, os.getcwd()).replace("\\", "/")
        except ValueError:  # Windows 跨盘符 relpath 不可行
            return _norm(f)

    sources = {_to_rel(f): _read(f) for f in reference_files}

    # ── 保护名单: 所有包 __init__.py 的 __all__ 导出 (公开 API 不能仅凭
    #    仓库内部零调用删除 — 顶层 fem2d/__init__.py 的导出全部在内) ──
    protected = set()
    for path, src in sources.items():
        if src is None or not path.endswith("__init__.py"):
            continue
        try:
            protected |= _module_all_names(ast.parse(src))
        except SyntaxError:
            pass

    name_uses = Counter()
    attr_calls = Counter()
    from_imports = set()
    class_self_calls = {}
    class_names = set()
    immune_attrs = set()

    # 全局类属性类型表 — 跨文件链式归因 (mesh: Mesh → Mesh.locator 类型)
    global_attr_types = {}
    for src in sources.values():
        if src is None:
            continue
        try:
            tree0 = ast.parse(src)
        except SyntaxError:
            continue
        for cls_name, attrs in _collect_attr_types(tree0)[0].items():
            global_attr_types.setdefault(cls_name, {}).update(attrs)

    for path, src in sources.items():
        if src is None:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        from_external = (path.startswith("tests/")
                         or path.startswith("scripts/"))
        _collect_file_refs(path, tree, name_uses, attr_calls, from_imports,
                           class_self_calls, class_names, immune_attrs,
                           from_external, global_attr_types)

    # 模糊调用: base 不是任何类名的实例形式调用 (x.method()) — 无法归因,
    # 不免疫候选 (防漏报), 但在候选输出时给出人工核实提示
    fuzzy_attrs = {
        attr for (base, attr) in attr_calls
        if base is None or base not in class_names
    }

    dim_labels = {
        "import": "未使用导入",
        "unused_def": "函数/类零引用",
        "unused_param": "未使用参数",
        "unused_const": "赋值后未读",
        "unreachable": "不可达语句",
        "unused_method": "方法零调用",
    }

    total = 0
    by_dim = Counter()
    for raw_path in files:
        path = _to_rel(raw_path)
        src = sources.get(path)
        if src is None:
            continue
        try:
            findings = analyze(path, src, name_uses, attr_calls,
                               from_imports, class_self_calls,
                               protected, fuzzy_attrs, immune_attrs)
        except SyntaxError:
            print(f"  [SyntaxError] {path}")
            continue
        for dim, items in findings.items():
            for line, name, _reason in items:
                total += 1
                by_dim[dim] += 1
                print(f"  {path}:{line}: {name} [{dim_labels[dim]}]")
    detail = " | ".join(f"{dim_labels[k]} {n}" for k, n in by_dim.most_common())
    print(f"\n共 {total} 个候选 (需人工核实): {detail}")
    if protected:
        print(f"保护名单: {len(protected)} 个 __all__ 导出名字 (含顶层 fem2d/__init__.py)")


if __name__ == "__main__":
    main()
