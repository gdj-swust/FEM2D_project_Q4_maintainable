"""深度检查: 所有 import (含函数内延迟导入) 的目标模块/名字存在性.

删除代码后最常见的隐藏 bug 是"函数内延迟导入引用了已删名字" —
模块级 import 失败会在导入时报错, 但函数内 import 只有调用时才暴露.
本脚本静态解析全部 ImportFrom (含函数内), 验证目标模块存在且目标名字
在目标模块的顶层定义中存在。
"""
import ast
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# 支持命令行目录参数 (smoke 测试用临时目录验证崩溃场景) —
# 曾硬编码 ["fem2d", "scripts"], 传入的目录被静默忽略 (评审发现测试写空)
roots = sys.argv[1:] or ["fem2d", "scripts"]

files = []
for d in roots:
    if os.path.isfile(d):
        files.append(d.replace("\\", "/"))  # 直接传文件
        continue
    for root, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in fs:
            if f.endswith(".py"):
                files.append(os.path.join(root, f).replace("\\", "/"))
if not sys.argv[1:]:
    for f in ("run.py", "run_demo.py"):
        if os.path.isfile(f):
            files.append(f)

ast_cache = {}


def get_ast(path):
    if path not in ast_cache:
        try:
            ast_cache[path] = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            ast_cache[path] = None
    return ast_cache[path]


def _norm(path):
    """统一分隔符并去掉 Windows 盘符."""
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = p[2:].lstrip("/")
    return p


def _rel_to_roots(path):
    """文件路径相对扫描 roots 的模块路径.

    仅当 roots 是绝对路径时相对化 (临时目录 roots 下 a.py 的模块名
    必须是 'a' 而非 'D.ABAQUS.temp.tmpX.a'); 项目默认 roots 为相对
    路径 (fem2d), 保持原模块名语义 (fem2d/x.py → 'fem2d.x').
    """
    np_ = _norm(path)
    for r in roots:
        if not os.path.isabs(r):
            continue
        nr = _norm(r).rstrip("/")
        if np_ == nr or np_.startswith(nr + "/"):
            return np_[len(nr):].lstrip("/")
    return np_


def module_of(path):
    parts = _rel_to_roots(path)[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_target(src_path, level, module):
    if level == 0:
        return module
    parts = _rel_to_roots(src_path)[:-3].split("/")
    base = parts[:-1]
    if level > 1:
        base = base[:-(level - 1)]
    return ".".join(base + (module.split(".") if module else []))


def _collect_module_level(tree):
    """模块顶层绑定名 — 递归遍历顶层控制流 (If/Try/For/With 等) 但
    不深入函数/类体 (评审: ast.walk 会把函数局部误当模块顶层导出)."""
    names = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                                 ast.AsyncFunctionDef)):
                names.add(node.name)  # 定义名本身是模块级绑定
                continue             # 不深入函数/类体
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (node.targets if isinstance(node, ast.Assign)
                           else [node.target])
                for t in targets:
                    for elt in ast.walk(t):
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    names.add(a.asname or a.name)
            elif isinstance(node, (ast.If, ast.For, ast.AsyncFor,
                                   ast.While)):
                # 有 orelse 的控制流
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                # 无 orelse 属性 (评审发现: With 会导致 AttributeError 崩溃)
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        names.add(item.optional_vars.id)  # with ... as item
                visit(node.body)
            elif isinstance(node, ast.Try):
                visit(node.body)
                visit(node.orelse)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.finalbody)

    visit(tree.body)
    return names


def top_names(tree):
    """目标模块的模块级导出名 (用于验证导入目标存在性)."""
    return _collect_module_level(tree) if tree else set()


mod_paths = {module_of(f): f for f in files}
errors = []

for f in files:
    tree = get_ast(f)
    if tree is None:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = resolve_target(f, node.level, node.module)
            target_file = mod_paths.get(target)
            if target_file is None and not target.startswith((
                    "fem2d", "scripts")):
                continue  # 外部包 (numpy/scipy 等) — 假定存在
            if target_file is None:
                errors.append(
                    f"断裂模块: {f}:{node.lineno} "
                    f"from {node.module or '(package)'} (level={node.level}) "
                    f"→ 模块 '{target}' 不存在")
                continue
            ttree = get_ast(target_file)
            if ttree is None:
                continue
            tnames = top_names(ttree)
            for a in node.names:
                if a.name == "*":
                    continue
                if a.name in tnames:
                    continue
                # from pkg import submodule 合法模式: 名字是目标包的子模块
                sub_path = (target.replace(".", "/") + "/" + a.name + ".py"
                            ).replace("\\", "/")
                if any(sub_path == f2 or sub_path + "/__init__.py" == f2
                       for f2 in files):
                    continue
                errors.append(
                    f"断裂名字: {f}:{node.lineno} "
                    f"from {target} import {a.name} (目标模块无此顶层定义)")
        elif isinstance(node, ast.Import):
            # 普通 import 曾完全不检查 — `import fem2d.xx` 断裂静默通过
            #
            for a in node.names:
                target = a.name
                if not target.startswith(("fem2d", "scripts")):
                    continue  # 外部包 (numpy/scipy 等) — 假定存在
                if target not in mod_paths:
                    errors.append(
                        f"断裂模块: {f}:{node.lineno} "
                        f"import {target} → 模块 '{target}' 不存在")

print("═══ 深度 import 完整性检查 ═══")
if errors:
    for e in errors:
        print("  ", e)
    print(f"共 {len(errors)} 个问题")
    sys.exit(1)
print(f"0 个问题 — 全部 {len(files)} 个文件的 import "
      "(含函数内延迟导入) 目标完整")
