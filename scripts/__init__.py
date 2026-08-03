# scripts 工具层包声明 — 使打包后 `import scripts.geo_spec` 仍可用。
# (原为 PEP 420 namespace package, 依赖项目根 sys.path 注入; 打包安装
# 场景下需要显式 __init__.py。)
