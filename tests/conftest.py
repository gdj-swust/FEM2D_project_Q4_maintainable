"""测试会话级配置 — 中文字体缺字警告收敛 (包 6).

绘图测试经 plot_three/colorbar 渲染带中文标签 (如 "磨平"/"面力") 的
图形。matplotlib 在无 CJK 字体机器上对每个缺失字形发 UserWarning
"Glyph ... missing from font(s)" — 纯渲染噪音: 测试只断言图内容
(collections/lines), 不检查字形; fem2d.visualize 在字体存在时已显式
选用 CJK 字体, 本过滤只兜底字体缺失环境, 不改任何断言语义。

必须用 pytest 的 filterwarnings 机制 (pytest_configure 注册), 而非
模块级 warnings.filterwarnings — pytest 以 catch_warnings(record=True)
捕获 (其 "always" 记录器位于全局过滤器之前), 模块级过滤被绕过,
警告仍会出现在汇总里。
"""


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Glyph \\d+ \\(.*\\) missing from font\\(s\\):UserWarning",
    )
