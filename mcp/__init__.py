"""本地 mcp 包。

本项目自身的 mcp 目录与官方 MCP Python SDK 同名。这里把 SDK 的包目录追加到
``__path__``，使 ``mcp.types``、``mcp.client`` 等 SDK 子模块仍可正常导入，
同时保留本地模块与现有 ``import mcp.*`` 写法。
"""

from importlib import metadata
from pathlib import Path

try:
    _sdk_dist = metadata.distribution("mcp")
except metadata.PackageNotFoundError as exc:  # pragma: no cover - 依赖缺失时直接失败
    raise ImportError("未安装官方 MCP SDK（mcp），无法加载 mcp 子模块") from exc

_sdk_dir = Path(_sdk_dist.locate_file("mcp"))
if _sdk_dir.is_dir() and str(_sdk_dir) not in __path__:
    __path__.append(str(_sdk_dir))
