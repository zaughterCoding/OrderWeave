"""离线测试不读取项目根目录的 .env，必要配置由测试显式注入。"""
from __future__ import annotations

import dotenv
import pytest

# 阻止测试导入 api.main 等模块时自动加载本地密钥或运行配置。
dotenv.load_dotenv = lambda *args, **kwargs: False


@pytest.fixture(autouse=True)
def isolate_semantic_encoder(monkeypatch):
    """普通Fake回归不加载真实Qwen；语义用例自行开启或清除覆盖。"""
    monkeypatch.setenv("RETAIL_SEMANTIC_ROUTING", "false")
