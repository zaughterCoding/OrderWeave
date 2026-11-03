"""按需 Skill 的不变量：政策原文完整拆分、写工具须先加载绑定 Skill。"""
from types import SimpleNamespace

import pytest

from agents.agent_loop import LoopContext
from agents.retail_hooks import SkillGateHook
from core.retail_skills import RetailSkillLibrary
from evaluation.harness import config as cfg

pytestmark = pytest.mark.skipif(not cfg.POLICY_PATH.exists(), reason="缺少 τ² retail 政策原文")


@pytest.fixture()
def library():
    return RetailSkillLibrary(cfg.POLICY_PATH.read_text(encoding="utf-8"))


def test_policy_split_is_lossless(library):
    policy = cfg.POLICY_PATH.read_text(encoding="utf-8")
    base = library.base_policy()
    loaded = [library.load(name)["policy_text"] for name in library.skills]
    # 每个政策章节要么常驻，要么恰好属于一个 Skill；拼回去与原文一致。
    assert library.preamble + "".join(library.sections.values()) == policy
    for title, text in library.sections.items():
        hits = int(text.strip() in base) + sum(text.strip() in body for body in loaded)
        assert hits == 1, title


def test_write_tool_requires_loaded_skill(library):
    hook = SkillGateHook()
    req = SimpleNamespace(retail_skills=library, loaded_skills=[])
    ctx = LoopContext(req=req, agent=SimpleNamespace(language="en"), state={})
    assert hook.pre_tool(ctx, "cancel_pending_order", {}) is not None
    assert hook.pre_tool(ctx, "get_order_details", {}) is None
    hook.post_tool(ctx, "load_skill", {"name": "cancel_pending_order"}, library.load("cancel_pending_order"))
    assert hook.pre_tool(ctx, "cancel_pending_order", {}) is None
    assert hook.pre_tool(ctx, "return_delivered_order_items", {}) is not None
    # 之前轮次已加载（来自消息日志）同样有效。
    later = LoopContext(req=SimpleNamespace(retail_skills=library, loaded_skills=["return_delivered_order"]),
                        agent=SimpleNamespace(language="en"), state={})
    assert hook.pre_tool(later, "return_delivered_order_items", {}) is None
