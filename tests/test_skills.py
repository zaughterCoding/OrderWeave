from pathlib import Path

from core.skill_loader import SkillManager

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def test_skills_load_and_match_their_agents():
    manager = SkillManager(root_dir=str(SKILLS_DIR))

    skills = manager.load()

    assert {skill.name for skill in skills} == {
        "通用客服接待规范",
        "技术支持处理规范",
        "账单退款处理规范",
        "零售售后处理规范",
    }
    assert manager.errors == []
    assert manager.summary()["count"] == 4

    assert "通用客服接待规范" in manager.prompt_for("你好，我想咨询一下", "general")
    assert "技术支持处理规范" in manager.prompt_for("接口报 500 错误", "technical")
    assert "账单退款处理规范" in manager.prompt_for("我要申请退款", "billing")

    # 零售 Skill 每轮注入，且不混入旧账单审核规则。
    retail_prompt = manager.prompt_for("你好", "retail")
    assert "零售售后处理规范" in retail_prompt
    assert "账单退款处理规范" not in retail_prompt
    assert "零售售后处理规范" not in manager.prompt_for("你好", "general")

    # 限定给其他 Agent 的 Skills 不得泄漏进当前 Agent 的 prompt。
    assert "技术支持处理规范" not in manager.prompt_for("接口报 500 错误", "general")


def test_skills_dir_env_override_and_reload(tmp_path, monkeypatch):
    first = tmp_path / "first"
    first.mkdir()
    (first / "SKILL.md").write_text(
        "---\n"
        "name: 测试 Skill 一\n"
        "keywords: 退款\n"
        "agents: billing\n"
        "---\n"
        "# 测试 Skill 一\n"
        "第一条规则。\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ECHOMIND_SKILLS_DIR", str(tmp_path))

    manager = SkillManager()
    assert manager.root_dir == tmp_path.resolve()
    assert [skill.name for skill in manager.load()] == ["测试 Skill 一"]

    second = tmp_path / "second"
    second.mkdir()
    (second / "SKILL.md").write_text(
        "---\n"
        "name: 测试 Skill 二\n"
        "keywords: 报错\n"
        "agents: technical\n"
        "---\n"
        "# 测试 Skill 二\n"
        "第二条规则。\n",
        encoding="utf-8",
    )

    reloaded = manager.reload()

    assert {skill.name for skill in reloaded} == {"测试 Skill 一", "测试 Skill 二"}
    assert manager.errors == []
