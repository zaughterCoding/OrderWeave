"""零售业务 Skill 的渐进式加载。

固定政策按章节拆分：通用规则（身份、领域说明、通用动作规则）常驻 system prompt，
各业务动作的政策原文章节与项目编写的操作要点组成 Skill，提示词只列出名称与一句话说明，
Agent 需要时调用 load_skill 取回正文。每个 Skill 绑定其写工具；钩子要求先加载对应
Skill 才能调用绑定的写工具，保证执行前读过该动作的政策原文。政策原文逐字保留，只是
分处两处；Skill 不能授权写入，确认仍由服务端判定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.tools import AgentToolSpec

SKILL_TOOL = "load_skill"
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "skills"
_MARKER = "retail_on_demand"


@dataclass(frozen=True)
class RetailSkill:
    name: str
    description: str
    tools: Tuple[str, ...]
    policy_sections: Tuple[str, ...]
    playbook: str


def _split_policy(policy: str) -> Tuple[str, Dict[str, str]]:
    """按二级标题切分政策；返回（标题前的序言，{标题: 含子标题的原文}）。"""
    parts = re.split(r"(?m)^(?=## )", policy)
    preamble, sections = parts[0], {}
    for part in parts[1:]:
        title = part.splitlines()[0][3:].strip()
        sections[title] = part
    return preamble, sections


class RetailSkillLibrary:
    def __init__(self, policy_text: str, root_dir: Optional[Path] = None):
        from core.skill_loader import SkillManager

        self.policy_text = policy_text
        self.preamble, self.sections = _split_policy(policy_text)
        parser = SkillManager.__new__(SkillManager)
        skills: Dict[str, RetailSkill] = {}
        for path in sorted(Path(root_dir or _DEFAULT_DIR).glob("*/SKILL.md")):
            meta, body = parser._split_front_matter(path.read_text(encoding="utf-8"))
            if meta.get("agents") != _MARKER or str(meta.get("enabled", "true")).lower() == "false":
                continue
            wanted = tuple(s.strip() for s in str(meta.get("policy_sections") or "").split(",") if s.strip())
            missing = [s for s in wanted if s not in self.sections]
            if missing:
                raise ValueError(f"Skill {meta['name']} 引用了政策中不存在的章节: {missing}")
            skills[meta["name"]] = RetailSkill(
                name=meta["name"], description=meta.get("description", ""),
                tools=tuple(t.strip() for t in str(meta.get("tools") or "").split(",") if t.strip()),
                policy_sections=wanted, playbook=body.strip())
        if not skills:
            raise ValueError("未找到零售按需 Skill")
        self.skills = skills
        claimed = {s for skill in skills.values() for s in skill.policy_sections}
        # 未被任何 Skill 认领的章节常驻；认领章节只在加载后出现。
        self.base_sections = [title for title in self.sections if title not in claimed]

    def bound_skills(self, tool: str) -> List[str]:
        return [s.name for s in self.skills.values() if tool in s.tools]

    def base_policy(self) -> str:
        """常驻部分：序言、未拆分章节原文与 Skill 目录。"""
        body = self.preamble + "".join(self.sections[t] for t in self.base_sections)
        catalog = "\n".join(f"- {s.name}: {s.description} (tools: {', '.join(s.tools)})" for s in self.skills.values())
        return (body.rstrip() + "\n\n## Action skills\n\n"
                "The policy sections for each action are provided as skills. Before preparing any action, call "
                f"{SKILL_TOOL} with the skill name to read that action's policy text and procedure; a write tool "
                "is rejected until its skill has been loaded in this conversation.\n\n" + catalog + "\n")

    def load(self, name: str) -> Dict[str, Any]:
        skill = self.skills.get(str(name or "").strip())
        if skill is None:
            return {"success": False, "error": f"Unknown skill. Available: {', '.join(self.skills)}"}
        policy = "".join(self.sections[t] for t in skill.policy_sections).strip()
        return {"success": True, "skill": skill.name, "tools": list(skill.tools),
                "policy_text": policy, "playbook": skill.playbook}

    def tool_spec(self) -> AgentToolSpec:
        def handler(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            return self.load(args.get("name", ""))

        return AgentToolSpec(
            name=SKILL_TOOL,
            description=("Load an action skill: returns the original policy text for that action and its procedure. "
                         "Call it before preparing that action; loading does not change any data."),
            input_schema={"type": "object", "properties": {"name": {"type": "string", "enum": list(self.skills),
                                                                    "description": "Skill name"}},
                          "required": ["name"]},
            handler=handler,
        )
