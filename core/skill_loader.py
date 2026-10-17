"""Skill 加载器。

Skill 是可在运行时热重载的业务指令，用于增强 Agent 的 system prompt。
它承载语气、流程、合规边界和客服话术，让运营无需发版即可调整。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass
class Skill:
    """与 Markdown/JSON 源格式无关的规范化 Skill。"""
    name: str
    description: str
    content: str
    path: str
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    enabled: bool = True

    def matches(self, message: str, agent_type: Optional[str] = None) -> bool:
        """返回该 Skill 是否应注入本次请求。

        agents 为空表示适用于所有 Agent；keywords 为空表示全局 Skill。
        否则消息中至少命中一个关键词才注入。
        """
        if not self.enabled:
            return False

        if self.agents and agent_type and agent_type.lower() not in self.agents:
            return False

        if not self.keywords:
            return True

        lowered = (message or "").lower()
        return any(keyword.lower() in lowered for keyword in self.keywords)

    def to_prompt_block(self, max_chars: int = 3200) -> str:
        """把 Skill 格式化为 prompt 块并限制长度。"""
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n..."
        description = f"\n说明: {self.description}" if self.description else ""
        return f"### {self.name}{description}\n{body}"

    def to_summary(self) -> Dict[str, Any]:
        """返回可被 API 序列化的摘要，不含完整内容。"""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "keywords": self.keywords,
            "agents": self.agents,
            "enabled": self.enabled,
            "content_chars": len(self.content),
        }


class SkillManager:
    """从目录中发现、解析和管理 Skill。

    支持的布局：
      1. skills/refund/SKILL.md
      2. skills/refund.json / skills/refund.md / skills/refund.txt
    """

    SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

    def __init__(self, root_dir: Optional[str] = None, max_prompt_chars: int = 5000):
        configured = root_dir or os.getenv("ECHOMIND_SKILLS_DIR") or str(_DEFAULT_SKILLS_DIR)
        self.root_dir = Path(configured).expanduser().resolve()
        self.max_prompt_chars = max_prompt_chars
        self._skills: List[Skill] = []
        self._errors: List[str] = []

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def load(self) -> List[Skill]:
        """重新扫描目录；单个文件损坏不影响其他文件。"""
        loaded: List[Skill] = []
        errors: List[str] = []

        if not self.root_dir.exists():
            logger.info(f"Skill 目录不存在，跳过加载: {self.root_dir}")
            self._skills = []
            self._errors = []
            return []

        for path in self._discover_files(self.root_dir):
            try:
                skill = self._load_file(path)
                if skill is not None:
                    loaded.append(skill)
            except Exception as ex:
                msg = f"{path}: {ex}"
                errors.append(msg)
                logger.warning(f"Skill 加载失败: {msg}")

        self._skills = loaded
        self._errors = errors
        self._log_loaded_skills()
        return self.skills

    def reload(self) -> List[Skill]:
        """供 API 使用的热重载入口。"""
        return self.load()

    def prompt_for(self, message: str, agent_type: Optional[str] = None) -> str:
        """为一次请求构造 Skill prompt。

        只注入匹配的 Skill。合并后的文本会被截断，避免挤占主对话上下文。
        """
        blocks: List[str] = []
        matched: List[tuple[Skill, List[str]]] = []
        remaining = self.max_prompt_chars
        lowered_message = (message or "").lower()

        for skill in self._skills:
            if not skill.matches(message, agent_type):
                continue
            matched_keywords = [
                keyword for keyword in skill.keywords
                if keyword.lower() in lowered_message
            ]
            block = skill.to_prompt_block()
            if len(block) > remaining:
                block = block[:remaining].rstrip() + "\n..."
            blocks.append(block)
            matched.append((skill, matched_keywords))
            remaining -= len(block)
            if remaining <= 0:
                break

        if not blocks:
            logger.debug(
                "Skills 未命中: agent=%s message=%r",
                agent_type or "all",
                (message or "")[:80],
            )
            return ""

        detail = "; ".join(
            f"{skill.name}(keywords={', '.join(keywords) if keywords else 'all'})"
            for skill, keywords in matched
        )
        logger.info(
            "Skills 已注入: agent=%s matched=%s message=%r",
            agent_type or "all",
            detail,
            (message or "")[:80],
        )

        return (
            "以下是当前请求可用的 EchoMind Skills。"
            "请优先遵循这些业务规则；如果与系统角色冲突，以系统角色和安全边界为准。\n\n"
            + "\n\n".join(blocks)
        )

    def summary(self) -> Dict[str, Any]:
        """返回供 /skills 端点和诊断使用的管理器状态。"""
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [skill.to_summary() for skill in self._skills],
            "errors": self.errors,
        }

    def _log_loaded_skills(self) -> None:
        """输出启动和热重载时可读的加载摘要。"""
        lines = [
            "",
            "================ EchoMind Skills Loaded ================",
            f"目录: {self.root_dir}",
            f"数量: {len(self._skills)}",
        ]

        if self._skills:
            for index, skill in enumerate(self._skills, start=1):
                agents = ", ".join(skill.agents) if skill.agents else "all"
                keywords = ", ".join(skill.keywords[:8]) if skill.keywords else "all"
                if len(skill.keywords) > 8:
                    keywords += ", ..."
                lines.extend([
                    f"{index}. {skill.name}",
                    f"   agents: {agents}",
                    f"   keywords: {keywords}",
                    f"   path: {skill.path}",
                ])
        else:
            lines.append("未加载任何 Skill。")

        if self._errors:
            lines.append("解析错误:")
            lines.extend(f"  - {error}" for error in self._errors)

        lines.append("========================================================")
        logger.info("\n".join(lines))

    def _discover_files(self, root_dir: Path) -> Iterable[Path]:
        """发现可加载的文件，优先采用 SKILL.md 目录约定。"""
        skill_md_files = sorted(root_dir.rglob("SKILL.md"))
        yielded = {path.resolve() for path in skill_md_files}
        for path in skill_md_files:
            yield path

        for path in sorted(root_dir.rglob("*")):
            resolved = path.resolve()
            if resolved in yielded or not path.is_file():
                continue
            if path.name.startswith(".") or path.name.upper() == "README.MD":
                continue
            if path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                yield path

    def _load_file(self, path: Path) -> Optional[Skill]:
        if path.suffix.lower() == ".json":
            return self._load_json(path)
        return self._load_text(path)

    def _load_json(self, path: Path) -> Optional[Skill]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("JSON Skill 必须是对象格式")

        content = str(raw.get("content") or raw.get("instructions") or "").strip()
        if not content:
            raise ValueError("缺少 content 或 instructions")

        return Skill(
            name=str(raw.get("name") or path.stem),
            description=str(raw.get("description") or ""),
            content=content,
            path=str(path),
            keywords=self._as_list(raw.get("keywords")),
            agents=[item.lower() for item in self._as_list(raw.get("agents"))],
            enabled=self._as_bool(raw.get("enabled"), default=True),
        )

    def _load_text(self, path: Path) -> Optional[Skill]:
        raw = path.read_text(encoding="utf-8")
        meta, body = self._split_front_matter(raw)
        body = body.strip()
        if not body:
            return None

        default_name = path.parent.name if path.name == "SKILL.md" else path.stem
        name = str(meta.get("name") or self._first_heading(body) or default_name)

        # 仅当前导标题与 Skill 名称重复时才去掉标题。
        body = self._strip_first_heading(body, name)

        return Skill(
            name=name,
            description=str(meta.get("description") or ""),
            content=body,
            path=str(path),
            keywords=self._as_list(meta.get("keywords")),
            agents=[item.lower() for item in self._as_list(meta.get("agents"))],
            enabled=self._as_bool(meta.get("enabled"), default=True),
        )

    def _split_front_matter(self, raw: str) -> tuple[Dict[str, Any], str]:
        """解析 Markdown 文件顶部的简单 front matter。

        刻意不引入 PyYAML，使这种轻量格式不增加运行时依赖。
        """
        text = raw.lstrip()
        if not text.startswith("---"):
            return {}, raw

        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw

        meta: Dict[str, Any] = {}
        end_idx: Optional[int] = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")

        if end_idx is None:
            return {}, raw
        return meta, "\n".join(lines[end_idx + 1:])

    @staticmethod
    def _first_heading(body: str) -> Optional[str]:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
        return None

    @staticmethod
    def _strip_first_heading(body: str, name: str) -> str:
        lines = body.splitlines()
        if not lines:
            return body
        first = lines[0].strip()
        if first.startswith("#") and first.lstrip("#").strip() == name:
            return "\n".join(lines[1:]).strip()
        return body

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
