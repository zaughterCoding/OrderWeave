"""自进化候选的不变量：只允许改描述与操作要点，不得改绑定工具或写入案例原文。"""
import shutil

from evaluation.harness import evolve


def _copy(tmp_path):
    out = tmp_path / "cand"
    shutil.copytree(evolve.SKILLS, out, ignore=lambda d, names: [n for n in names if d == str(evolve.SKILLS)
                                                                  and not n.startswith("retail-")])
    return out


def test_playbook_change_passes(tmp_path):
    out = _copy(tmp_path)
    change = {"changes": [{"skill": "retail-cancel-pending-order", "description": "Cancel a pending order.",
                           "playbook": "# Playbook\n\n1. Check the status first."}]}
    assert evolve.write_candidate(change, out) == []
    assert "Check the status first" in (out / "retail-cancel-pending-order" / "SKILL.md").read_text(encoding="utf-8")


def test_bound_tools_and_case_text_are_rejected(tmp_path):
    out = _copy(tmp_path)
    path = out / "retail-cancel-pending-order" / "SKILL.md"
    path.write_text(path.read_text(encoding="utf-8").replace("tools: cancel_pending_order",
                                                             "tools: cancel_pending_order, modify_user_address"),
                    encoding="utf-8")
    other = out / "retail-return-delivered-order" / "SKILL.md"
    other.write_text(other.read_text(encoding="utf-8") + "\nFor order #W1234567 refund to gift_card_123.\n",
                     encoding="utf-8")
    problems = evolve.validate(out)
    assert any("retail-cancel-pending-order" in p for p in problems)
    assert any("retail-return-delivered-order" in p for p in problems)
