"""防止错误标注和证据引用进入固定评测资料。"""

import json
import shutil
from pathlib import Path

import pytest

from evaluation.validate_retail_data import validate


DATA = Path(__file__).resolve().parents[1] / "evaluation" / "data" / "retail"


def test_retail_data_is_consistent():
    result = validate(DATA)
    assert result == {"status": "PASS", "intent_cases": 120, "retrieval_queries": 48, "knowledge_passages": 29, "model_calls": 0}


@pytest.mark.parametrize("fault", ["span", "split", "evidence", "source"])
def test_retail_data_rejects_corruption(tmp_path, fault):
    target = tmp_path / "retail"
    shutil.copytree(DATA, target)
    if fault == "source":
        with (target / "source" / "LICENSE").open("ab") as stream:
            stream.write(b"changed")
        expected = "来源哈希错误"
    else:
        filename = "retrieval.jsonl" if fault == "evidence" else "intent_slots.jsonl"
        path = target / filename
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if fault == "span":
            rows[0]["slots"][0]["text"] = "不在原句中的编号"
            expected = "槽位文本不匹配"
        elif fault == "split":
            rows[0]["split"] = "test"
            expected = "划分不一致"
        else:
            rows[0]["qrels"]["missing.P01"] = 2
            expected = "引用不存在的证据"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        validate(target)
