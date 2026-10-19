"""T13 真实运行器的冻结配置、费用口径与差异检查。"""
from __future__ import annotations

import json
from pathlib import Path

from agents.retail_agent import RetailAgent
from evaluation import run_retail_live as live
from mcp.retail_store import RetailStore


def test_frozen_assets_and_prices_match_records():
    assets = live._asset_record()
    assert all(assets[name]["ok"] for name in ("seed", "policy", "knowledge", "license"))
    assert assets["knowledge_count"]["actual"] == 29
    assert assets["seed_counts_ok"] is True

    prices = live._pricing()
    assert prices["input_price_cny_per_mtok"] == 2.0
    assert prices["output_price_cny_per_mtok"] == 8.0

    snapshot = live._seed_snapshot()
    assert len(snapshot["users"]) == 500
    assert len(snapshot["orders"]) == 1000
    assert len(snapshot["products"]) == 50
    assert snapshot["orders"]["#W8171054"]["status"] == "delivered"


def test_case_assets_are_isolated_and_chroma_safe():
    first = live._case_assets({"id": "V01"}, "abcd1234")
    second = live._case_assets({"id": "V02"}, "abcd1234")
    assert first["collection"] == "t13v01_abcd1234"
    assert second["collection"] == "t13v02_abcd1234"
    assert first["collection"] != second["collection"]
    assert first["memory_collections"] == [
        "episodic_t13v01_abcd1234",
        "user_profile_t13v01_abcd1234",
    ]
    assert first["redis_prefix"] == "t13live:abcd1234:V01:"
    assert first["client_user"] != second["client_user"]


def test_order_diff_only_accepts_return_fields():
    baseline = {
        "users": {"u1": {"user_id": "u1"}},
        "products": {"p1": {"product_id": "p1"}},
        "orders": {
            "#A": {"order_id": "#A", "status": "delivered", "items": [1]},
            "#B": {"order_id": "#B", "status": "delivered"},
        },
    }
    final = {
        "users": {"u1": {"user_id": "u1"}},
        "products": {"p1": {"product_id": "p1"}},
        "orders": {
            "#A": {
                "order_id": "#A",
                "status": "return requested",
                "items": [1],
                "return_items": ["i1"],
                "return_payment_method_id": "pm1",
            },
            "#B": {"order_id": "#B", "status": "delivered"},
        },
    }
    diff = live._order_diff(baseline, final)
    assert live._only_allowed_order_fields(diff, {"#A": {}}) is True
    assert diff["orders"][0]["changed_fields"] == [
        "return_items",
        "return_payment_method_id",
        "status",
    ]

    changed_user = json.loads(json.dumps(final))
    changed_user["users"]["u1"]["name"] = "changed"
    assert live._only_allowed_order_fields(
        live._order_diff(baseline, changed_user), {"#A": {}}
    ) is False

    unexpected_order = json.loads(json.dumps(final))
    unexpected_order["orders"]["#B"]["status"] = "return requested"
    assert live._only_allowed_order_fields(
        live._order_diff(baseline, unexpected_order), {"#A": {}}
    ) is False


def test_forbidden_claims_ignore_negated_disclaimers():
    pattern = r"(退款已到账|邮件已发送|仓库已收货)"
    disclaimer = (
        "提交后订单状态将变为 return requested，不代表退款已到账、"
        "邮件已发送或仓库已收货。"
    )
    assert live._forbidden_claims(disclaimer, [pattern]) == []

    claim = "您的退款已到账，请查收。"
    hits = live._forbidden_claims(claim, [pattern])
    assert hits and hits[0]["match"] == "退款已到账"


def test_harness_queries_match_store_columns(tmp_path):
    store = RetailStore(tmp_path / "retail.db", live.SEED_PATH)
    client, conv = "client-a", "conv-a"
    assert store.find_user_by_email(client, conv, "chen.silva2698@example.com")["success"]
    assert live._bound_user(store.db_path, client, conv) == "chen_silva_7485"

    store.begin_turn(client, conv)
    prepared = store.prepare_return(
        client, conv, "#W3069600", ["5012998807"], "credit_card_1565124"
    )
    assert prepared["success"], prepared
    assert store.mark_pending_displayed(
        client, conv, prepared["op_id"], 1, False
    )
    store.begin_turn(client, conv)
    submitted = store.submit_return(
        client,
        conv,
        "#W3069600",
        ["5012998807"],
        "credit_card_1565124",
        "确认，就按这些申请退货",
    )
    assert submitted["success"], submitted

    operations = live._return_operations(store.db_path, client, conv)
    assert len(operations) == 1
    assert operations[0]["order_id"] == "#W3069600"


def test_plan_counts_declared_requests_and_hard_upper(tmp_path):
    case = next(
        item
        for item in live._load_json(live.CASES_PATH)["cases"]
        if item["id"] == "V01"
    )
    store = RetailStore(tmp_path / "retail.db", live.SEED_PATH)
    agent = RetailAgent(
        client=None,
        model="fake-model",
        store=store,
        tool_manager=object(),
        skill_manager=None,
        max_tokens=live.AGENT_MAX_TOKENS,
    )
    sample = live._sample_payloads(
        store,
        "chen.silva2698@example.com",
        "#W8171054",
        ["9747045638"],
        "credit_card_1565124",
    )
    plan = live._simulate_case_plan(
        agent,
        case,
        sample,
        "chen_silva_7485",
        live._pricing(),
    )
    assert plan["planned_requests"] == sum(case["max_requests_per_turn"])
    assert plan["planned_cost_cny"] > 0
    assert plan["hard_upper_max_cny"] > 0
    assert all(
        item["hard_input_tokens"] > item["plan_input_tokens"]
        for item in plan["requests"]
    )
