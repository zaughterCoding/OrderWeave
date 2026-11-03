"""零售写操作的业务语义，逐条对应 τ² retail 官方工具。

语义移植自 sierra-research/tau2-bench 固定提交 fc0055dc 的
src/tau2/domains/retail/tools.py（MIT License，Copyright (c) 2025 Sierra Research）。
为使数据库状态与官方评分逐字段一致，保留其浮点计算与既有行为
（如修改待处理订单商品时，所有被改商品取循环最后一个新商品的价格与选项）。
本模块只处理字典形式的订单、用户和商品，不做身份、确认或持久化。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

WRITE_KINDS = {
    "return_delivered_order_items": "return",
    "exchange_delivered_order_items": "exchange",
    "cancel_pending_order": "cancel",
    "modify_pending_order_items": "modify_items",
    "modify_pending_order_address": "modify_address",
    "modify_pending_order_payment": "modify_payment",
    "modify_user_address": "modify_user_address",
}
ADDRESS_FIELDS = ("address1", "address2", "city", "state", "country", "zip")
CANCEL_REASONS = ("no longer needed", "ordered by mistake")


class DomainError(ValueError):
    """业务规则不允许该操作；消息与官方工具报错一致。"""


def _payment_method(user: Dict[str, Any], payment_method_id: str) -> Dict[str, Any]:
    method = (user.get("payment_methods") or {}).get(payment_method_id)
    if method is None:
        raise DomainError("Payment method not found")
    return method


def _variant(products: Dict[str, Any], product_id: str, variant_id: str) -> Dict[str, Any]:
    product = products.get(product_id)
    if product is None:
        raise DomainError("Product not found")
    variant = (product.get("variants") or {}).get(variant_id)
    if variant is None:
        raise DomainError("Variant not found")
    return variant


def _check_items_exist(order: Dict[str, Any], item_ids: List[str], message: str) -> None:
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            raise DomainError(message.format(item_id=item_id))


def _price_difference(order, products, item_ids, new_item_ids, same_item_error: bool):
    if len(item_ids) != len(new_item_ids):
        raise DomainError("The number of items to be exchanged should match")
    diff_price = 0
    variant = None
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        if same_item_error and item_id == new_item_id:
            raise DomainError("The new item id should be different from the old item id")
        item = next((item for item in order["items"] if item["item_id"] == item_id), None)
        if item is None:
            raise DomainError(f"Item {item_id} not found")
        variant = _variant(products, item["product_id"], new_item_id)
        if not variant.get("available"):
            raise DomainError(f"New item {new_item_id} not found or available")
        diff_price += variant["price"] - item["price"]
    return diff_price, variant


def apply_write(kind: str, args: Dict[str, Any], order: Optional[Dict[str, Any]],
                user: Dict[str, Any], products: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """在副本上执行写操作，返回（新订单、新用户、预览信息）；不满足规则时抛 DomainError。"""
    order = copy.deepcopy(order) if order is not None else None
    user = copy.deepcopy(user)
    preview: Dict[str, Any] = {}
    if kind == "return":
        if order["status"] != "delivered":
            raise DomainError("Non-delivered order cannot be returned")
        method = _payment_method(user, args["payment_method_id"])
        if method.get("source") != "gift_card" and args["payment_method_id"] != order["payment_history"][0]["payment_method_id"]:
            raise DomainError("Payment method should be the original payment method")
        _check_items_exist(order, args["item_ids"], "Some item not found")
        order["status"] = "return requested"
        order["return_items"] = sorted(args["item_ids"])
        order["return_payment_method_id"] = args["payment_method_id"]
        preview["refund_amount"] = round(sum(
            next(i["price"] for i in order["items"] if i["item_id"] == item_id) for item_id in args["item_ids"]), 2)
    elif kind == "exchange":
        if order["status"] != "delivered":
            raise DomainError("Non-delivered order cannot be exchanged")
        _check_items_exist(order, args["item_ids"], "Number of {item_id} not found.")
        diff_price, _ = _price_difference(order, products, args["item_ids"], args["new_item_ids"], False)
        diff_price = round(diff_price, 2)
        method = _payment_method(user, args["payment_method_id"])
        if method.get("source") == "gift_card" and method["balance"] < diff_price:
            raise DomainError("Insufficient gift card balance to pay for the price difference")
        order["status"] = "exchange requested"
        order["exchange_items"] = sorted(args["item_ids"])
        order["exchange_new_items"] = sorted(args["new_item_ids"])
        order["exchange_payment_method_id"] = args["payment_method_id"]
        order["exchange_price_difference"] = diff_price
        preview["price_difference"] = diff_price
    elif kind == "cancel":
        if order["status"] != "pending":
            raise DomainError("Non-pending order cannot be cancelled")
        if args["reason"] not in CANCEL_REASONS:
            raise DomainError("Invalid reason")
        refunds = []
        for payment in order["payment_history"]:
            refunds.append({"transaction_type": "refund", "amount": payment["amount"],
                            "payment_method_id": payment["payment_method_id"]})
            method = _payment_method(user, payment["payment_method_id"])
            if method.get("source") == "gift_card":
                method["balance"] += payment["amount"]
                method["balance"] = round(method["balance"], 2)
        order["status"] = "cancelled"
        order["cancel_reason"] = args["reason"]
        order["payment_history"].extend(refunds)
        preview["refund_amount"] = round(sum(r["amount"] for r in refunds), 2)
    elif kind == "modify_items":
        if order["status"] != "pending":
            raise DomainError("Non-pending order cannot be modified")
        _check_items_exist(order, args["item_ids"], "{item_id} not found")
        diff_price, variant = _price_difference(order, products, args["item_ids"], args["new_item_ids"], True)
        method = _payment_method(user, args["payment_method_id"])
        if method.get("source") == "gift_card" and method["balance"] < diff_price:
            raise DomainError("Insufficient gift card balance to pay for the new item")
        order["payment_history"].append({"transaction_type": "payment" if diff_price > 0 else "refund",
                                         "amount": abs(diff_price), "payment_method_id": args["payment_method_id"]})
        if method.get("source") == "gift_card":
            method["balance"] -= diff_price
            method["balance"] = round(method["balance"], 2)
        for item_id, new_item_id in zip(args["item_ids"], args["new_item_ids"]):
            item = next((item for item in order["items"] if item["item_id"] == item_id), None)
            if item is None:
                raise DomainError(f"Item {item_id} not found")
            # 官方实现使用循环最后一个新商品的价格与选项；为状态一致原样保留。
            item["item_id"] = new_item_id
            item["price"] = variant["price"]
            item["options"] = copy.deepcopy(variant["options"])
        order["status"] = "pending (item modified)"
        preview["price_difference"] = round(diff_price, 2)
    elif kind == "modify_address":
        if "pending" not in order["status"]:
            raise DomainError("Non-pending order cannot be modified")
        order["address"] = {field: args[field] for field in ADDRESS_FIELDS}
    elif kind == "modify_payment":
        if "pending" not in order["status"]:
            raise DomainError("Non-pending order cannot be modified")
        method = _payment_method(user, args["payment_method_id"])
        history = order["payment_history"]
        if len(history) != 1 or history[0]["transaction_type"] != "payment":
            raise DomainError("There should be exactly one payment for a pending order")
        if history[0]["payment_method_id"] == args["payment_method_id"]:
            raise DomainError("The new payment method should be different from the current one")
        amount = history[0]["amount"]
        if method.get("source") == "gift_card" and method["balance"] < amount:
            raise DomainError("Insufficient gift card balance to pay for the order")
        old_id = history[0]["payment_method_id"]
        history.extend([{"transaction_type": "payment", "amount": amount, "payment_method_id": args["payment_method_id"]},
                        {"transaction_type": "refund", "amount": amount, "payment_method_id": old_id}])
        if method.get("source") == "gift_card":
            method["balance"] -= amount
            method["balance"] = round(method["balance"], 2)
        old_method = _payment_method(user, old_id)
        if old_method.get("source") == "gift_card":
            old_method["balance"] += amount
            old_method["balance"] = round(old_method["balance"], 2)
        preview["amount"] = amount
    elif kind == "modify_user_address":
        user["address"] = {field: args[field] for field in ADDRESS_FIELDS}
    else:
        raise DomainError(f"Unknown write kind: {kind}")
    return order, user, preview


def canonical_args(kind: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """规范化并校验参数形状；列表保持原顺序（官方逐位置配对新旧商品）。"""
    def text(name: str) -> str:
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            raise DomainError(f"{name} must be a non-empty string")
        return value.strip()

    def items(name: str) -> List[str]:
        value = args.get(name)
        if not isinstance(value, list) or not value or len(value) > 20 or not all(
                isinstance(v, str) and v.strip() for v in value):
            raise DomainError(f"{name} must be a non-empty list of item ids")
        return [v.strip() for v in value]

    if kind == "modify_user_address":
        result: Dict[str, Any] = {}
    else:
        order_id = text("order_id")
        result = {"order_id": order_id if order_id.startswith("#") else "#" + order_id.upper()}
    if kind in {"return", "exchange", "modify_items"}:
        result["item_ids"] = items("item_ids")
    if kind in {"exchange", "modify_items"}:
        result["new_item_ids"] = items("new_item_ids")
    if kind in {"return", "exchange", "modify_items", "modify_payment"}:
        result["payment_method_id"] = text("payment_method_id")
    if kind == "cancel":
        result["reason"] = text("reason")
    if kind in {"modify_address", "modify_user_address"}:
        for field in ADDRESS_FIELDS:
            value = args.get(field, "")
            if not isinstance(value, str) or (field != "address2" and not value.strip()):
                raise DomainError(f"{field} must be a string")
            result[field] = value
    return result
