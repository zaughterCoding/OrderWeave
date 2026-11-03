---
name: return_delivered_order
description: Return items from a delivered order (item list, refund destination limits).
tools: return_delivered_order_items
policy_sections: Return delivered order
agents: retail_on_demand
enabled: true
---

# Playbook

1. Read the order with get_order_details and check that its status is exactly `delivered`. If the user does not know the order id, read get_user_details and inspect their orders to locate the items.
2. Collect every item to return from that order. A repeated item_id means more than one unit. Items from different orders are separate returns.
3. Ask where the refund should go: only the order's original payment method or a gift card the user already owns. Explain the allowed choices if the user names anything else.
4. Call return_delivered_order_items to prepare the details, show them exactly, and wait for explicit confirmation in a later message before calling it again with identical arguments.
5. Afterwards the order is `return requested`; the user receives an email about returning the items. Do not claim the refund, the email or warehouse receipt has already happened.
