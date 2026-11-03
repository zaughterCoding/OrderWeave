---
name: exchange_delivered_order
description: Exchange items of a delivered order for other variants of the same product (once per order, price difference).
tools: exchange_delivered_order_items
policy_sections: Exchange delivered order
agents: retail_on_demand
enabled: true
---

# Playbook

1. Read the order with get_order_details and check that its status is exactly `delivered`.
2. An exchange can happen only once per order, so ask the user to list every item to exchange and confirm the list is complete.
3. For each item, read the product with get_product_details and pick the variant matching the requested options: same product, different option set, `available`. If several variants fit or the request is ambiguous, ask the user rather than choosing.
4. Ask which payment method pays or receives the price difference; a gift card must have enough balance to cover the difference.
5. Pass item_ids and new_item_ids as two lists in the same order. Call exchange_delivered_order_items to prepare the details, show them exactly, and wait for explicit confirmation in a later message before calling it again with identical arguments.
6. Afterwards the order is `exchange requested`; no new order is needed. Do not claim the exchange has shipped or the difference was settled.
