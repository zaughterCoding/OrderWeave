---
name: modify_pending_order
description: Change a pending order's items, shipping address or payment method (once-only item change, same-product variants, price difference).
tools: modify_pending_order_items, modify_pending_order_address, modify_pending_order_payment
policy_sections: Modify pending order
agents: retail_on_demand
enabled: true
---

# Playbook

1. Read the order with get_order_details and check that its status is exactly `pending`.
2. Items:
   - Item modification can happen only once per order and afterwards the order cannot be modified or cancelled again, so first ask the user to list every item they want to change and confirm the list is complete.
   - For each item, read the product with get_product_details and pick the variant that matches the options the user asked for; it must be the same product, a different option set, and `available`. If several variants fit, ask the user to choose instead of choosing for them.
   - Ask which payment method pays or receives the price difference; a gift card must have enough balance to cover the difference.
   - Pass item_ids and new_item_ids as two lists in the same order.
3. Address: collect the complete new address (address1, address2, city, state, country, zip). If the user also wants their default address changed, that is a separate action (modify_user_address skill).
4. Payment: the new method must be a single existing method of the user, different from the current one; a gift card must cover the order total.
5. Call the matching tool to prepare the details, show them exactly, and wait for explicit confirmation in a later message before calling it again with identical arguments. Handle one action at a time.
