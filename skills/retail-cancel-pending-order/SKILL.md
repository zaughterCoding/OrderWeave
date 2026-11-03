---
name: cancel_pending_order
description: Cancel a whole pending order (allowed reasons, status check, refund timing).
tools: cancel_pending_order
policy_sections: Cancel pending order
agents: retail_on_demand
enabled: true
---

# Playbook

1. Read the order with get_order_details and check that its status is exactly `pending`. A `pending (items modified)`, processed, delivered or cancelled order cannot be cancelled; say so and offer what the policy allows.
2. Ask for the cancellation reason if the user has not given one, and map it only to `no longer needed` or `ordered by mistake`. Do not choose a reason for the user; if their reason fits neither, explain the two accepted reasons.
3. Cancellation applies to the whole order. If the user wants to drop only some items, explain that partial cancellation is not available.
4. Call cancel_pending_order to prepare the details, show them exactly, and wait for the user's explicit confirmation in a later message before calling it again with identical arguments.
5. After it is carried out, state the refund timing from the policy (gift card immediately, otherwise 5 to 7 business days) without claiming the refund has arrived.
