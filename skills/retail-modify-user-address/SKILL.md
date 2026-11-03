---
name: modify_user_address
description: Change the user's default profile address (not an order's shipping address).
tools: modify_user_address
policy_sections:
agents: retail_on_demand
enabled: true
---

# Playbook

1. This changes the default address in the user's profile only; it does not change the shipping address of existing orders. If the user also wants a pending order shipped elsewhere, that is a separate action (modify_pending_order skill).
2. Collect the complete new address (address1, address2, city, state, country, zip); ask for any missing field rather than guessing.
3. Call modify_user_address to prepare the details, show them exactly, and wait for explicit confirmation in a later message before calling it again with identical arguments.
