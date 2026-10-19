"""The negotiation report as plain text, attached by code to final answers so no LLM can drop or reword it.

Two parts, each built from the negotiator's structured outcome:
- a side-by-side table of each seller's candidate (specs, list price, negotiated price, quote) and the
  negotiator's pick with its reason;
- the chat: every offer that reached a seller, in order, as
  {"seller", "round", "buyer_offer", "buyer_message", "seller_action", "seller_price", "seller_message"}.
Prices come from the sellers' replies and quotes.
"""
from typing import Iterable

COMPARISON_HEADER = "Comparison (specs and negotiated prices):"
CHAT_HEADER = "Negotiation chat (offers and seller replies):"
ACTIONS = {"accept": "accepted at", "counter": "countered at", "hold": "held at"}
DASH = "\u2013"


def money(value) -> str:
    return DASH if value is None else f"${float(value):.2f}"


def spec_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return DASH if value is None else str(value)


def cell(text) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def format_comparison(outcome: dict) -> str:
    offers = (outcome or {}).get("offers") or []
    if not offers:
        return ""
    sellers = [o["seller"] for o in offers]
    spec_keys: list[str] = []
    for o in offers:
        for key in (o.get("product") or {}).get("specs") or {}:
            if key not in spec_keys:
                spec_keys.append(key)

    def row(label: str, values: list) -> str:
        return "| " + " | ".join(cell(v) for v in [label, *values]) + " |"

    def column(o: dict, field: str) -> str:
        product = o.get("product") or {}
        if not o.get("available"):
            return f"unavailable ({o.get('error', 'no quote')})" if field == "product" else DASH
        if field == "product":
            return f"{product.get('name', DASH)} ({product.get('brand', DASH)})"
        if field == "saved":
            return f"{money(o['savings'])} ({o['savings_pct']}%)"
        if field == "quote":
            return f"{o['quote_id']} ({o['status']})"
        if field in ("list_price", "final_price"):
            return money(o.get(field))
        return spec_value(product.get(field))

    lines = [COMPARISON_HEADER, row("", sellers), "|" + "---|" * (len(sellers) + 1)]
    for label, field in (("Product", "product"), ("SKU", "sku"), ("Model no.", "mpn")):
        lines.append(row(label, [column(o, field) for o in offers]))
    for key in spec_keys:
        lines.append(row(key, [spec_value(((o.get("product") or {}).get("specs") or {}).get(key))
                               if o.get("available") else DASH for o in offers]))
    for label, field in (("Rating", "rating"), ("List price", "list_price"), ("Negotiated price", "final_price"),
                         ("Saved", "saved"), ("Quote", "quote")):
        lines.append(row(label, [column(o, field) for o in offers]))

    qty = outcome.get("quantity") or 1
    same = "the same model and configuration at every seller" if outcome.get("same_model") \
        else "different models or configurations, so compare the specs, not only the price"
    lines += ["", f"These are {same}. Quote only: nothing was bought."]
    budget = outcome.get("max_budget")
    winner = next((o for o in offers if o["seller"] == outcome.get("winner")), None)
    if winner:
        note = "" if budget is None else f", within the {money(budget)} budget"
        total = f" ({money(winner['total'])} for {qty})" if qty > 1 else ""
        lines.append(f"Negotiator's pick: {winner['seller']}, {winner['product'].get('name')} "
                     f"({winner['product'].get('sku')}) at {money(winner['final_price'])}{total}{note}.")
        if outcome.get("winner_reason"):
            lines.append(f"Why: {outcome['winner_reason']}")
        if outcome.get("decided_by") and outcome["decided_by"] != "negotiator":
            lines.append(f"(Chosen by {outcome['decided_by']}.)")
    elif any(o.get("available") for o in offers):
        lines.append("Negotiator's pick: none. " + (outcome.get("winner_reason") or "No offer fits the request."))
    else:
        lines.append("No seller gave a usable quote.")
    return "\n".join(lines)


def format_chat(outcome: dict) -> str:
    chat = (outcome or {}).get("chat") or []
    if not chat:
        return ""
    lines = [CHAT_HEADER]
    for turn in chat:
        offer = f"- Negotiator -> {turn['seller']} (offer {int(turn['round'])}): ${float(turn['buyer_offer']):.2f}"
        if turn.get("buyer_message"):
            offer += f' "{turn["buyer_message"]}"'
        reply = (f"  {turn['seller']}: {ACTIONS.get(turn.get('seller_action'), 'answered at')} "
                 f"${float(turn['seller_price']):.2f}")
        if turn.get("seller_message"):
            reply += f' "{turn["seller_message"]}"'
        lines += [offer, reply]
    return "\n".join(lines)


def attach_report(text: str, outcomes: Iterable[dict]) -> str:
    """Append each negotiation's comparison and chat, skipping a part the text already carries."""
    text = text or ""
    outcomes = list(outcomes or [])
    parts = []
    if COMPARISON_HEADER not in text:
        parts += [format_comparison(o) for o in outcomes]
    if CHAT_HEADER not in text:
        parts += [format_chat(o) for o in outcomes]
    parts = [p for p in parts if p]
    return "\n\n".join([text.rstrip(), *parts]) if parts else text
