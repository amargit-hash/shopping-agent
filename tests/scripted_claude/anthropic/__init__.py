"""Stand-in for the `anthropic` package used by the end-to-end test: deterministic, scripted
"Claude" decisions chosen by which tools the caller offers. No network, no API key.

Put the parent folder first on PYTHONPATH to use it (tests/e2e_negotiation.py does this).
Each brain only reads what the real model would see (system prompt, messages, tool results).
"""
import json
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

__version__ = "scripted"


class APIError(Exception):
    message = "scripted error"


@dataclass
class Block:
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None

    def model_dump(self, mode: str = "json", exclude_none: bool = False) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None} if exclude_none else data


@dataclass
class Response:
    content: list
    stop_reason: str


def say(text: str) -> Response:
    return Response([Block("text", text=text)], "end_turn")


def use(name: str, **arguments: Any) -> Response:
    return Response([Block("tool_use", id=f"toolu_{uuid.uuid4().hex[:12]}", name=name, input=arguments)], "tool_use")


def field(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def blocks(message: dict) -> list:
    content = message.get("content")
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def turn_calls(messages: list[dict]) -> list[tuple[str, dict, str]]:
    """(tool name, input, result text) for every tool call since the last plain user message."""
    start = 0
    for i, message in enumerate(messages):
        if message["role"] == "user" and not any(field(b, "type") == "tool_result" for b in blocks(message)):
            start = i
    calls, by_id = [], {}
    for message in messages[start:]:
        for block in blocks(message):
            if field(block, "type") == "tool_use":
                by_id[field(block, "id")] = (field(block, "name"), field(block, "input") or {})
            elif field(block, "type") == "tool_result":
                name, arguments = by_id.get(field(block, "tool_use_id"), ("?", {}))
                content = field(block, "content")
                if isinstance(content, list):
                    content = "".join(field(c, "text") or "" for c in content)
                calls.append((name, arguments, str(content)))
    return calls


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            texts = [field(b, "text") for b in blocks(message) if field(b, "type") == "text"]
            if texts:
                return "\n".join(t for t in texts if t)
    return ""


def first_user_text(messages: list[dict]) -> str:
    return next((field(b, "text") for b in blocks(messages[0]) if field(b, "type") == "text"), "") if messages else ""


def money(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern.replace(r"([\d.]+)", r"(\d+(?:\.\d+)?)"), text)
    return float(match.group(1)) if match else None


def seller(system: str, messages: list[dict]) -> Response:
    """Counter halfway on the first offers; accept anything at or above the floor on the last one."""
    prompt = first_user_text(messages)
    floor = money(r"may accept is \$([\d.]+)", system)
    ask = money(r"current asking price is \$([\d.]+)", system)
    offer = money(r"Buyer now offers \$([\d.]+)", prompt)
    round_no = int(re.search(r"buyer offer (\d+) of (\d+)", system).group(1))
    last = round_no >= int(re.search(r"buyer offer \d+ of (\d+)", system).group(1))
    if offer >= floor and (last or offer >= ask * 0.97):
        return use("respond", action="accept", message="Deal, that works for us.")
    if last:
        return use("respond", action="counter", price=floor, message="This is our best and final price.")
    return use("respond", action="counter", price=round(max((offer + ask) / 2, floor), 2),
               message="We can meet you partway.")


CANDIDATE_RE = re.compile(r"^(\d+)\. (.+?): candidate (\S+) (.+?) \((\S+)\), list \$([\d.]+), "
                          r"current price \$([\d.]+), (\w+), (\d+) offer\(s\) left$", re.M)
NO_CANDIDATE_RE = re.compile(r"^(\d+)\. (.+?): no candidate yet$", re.M)
SHORTLIST_RE = re.compile(r"^Seller (\d+), (.+?):\n((?:  - .*\n?)*)", re.M)


def shortlists(opening: str) -> dict[int, list[dict]]:
    out = {}
    for number, _name, rows in SHORTLIST_RE.findall(opening):
        items = []
        for row in rows.strip("\n").splitlines():
            sku, name, _brand, mpn, price, rating, specs = [c.strip() for c in row.strip()[2:].split("|")]
            items.append({"sku": sku, "name": name, "mpn": mpn, "price": float(price.lstrip("$")),
                          "rating": float(rating.split()[-1]), "specs": specs})
        out[int(number)] = items
    return out


def buyer(messages: list[dict]) -> Response:
    """Pick the top-rated affordable product at each seller, bid down the priciest open seller each round,
    then pick the lowest negotiated price within budget and explain the spec trade-off."""
    opening, state = first_user_text(messages), last_user_text(messages)
    budget = money(r"maximum unit price \(private\): \$([\d.]+)", opening)
    waiting = NO_CANDIDATE_RE.findall(state)
    if waiting:
        number = int(waiting[0][0])
        tried = sum(1 for name, args, _ in turn_calls(messages)
                    if name == "choose_candidate" and int(args.get("seller", 0)) == number)
        options = sorted(shortlists(opening).get(number, []),
                         key=lambda p: (budget is not None and p["price"] > budget * 1.1, -p["rating"]))
        if tried < len(options):
            pick = options[tried]
            return use("choose_candidate", seller=number, sku=pick["sku"],
                       reason=f"Top-rated match for the request at rating {pick['rating']}.")
        return use("finish", winner=0, reason=f"Seller {number} has no product that fits the request.")
    sellers = CANDIDATE_RE.findall(state)
    open_ones = [s for s in sellers if s[7] == "open" and int(s[8]) > 0]
    if open_ones:
        lowest = min(float(s[6]) for s in sellers)
        target = max(open_ones, key=lambda s: float(s[6]))
        current = float(target[6])
        price = round(min(lowest * 0.96, current * 0.96), 2)
        note = f" Another seller quotes ${lowest:.2f} for a comparable model." if lowest < current else ""
        return use("make_offer", seller=int(target[0]), price=price, message=f"Could you do ${price:.2f}?{note}")
    fitting = [s for s in sellers if budget is None or float(s[6]) <= budget]
    if not fitting:
        return use("finish", winner=0, reason="No negotiated price is within the budget.")
    best = min(fitting, key=lambda s: float(s[6]))
    others = [s for s in sellers if s is not best]
    same = all(s[4] == best[4] for s in others)
    comparison = "the same model and configuration" if same else "a different model or configuration"
    gap = ", ".join(f"${float(s[6]) - float(best[6]):.2f} less than {s[1]}'s {s[3]}" for s in others)
    return use("finish", winner=int(best[0]),
               reason=f"{best[1]}'s {best[3]} is the best value at ${float(best[6]):.2f}: {gap}, "
                      f"and the others are {comparison}.")


def choose(messages: list[dict]) -> Response:
    listing = first_user_text(messages)
    if listing.startswith("My job: expert specialist advice"):
        return choose_specialist(listing)
    for number, name in re.findall(r"^(\d+)\. (.+?) \(index score", listing, re.M):
        if "shopping" in name.lower():
            return use("choose_agent", index=int(number), reason=f"{name} shops and can get negotiated quotes.")
    return use("choose_agent", index=1, reason="best match")


def choose_specialist(listing: str) -> Response:
    """The candidate whose description and skills share the most words with the request."""
    request = listing.split("\nRequests:\n- ", 1)[-1].split("\n", 1)[0].lower()
    words = {w for w in re.findall(r"[a-z]{4,}", request)} - {"under", "with", "best"}
    blocks_ = re.split(r"\n(?=\d+\. )", listing.split("Candidates from the NANDA Index:\n", 1)[-1])
    scored = []
    for block in blocks_:
        head = re.match(r"(\d+)\. (.+?) \(index score", block)
        if head:
            text = block.lower()
            scored.append((sum(1 for w in words if w.rstrip("s") in text), -int(head.group(1)), head))
    if not scored:
        return use("choose_agent", index=0, reason="no candidates")
    _, _, head = max(scored)
    return use("choose_agent", index=int(head.group(1)), reason=f"{head.group(2)} specialises in this request.")


def specialist(system: str, messages: list[dict]) -> Response:
    """One recommend_products call within the budget, then a reply naming the picks."""
    request = first_user_text(messages)
    calls = turn_calls(messages)
    if not calls:
        category = next((c for c in ("laptops", "headphones", "cameras", "watches", "books")
                         if c.rstrip("s") in request.lower()), None)
        return use("recommend_products", budget=money(r"under \$?([\d.]+)", request) or 100000, category=category,
                   preferences=re.findall(r"[a-z]{5,}", request.lower())[:4], limit=3)
    picks = json.loads(calls[-1][2]).get("recommendations", [])
    name = re.search(r"You are the (.+?),", system).group(1)
    if not picks:
        return say(f"{name}: nothing in the catalog fits that budget.")
    lines = [f"- {p['id']} {p['name']} at ${p['price']:.2f}: well rated ({p['rating']})." for p in picks]
    return say(f"{name} recommends:\n" + "\n".join(lines) + f"\nTop pick: {picks[0]['id']}.")


def planner(messages: list[dict]) -> Response:
    calls = turn_calls(messages)
    if not calls:
        request = first_user_text(messages).removeprefix("Request: ").strip()
        return use("ask_shopping_agent", message=f"{request}. Have a negotiator pick the best product at each "
                                                 "seller, negotiate every price, and report the best deal. Quote only.")
    return say(f"Summary from the shopping agent:\n{calls[-1][2]}")


def marketplace(messages: list[dict]) -> Response:
    request = first_user_text(messages)
    calls = turn_calls(messages)
    if calls and (calls[-1][2].startswith(calls[-1][0] + " failed") or '"error"' in calls[-1][2][:20]):
        return say(f"I couldn't finish: {calls[-1][2]}")
    done = [c[0] for c in calls]
    results = {c[0]: c for c in calls}
    budget = money(r"under \$?([\d.]+)", request)
    if "find_sellers" not in done:
        return use("find_sellers")
    if "find_agents" not in done:
        return use("find_agents", task="negotiate the best deal for a product with several sellers")
    if "delegate_negotiation" not in done:
        category = next((c for c in ("laptops", "headphones", "phones", "cameras", "tablets", "watches")
                         if c.rstrip("s") in request.lower()), None)
        keywords = "gaming" if "gaming" in request.lower() else ""
        candidates = json.loads(results["find_agents"][2]).get("candidates") or []
        agent = next((c for c in candidates if "negotiat" in c["name"].lower()), None)
        if agent is None:
            return say("The NANDA Index has no negotiator agent.")
        wanted = request.split(". ")[0]
        return use("delegate_negotiation", agent=agent["identifier"], request=wanted, category=category,
                   keywords=keywords, max_budget=budget)
    return say(json.loads(results["delegate_negotiation"][2])["negotiator_reply"])


class Messages:
    async def create(self, *, system: str = "", tools: Optional[list] = None, messages: list, **_: Any) -> Response:
        names = {t["name"] for t in tools or []}
        if "respond" in names:
            return seller(system, messages)
        if "finish" in names:
            return buyer(messages)
        if "choose_agent" in names:
            return choose(messages)
        if "ask_shopping_agent" in names:
            return planner(messages)
        if "delegate_negotiation" in names:
            return marketplace(messages)
        if "recommend_products" in names and "add_to_cart" not in names:
            return specialist(system, messages)
        return say("(scripted Claude has no plan for this request)")


class AsyncAnthropic:
    def __init__(self, *_, **__) -> None:
        self.messages = Messages()
