"""Specialist advisor agents (laptops, gifts, travel gear) as A2A servers, found through the NANDA Index.

    gift planner --NANDA agentic search--> specialist cards --A2A--> specialist --MCP (read-only)--> store

Each process runs one profile. The specialist finds its store through NANDA at startup (NANDA_LOCATOR
or NANDA_SEARCH; PRODUCT_SERVER_URL otherwise) and only uses the store's read-only catalog tools:
it advises, compares and recommends, but has no cart, never checks out and never negotiates.
With ANTHROPIC_API_KEY set, Claude writes the advice from the catalog tools; otherwise a rule-based
advisor recommends the best-rated matches within the budget.

    NANDA_INDEX_URL=http://localhost:3001 NANDA_LOCATOR="urn:air:..." python specialist_agent.py --profile laptop  # :8085
    ... --profile gift    # :8086
    ... --profile travel  # :8089

Register each one in the index (the description and tags come from its card):
    python nanda_register.py --kind specialist --card-url http://localhost:8085/.well-known/agent-card.json
"""
import argparse
import asyncio
import json
import os
import re
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic
import uvicorn
from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill
from a2a.utils.errors import UnsupportedOperationError
from starlette.applications import Starlette

from a2a_link import request_metadata
from claude_agent import DEFAULT_MODEL
from nanda_discovery import DiscoveryError, discover, mcp_session
from shopping_agent import DEFAULT_SERVER_URL, MAX_PRICE_RE, PRODUCT_ID_RE
from tracing import TracingSession, base_of, current_sink, current_trace, open_sink, record, record_lane

# The store's catalog tools that don't change anything; negotiation tools are left out on purpose.
READ_ONLY_TOOLS = {"list_categories", "search_products", "get_product", "compare_products", "recommend_products",
                   "check_availability"}
MAX_TOOL_ROUNDS = 8
PICKS = 3
NO_BUDGET = 100_000.0
ACTION_RE = re.compile(r"\b(?:add|put)\b.*\bcart\b|\bcheck ?out\b|\bplace (?:an |the )?order\b|\bnegotiat", re.I)
WORD_RE = re.compile(r"[a-z][a-z-]{2,}")
STOP_WORDS = {"the", "and", "for", "with", "under", "below", "best", "good", "great", "want", "need", "some", "that",
              "this", "from", "less", "than", "within", "max", "about", "would", "like", "find", "get", "buy",
              "recommend", "suggest", "advice", "what", "which", "should", "can", "you", "your", "me", "my", "who"}


@dataclass
class Profile:
    key: str
    name: str
    email: str
    port: int
    description: str
    skill_name: str
    skill_description: str
    tags: list[str]
    examples: list[str]
    focus: str
    categories: list[str] = field(default_factory=list)  # empty: any category
    preferences: list[str] = field(default_factory=list)  # always added to the shopper's own words


PROFILES = {
    "laptop": Profile(
        key="laptop", name="Laptop Advisor Agent", email="laptop-advisor@nanda.local", port=8085,
        description="Laptop specialist agent: expert advice on choosing a laptop or notebook computer for work, "
                    "school, gaming, programming or video editing within a budget. Compares laptop specs such as "
                    "RAM, storage, screen size and weight. Read-only advice: no cart, no checkout.",
        skill_name="Laptop buying advice",
        skill_description="Recommends and compares in-stock laptops for a use and budget, explaining the spec "
                          "trade-offs (RAM, storage, screen, weight).",
        tags=["laptops", "laptop", "notebook", "computers", "advice", "recommendations", "comparison", "read-only"],
        examples=["a laptop for video editing under 1200", "a light laptop for a student under 700"],
        focus="You only advise on laptops (the laptops category); a tablet or a phone is not a laptop. Explain the "
              "spec trade-offs (ram_gb, storage_gb, screen_in, weight_kg) for the stated use.",
        categories=["laptops"],
    ),
    "gift": Profile(
        key="gift", name="Gift Ideas Agent", email="gift-ideas@nanda.local", port=8086,
        description="Gift ideas specialist agent: expert advice on gift ideas and presents for birthdays, "
                    "holidays, anniversaries and thank-yous, for mom, dad, kids, friends or coworkers, within a "
                    "budget. Read-only advice: no cart, no checkout.",
        skill_name="Gift ideas",
        skill_description="Suggests well-rated, in-stock gift ideas from any category for a recipient, an "
                          "occasion and a budget, with a reason for each.",
        tags=["gifts", "gift-ideas", "presents", "birthday", "holidays", "advice", "recommendations", "read-only"],
        examples=["a birthday gift for my mom under 50", "a present for a 10 year old who likes games under 40"],
        focus="Suggest gift ideas from any category that suit the recipient and occasion; prefer well-reviewed, "
              "in-stock products, and give one short reason per idea.",
    ),
    "travel": Profile(
        key="travel", name="Travel Gear Agent", email="travel-gear@nanda.local", port=8089,
        description="Travel gear specialist agent: expert advice on travel gear for trips, flights and vacations: "
                    "noise-cancelling headphones, portable chargers, action cameras, e-readers and water bottles, "
                    "for packing light within a budget. Read-only advice: no cart, no checkout.",
        skill_name="Travel gear advice",
        skill_description="Recommends portable, travel-friendly products (headphones, chargers, cameras, "
                          "e-readers, bottles) for a trip and budget.",
        tags=["travel", "travel-gear", "trips", "flights", "luggage", "advice", "recommendations", "read-only"],
        examples=["headphones for a long flight under 200", "gear for a hiking trip under 150"],
        focus="Advise on travel-friendly products: prefer items tagged travel, portable or lightweight, and "
              "mention weight and battery life when the catalog has them.",
        categories=["headphones", "electronics", "cameras", "fitness"],
        preferences=["travel", "portable", "lightweight"],
    ),
}

PROMPT = """\
You are the {name}, a specialist advisor with read-only access to a store's product catalog.
{focus}
Use the catalog tools to look up real products; never invent products, prices, specs or stock.
Recommend up to 3 in-stock products within the shopper's budget, best first. For each one give the
product ID (like P003), name, price and one sentence on why it fits. Then name your top pick.
You can't add anything to a cart, check out or negotiate: if asked, say that a shopping agent does that.
If nothing fits the budget, say so and name the cheapest suitable product.
Reply in short plain text; simple "-" lists are fine."""


def parse_budget(text: str) -> Optional[float]:
    match = MAX_PRICE_RE.search(text.lower())
    return float(match.group(1)) if match else None


def shopper_words(text: str) -> list[str]:
    """Words of the request that can match product tags or descriptions (no budget, no filler)."""
    words = WORD_RE.findall(re.sub(MAX_PRICE_RE, " ", text.lower()))
    return list(dict.fromkeys(w for w in words if w not in STOP_WORDS))


def detect_categories(profile: Profile, text: str, known: list[str]) -> list[str]:
    """Catalog categories named in the request ('headphones', 'a laptop'), limited to the profile's own."""
    allowed = profile.categories or known
    lower = text.lower()
    named = [c for c in allowed if re.search(rf"\b{re.escape(c.rstrip('s'))}s?\b", lower)]
    return named or list(profile.categories)


def pick_view(p: dict) -> dict:
    return {k: p.get(k) for k in ("id", "name", "brand", "category", "price", "rating", "reviews", "specs")
            if p.get(k) is not None}


def product_line(number: int, p: dict) -> str:
    specs = ", ".join(f"{k} {v}" for k, v in (p.get("specs") or {}).items())
    return (f"{number}. {p['id']} {p['name']} ({p.get('brand')}): ${float(p['price']):.2f}, rating {p.get('rating')}"
            + (f", {specs}" if specs else ""))


async def call_json(session: Any, tool: str, **arguments: Any) -> dict:
    result = await session.call_tool(tool, {k: v for k, v in arguments.items() if v is not None})
    try:
        return json.loads("".join(getattr(c, "text", "") for c in result.content) or "{}")
    except ValueError:
        return {"error": f"{tool} returned something that isn't JSON"}


class Advisor:
    """Answers one request with read-only catalog calls; returns (reply, advice metadata)."""

    def __init__(self, profile: Profile, session: Any, use_claude: bool, model: str = DEFAULT_MODEL) -> None:
        self.profile, self.session, self.use_claude, self.model = profile, session, use_claude, model
        self.categories: list[str] = []

    async def known_categories(self) -> list[str]:
        if not self.categories:
            self.categories = sorted((await call_json(self.session, "list_categories")).get("categories", {}))
        return self.categories

    async def answer(self, text: str) -> tuple[str, dict]:
        text = text.strip()
        advice = {"agent": self.profile.name, "profile": self.profile.key, "request": text,
                  "budget": parse_budget(text), "picks": [], "read_only": True}
        if not text:
            return f"Ask me for advice, e.g. '{self.profile.examples[0]}'.", advice
        if ACTION_RE.search(text):
            reply = (f"I'm the {self.profile.name}: I only give read-only advice, so I can't add to a cart, check "
                     "out or negotiate. Ask a shopping agent for that.")
            return reply, {**advice, "refused": "needs a cart or negotiation skill"}
        if self.use_claude:
            try:
                reply, picks = await self.claude_answer(text)
                return reply, {**advice, "picks": picks, "mode": "Claude"}
            except anthropic.APIError as e:
                note = f"(Claude was unavailable: {getattr(e, 'message', e)}; rule-based advice follows.)\n"
                reply, picks = await self.rule_answer(text, advice["budget"])
                return note + reply, {**advice, "picks": picks, "mode": "rule-based"}
        reply, picks = await self.rule_answer(text, advice["budget"])
        return reply, {**advice, "picks": picks, "mode": "rule-based"}

    async def rule_answer(self, text: str, budget: Optional[float]) -> tuple[str, list[dict]]:
        categories = detect_categories(self.profile, text, await self.known_categories())
        preferences = list(dict.fromkeys(shopper_words(text) + self.profile.preferences))
        found: dict[str, dict] = {}
        for category in categories or [None]:
            data = await call_json(self.session, "recommend_products", budget=budget or NO_BUDGET, category=category,
                                   preferences=preferences, limit=PICKS)
            for p in data.get("recommendations", []):
                found.setdefault(p["id"], p)
        # Products that match more of the shopper's words first, then the better rated.
        ranked = sorted(found.values(), key=lambda p: (len(p.get("matched_preferences") or []), p.get("rating", 0),
                                                       p.get("reviews", 0)), reverse=True)[:PICKS]
        within = f" within ${budget:.2f}" if budget else ""
        if not ranked:
            return (f"{self.profile.name}: I found nothing in stock{within} for \u201c{text}\u201d. "
                    "Try a higher budget."), []
        details = await call_json(self.session, "compare_products", product_ids=[p["id"] for p in ranked])
        specs = {p["id"]: p.get("specs") for p in details.get("comparison", [])}
        picks = [pick_view({**p, "specs": specs.get(p["id"]) or p.get("specs")}) for p in ranked]
        top = picks[0]
        lines = [f"{self.profile.name}'s advice for \u201c{text}\u201d{within} (read-only, nothing was added to a cart):",
                 *[product_line(i, p) for i, p in enumerate(picks, 1)],
                 f"Top pick: {top['id']} {top['name']} at ${float(top['price']):.2f}, the best match for your words "
                 f"and rating {top.get('rating')}."]
        return "\n".join(lines), picks

    async def claude_answer(self, text: str) -> tuple[str, list[dict]]:
        listed = await self.session.list_tools()
        tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                 for t in listed.tools if t.name in READ_ONLY_TOOLS]
        client = anthropic.AsyncAnthropic()
        system = PROMPT.format(name=self.profile.name, focus=self.profile.focus)
        messages: list[dict] = [{"role": "user", "content": text}]
        seen: dict[str, dict] = {}
        reply = "I had to stop after too many catalog lookups; try a narrower request."
        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.messages.create(model=self.model, max_tokens=1500, system=system, tools=tools,
                                                    messages=messages)
            messages.append({"role": "assistant",
                             "content": [b.model_dump(mode="json", exclude_none=True) for b in response.content]})
            uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not uses:
                reply = "".join(b.text for b in response.content if b.type == "text").strip()
                break
            results = []
            for block in uses:
                if block.name not in READ_ONLY_TOOLS:  # Claude only sees these, but never trust the call
                    output, is_error = f"{block.name} is not available to an advisor", True
                else:
                    try:
                        data = await call_json(self.session, block.name, **(block.input or {}))
                        output, is_error = json.dumps(data), "error" in data
                        remember(seen, data)
                    except Exception as e:
                        output, is_error = f"{block.name} failed: {e}", True
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        # Picks are products the catalog returned and the reply names, never IDs Claude made up.
        ids = list(dict.fromkeys(i.upper() for i in PRODUCT_ID_RE.findall(reply)))
        return reply, [pick_view(seen[i]) for i in ids if i in seen][:PICKS]


def remember(seen: dict, data: Any) -> None:
    """Index every product in a catalog tool result by ID."""
    if isinstance(data, dict):
        if isinstance(data.get("id"), str) and "price" in data:
            seen.setdefault(data["id"].upper(), data)
        for value in data.values():
            remember(seen, value)
    elif isinstance(data, list):
        for value in data:
            remember(seen, value)


def build_agent_card(profile: Profile, base_url: str) -> AgentCard:
    return AgentCard(
        name=profile.name,
        description=profile.description,
        version="1.0.0",
        provider=AgentProvider(organization="Project NANDA demo", url=base_url),
        supported_interfaces=[
            AgentInterface(url=f"{base_url}/", protocol_binding="JSONRPC", protocol_version="1.0"),
            AgentInterface(url=f"{base_url}/", protocol_binding="JSONRPC", protocol_version="0.3"),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id=f"advice.{profile.key}", name=profile.skill_name,
                           description=profile.skill_description + " Read-only: it can't add to a cart, check out "
                                                                   "or negotiate.",
                           tags=profile.tags, examples=profile.examples)],
    )


class SpecialistExecutor(AgentExecutor):
    def __init__(self, profile: Profile, base_url: str) -> None:
        self.profile, self.lane = profile, base_of(base_url)
        self.base_url = base_url
        self.session: Optional[TracingSession] = None
        self.store_source = self.store_endpoint = self.store_name = ""
        self.use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        print(f"[{self.profile.key}] {context.context_id[:8]} <- {text!r}")
        trace: list[dict] = []
        token = current_trace.set(trace)
        sink = open_sink(request_metadata(context).get("trace_sink"))
        sink_token = current_sink.set(sink)
        advice: dict = {}
        try:
            record_lane(self.base_url, "a2a", self.profile.name)
            if self.store_source.startswith("NANDA"):
                record(kind="startup", src=self.lane, dst="index", phase="request",
                       label=f"earlier, at startup: {self.profile.name} resolved its store through NANDA",
                       detail={"product_server": self.store_source})
            record_lane(self.store_endpoint, "seller", self.store_name)
            try:
                reply, advice = await Advisor(self.profile, self.session, self.use_claude).answer(text)
            except Exception as e:
                reply = f"Something went wrong reading the store's catalog: {e}"
        finally:
            current_trace.reset(token)
            current_sink.reset(sink_token)
            if sink is not None:
                await sink.close()
        print(f"[{self.profile.key}] {context.context_id[:8]} -> {reply.splitlines()[0] if reply else ''}")
        message = new_text_message(reply, context_id=context.context_id, task_id=context.task_id)
        message.metadata.update({
            "trace": {"agent": "Claude" if self.use_claude else "rule-based", "product_server": self.store_source,
                      "mcp_calls": trace},
            "advice": advice,
        })
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Advice requests finish immediately and can't be cancelled")


def build_app(profile: Profile, base_url: str, endpoint: str, transport: str, source: str,
              name: str = "Product server") -> Starlette:
    card = build_agent_card(profile, base_url)
    executor = SpecialistExecutor(profile, base_url)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with AsyncExitStack() as stack:
            session = await stack.enter_async_context(mcp_session(endpoint, transport))
            executor.session = TracingSession(session, src=executor.lane, dst=base_of(endpoint), dst_name=name)
            executor.store_source, executor.store_endpoint, executor.store_name = source, endpoint, name
            mode = "Claude" if executor.use_claude else "rule-based"
            print(f"{profile.name}: store via {source}")
            print(f"A2A agent card: {base_url}/.well-known/agent-card.json ({mode} advice)")
            yield

    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True)
    return Starlette(routes=routes, lifespan=lifespan)


def find_store() -> tuple[str, str, str, str]:
    """(endpoint, transport, how it was found, store name), through NANDA when a locator or search is set."""
    locator, query = os.environ.get("NANDA_LOCATOR"), os.environ.get("NANDA_SEARCH")
    if not (locator or query):
        url = os.environ.get("PRODUCT_SERVER_URL", DEFAULT_SERVER_URL)
        return url, "sse", f"direct URL {url}", "Product server"
    server = asyncio.run(discover(locator, query))
    name = (server.card.get("seller") or {}).get("name") or server.name
    return server.endpoint, server.transport, f"NANDA Index: {server.describe()}", name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one specialist advisor agent as an A2A server")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=os.environ.get("SPECIALIST_PROFILE"),
                        required=not os.environ.get("SPECIALIST_PROFILE"))
    parser.add_argument("--port", type=int, help="Port (default: the profile's own, or SPECIALIST_PORT)")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    port = args.port or int(os.environ.get("SPECIALIST_PORT", profile.port))
    base_url = os.environ.get("SPECIALIST_PUBLIC_URL", f"http://localhost:{port}").rstrip("/")
    try:
        endpoint, transport, source, name = find_store()
    except DiscoveryError as e:
        raise SystemExit(f"Could not find the store through the NANDA Index: {e}") from None
    uvicorn.run(build_app(profile, base_url, endpoint, transport, source, name), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
