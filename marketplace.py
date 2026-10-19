"""Multi-seller shopping for the A2A shopping agent: sellers and the negotiator are found through NANDA.

    shopping agent --NANDA agentic search--> seller MCP servers
                   --NANDA agentic search--> negotiator agent --A2A--> (shortlists, negotiates and picks over MCP)

Sellers have their own SKUs and catalogs, so the shopping agent doesn't pick the product: it
describes the request, and the negotiator picks a candidate at each seller and the best deal.
Claude decides which agent to delegate to; code checks that the agent was really discovered,
that the sellers came from the index, and that quote-only requests can't touch the cart.
"""
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from mcp import ClientSession

from a2a_link import A2ALink
from claude_agent import SYSTEM_PROMPT, ClaudeShoppingAgent
from nanda_discovery import (
    NANDA_INDEX_URL, DiscoveredAgent, DiscoveredServer, find_agent_candidates, find_server_candidates, mcp_session,
)
from negotiation_report import attach_report
from tracing import TracingSession, base_of, current_sink, current_trace, record, record_lane, traced_http_hooks

SELLER_TASK = "product seller catalog: search products, check stock, give price quotes and negotiate prices"
SELLER_CACHE_SECONDS = 120
NEGOTIATION_TIMEOUT = 600
MAX_SELLERS = 4  # the negotiator takes 2 to 4 sellers
CART_TOOLS = {"add_to_cart", "remove_from_cart", "checkout"}


def is_seller_card(card: dict) -> bool:
    tools = {t.get("name") for t in card.get("tools", []) if isinstance(t, dict)}
    return {"start_negotiation", "make_offer", "get_negotiation"} <= tools


@dataclass
class Seller:
    identifier: str
    name: str
    endpoint: str
    session: Any


class SellerDirectory:
    """Seller MCP servers found through the NANDA Index, each with one long-lived MCP session.

    Sessions are opened in their own tasks: an SSE client must be closed by the task that opened it.
    """

    def __init__(self, index_url: str = NANDA_INDEX_URL) -> None:
        self.index_url = index_url
        self.sellers: dict[str, Seller] = {}
        self.refreshed = 0.0
        self.lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.holders: list[asyncio.Task] = []

    def add(self, server: DiscoveredServer, session: ClientSession) -> Seller:
        seller = Seller(server.identifier, (server.card.get("seller") or {}).get("name") or server.name, server.endpoint,
                        TracingSession(session, dst=base_of(server.endpoint), dst_name=server.name))
        self.sellers[server.endpoint] = seller
        return seller

    async def connect(self, server: DiscoveredServer) -> ClientSession:
        ready: asyncio.Future = asyncio.get_running_loop().create_future()

        async def hold() -> None:
            try:
                async with mcp_session(server.endpoint, server.transport) as session:
                    ready.set_result(session)
                    await self.stop.wait()
            except Exception as e:
                if not ready.done():
                    ready.set_exception(e)

        self.holders.append(asyncio.create_task(hold()))
        return await asyncio.wait_for(ready, 20)

    async def discover(self) -> list[Seller]:
        async with self.lock:
            if time.time() - self.refreshed > SELLER_CACHE_SECONDS:
                async with httpx.AsyncClient(timeout=20, event_hooks=traced_http_hooks("agent")) as http:
                    found = await find_server_candidates(http, SELLER_TASK, self.index_url)
                for server, _score, _description in found:
                    if is_seller_card(server.card) and server.endpoint not in self.sellers:
                        try:
                            self.add(server, await self.connect(server))
                        except Exception as e:
                            print(f"[sellers] could not connect to {server.name} at {server.endpoint}: {e}")
                self.refreshed = time.time()
        for seller in self.sellers.values():
            record_lane(seller.endpoint, "seller", seller.name)
        return list(self.sellers.values())

    async def close(self) -> None:
        self.stop.set()
        await asyncio.gather(*self.holders, return_exceptions=True)


def tool_json(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"))


MARKET_TOOLS = [
    {
        "name": "find_sellers",
        "description": "Find the seller MCP servers registered in the NANDA Index that give quotes and negotiate.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "find_agents",
        "description": "Ask the NANDA Index for A2A agents that can do a job, described in plain words. "
                       "Returns each candidate's description and skills.",
        "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
    },
    {
        "name": "delegate_negotiation",
        "description": "Send a shopping request to an agent returned by find_agents. It picks a candidate "
                       "product at every seller found by find_sellers, negotiates each price, and returns a spec and "
                       "price comparison with the recommended winner. Quote only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "description": "The agent's identifier from find_agents"},
                "request": {"type": "string", "description": "What the shopper wants, in their words"},
                "category": {"type": "string", "description": "Catalog category, e.g. 'laptops' (see list_categories)"},
                "keywords": {"type": "string", "description": "Optional search keywords, e.g. 'gaming'"},
                "max_budget": {"type": "number", "description": "Shopper's maximum unit price in USD, if known"},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
                "notes": {"type": "string", "description": "Preferences the negotiator should weigh (no secrets)"},
            },
            "required": ["agent", "request"],
        },
    },
]

MARKET_RULES = """\
Several sellers, each with its own SKUs and catalog, sell overlapping products: the same model, the same
model in a different configuration, or models only one of them carries. A negotiator agent can pick the
best candidate at each seller, bargain, and choose the best deal.
When the shopper wants the best price or deal, a negotiated price, or "the best <product>" from several sellers:
1. Call find_sellers to see the sellers in the NANDA Index.
2. Call find_agents with a short description of the job you need done, pick the agent whose
   description and skills fit, and call delegate_negotiation with that agent: the shopper's request,
   the catalog category, optional keywords, the budget and any preferences. Don't pick the product yourself.
3. Report the negotiator's pick and its reason, and each seller's product and negotiated price, exactly as
   the negotiator reported them. Never invent prices or specs.
If the request is quote-only, never add anything to the cart or check out.

"""


class MarketplaceShoppingAgent(ClaudeShoppingAgent):
    system_prompt = MARKET_RULES + SYSTEM_PROMPT

    def __init__(self, session: ClientSession, directory: SellerDirectory, own_url: str, **kwargs) -> None:
        super().__init__(session, **kwargs)
        self.directory = directory
        self.own_base = base_of(own_url)
        self.quote_only = False
        self.found_agents: dict[str, DiscoveredAgent] = {}
        self.seller_ids: list[str] = []  # NANDA identifiers of the sellers find_sellers returned
        self.negotiations: list[dict] = []

    async def start(self) -> None:
        await super().start()
        self.base_tools = self.claude_tools + MARKET_TOOLS

    async def handle(self, message: str) -> str:
        self.negotiations = []
        self.claude_tools = [t for t in self.base_tools if not (self.quote_only and t["name"] in CART_TOOLS)]
        if self.quote_only and message.strip().lower().split(" ")[0] in ("add", "remove", "checkout", "buy"):
            return "This is a quote-only request, so I won't change the cart or check out."
        reply = await super().handle(message)
        return attach_report(reply, self.negotiations) if reply else reply

    def describe_state(self) -> str:
        mode = "Request mode: QUOTE ONLY (cart tools are disabled)\n" if self.quote_only else ""
        return mode + super().describe_state()

    async def run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        handler = {"find_sellers": self.find_sellers, "find_agents": self.find_agents,
                   "delegate_negotiation": self.delegate_negotiation}.get(name)
        if handler is None:
            if self.quote_only and name in CART_TOOLS:
                return "Not allowed: this is a quote-only request.", True
            return await super().run_tool(name, args)
        try:
            return tool_json(await handler(**args)), False
        except Exception as e:
            return f"{name} failed: {e}", True

    async def find_sellers(self) -> dict:
        sellers = await self.directory.discover()
        self.seller_ids = [s.identifier for s in sellers][:MAX_SELLERS]
        return {"sellers": [{"name": s.name, "identifier": s.identifier} for s in sellers]}

    async def find_agents(self, task: str) -> dict:
        async with httpx.AsyncClient(timeout=20, event_hooks=traced_http_hooks("agent")) as http:
            candidates = await find_agent_candidates(http, task, self.directory.index_url)
        listed = []
        for c in candidates:
            if base_of(c.agent.endpoint) == self.own_base:
                continue  # that's this shopping agent
            self.found_agents[c.agent.identifier] = c.agent
            listed.append({
                "identifier": c.agent.identifier, "name": c.agent.name, "index_score": round(c.score, 2),
                "description": c.description,
                "skills": [{"name": s.get("name"), "description": s.get("description")}
                           for s in c.agent.card.get("skills", [])],
            })
        return {"task": task, "candidates": listed}

    async def delegate_negotiation(self, agent: str, request: str, category: Optional[str] = None,
                                   keywords: str = "", max_budget: Optional[float] = None, quantity: int = 1,
                                   notes: str = "") -> dict:
        target = self.found_agents.get(agent)
        if target is None:
            return {"error": "Unknown agent: use an identifier returned by find_agents"}
        if len(self.seller_ids) < 2:
            return {"error": f"Call find_sellers first; a best deal needs two sellers (found {len(self.seller_ids)})"}
        sellers = self.seller_ids
        task = {"request": str(request)[:300], "category": category or None, "keywords": str(keywords or "")[:80],
                "quantity": int(quantity or 1), "max_budget": max_budget, "sellers": sellers,
                "notes": str(notes or "")[:400], "quote_only": True}
        record_lane(target.endpoint, "negotiator", target.name)
        record(kind="a2a", src="agent", dst="negotiator", phase="request",
               label=f"A2A SendMessage: best deal for \u201c{task['request'][:50]}\u201d from {len(sellers)} sellers",
               detail=task)
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=NEGOTIATION_TIMEOUT) as http:
            sink = current_sink.get()
            link = A2ALink(target, http, me="shopping-agent", them="negotiator", quiet=True,
                           metadata={"trace_sink": sink.url} if sink else None)
            await link.connect()
            try:
                reply, metadata = await link.send(json.dumps(task))
            finally:
                await link.close()
        trace = current_trace.get()
        if trace is not None:
            trace.extend(metadata.get("trace") or [])
        result = metadata.get("negotiation")
        record(kind="a2a", src="negotiator", dst="agent", phase="response",
               label=f"reply: {reply.splitlines()[0] if reply else '(no text)'}",
               detail={"reply": reply, "negotiation": result}, ms=round((time.perf_counter() - started) * 1000, 1))
        if result:
            self.negotiations.append(result)
        return {"negotiator_reply": reply, "result": result}
