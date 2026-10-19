"""The shopping agent as an A2A server, so other agents can delegate shopping tasks to it.

    Other agent --A2A message/send--> this server --MCP tools--> product server

It serves an A2A agent card at /.well-known/agent-card.json and answers JSON-RPC on /
(A2A 1.0, plus 0.3 for older clients). Each A2A conversation (contextId) gets its own
shopping agent, so every calling agent has its own cart, budget and preferences.

Run the product server first, then:
    python a2a_server.py                       # finds the product server at PRODUCT_SERVER_URL
    NANDA_INDEX_URL=http://localhost:3001 NANDA_LOCATOR="urn:air:..." python a2a_server.py

Set ANTHROPIC_API_KEY to answer with Claude; otherwise the rule-based agent answers.
Register it in the NANDA Index with: python nanda_register.py --a2a --card-url <base>/.well-known/agent-card.json

Deployed (see render.yaml), the public URL comes from RENDER_EXTERNAL_URL or RAILWAY_PUBLIC_DOMAIN;
set A2A_PUBLIC_URL anywhere else, or the card will advertise localhost.
"""
import asyncio
import os
from collections import OrderedDict
from contextlib import AsyncExitStack, asynccontextmanager, suppress

import anyio
import httpx
import uvicorn
from a2a.helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentProvider, AgentSkill
from a2a.utils.errors import UnsupportedOperationError
from mcp.shared.exceptions import McpError
from starlette.applications import Starlette

from a2a_link import request_metadata
from marketplace import MarketplaceShoppingAgent, SellerDirectory
from nanda_discovery import DiscoveryError, discover, mcp_session
from shopping_agent import DEFAULT_SERVER_URL, ShoppingAgent
from tracing import TracingSession, base_of, current_sink, current_trace, open_sink, record, record_lane

# Platforms (Render, Railway, Heroku) pass the port to bind in PORT.
PORT = int(os.environ.get("A2A_PORT") or os.environ.get("PORT") or 8083)
# On a public deployment the card has to advertise the public URL: a caller that reads
# "localhost" from the card has nowhere to send its messages.
PUBLIC_BASE_URL = (
    os.environ.get("A2A_PUBLIC_URL")
    or os.environ.get("RENDER_EXTERNAL_URL")  # set automatically by Render
    or (f"https://{os.environ['RAILWAY_PUBLIC_DOMAIN']}" if os.environ.get("RAILWAY_PUBLIC_DOMAIN") else "")
).rstrip("/")
BASE_URL = PUBLIC_BASE_URL or f"http://localhost:{PORT}"
# Deployed, the platform's proxy reaches the app from another host, so bind every interface.
HOST = os.environ.get("A2A_HOST") or ("0.0.0.0" if PUBLIC_BASE_URL else "127.0.0.1")
MAX_CONVERSATIONS = 100
RECONNECT_DELAY_SECONDS = 5
# A dead connection to the store raises one of these, and the store may simply be asleep,
# so the connection is replaced rather than the error reported as a shopping failure.
CONNECTION_ERRORS = (McpError, httpx.HTTPError, anyio.BrokenResourceError, anyio.ClosedResourceError,
                     anyio.EndOfStream)

def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Personal Shopping Agent",
        description=(
            "Shops a product catalog on your behalf: finds, compares and recommends products "
            "within a budget, keeps a cart per conversation and places simulated orders."
        ),
        version="1.0.0",
        provider=AgentProvider(organization="Project NANDA demo", url=BASE_URL),
        supported_interfaces=[
            AgentInterface(url=f"{BASE_URL}/", protocol_binding="JSONRPC", protocol_version="1.0"),
            AgentInterface(url=f"{BASE_URL}/", protocol_binding="JSONRPC", protocol_version="0.3"),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="shopping.find_products",
                name="Find and recommend products",
                description="Search the catalog by category, price, rating and keywords; compare products; "
                            "recommend in-stock items within a budget.",
                tags=["shopping", "product-search", "recommendations", "comparison"],
                examples=["recommend headphones under 150", "search laptop under 400", "compare P001 P002"],
            ),
            AgentSkill(
                id="shopping.manage_cart",
                name="Build a cart and check out",
                description="Add or remove items with stock checks, show the cart total and place a simulated "
                            "order. The cart belongs to the A2A conversation (contextId).",
                tags=["shopping", "cart", "checkout"],
                examples=["add P801", "cart", "checkout"],
            ),
            AgentSkill(
                id="shopping.best_price",
                name="Best price from several sellers",
                description="Gets the best deal for a request from several sellers: a negotiator agent picks a "
                            "product at each seller, negotiates every price, and recommends the best value for "
                            "specs and price, with a comparison table. Quote only; needs Claude. Send message "
                            "metadata {\"quote_only\": true} to lock the cart.",
                tags=["shopping", "price-comparison", "negotiation", "quotes"],
                examples=["get me the best laptop under 900 at the best negotiated price"],
            ),
        ],
    )


class ConversationPool:
    """One shopping agent per A2A contextId, all sharing a single MCP session (plus the seller directory)."""

    def __init__(self, endpoint: str, transport: str, source: str, name: str) -> None:
        self.product_endpoint, self.transport = endpoint, transport
        self.product_source, self.product_name = source, name
        self.session: TracingSession | None = None
        self.directory: SellerDirectory | None = None
        self.use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.agents: OrderedDict[str, tuple[ShoppingAgent, asyncio.Lock]] = OrderedDict()
        self.create_lock = asyncio.Lock()
        self.broken = asyncio.Event()

    def attach(self, session: TracingSession) -> None:
        self.session = session
        self.broken.clear()

    def detach(self) -> None:
        """Drop a connection that is gone, and the conversations that were holding it.

        Their agents keep a reference to the dead session, and a cart the store can no longer
        price is worth less than an agent that answers again.
        """
        self.session = None
        self.agents.clear()
        self.broken.clear()

    def mark_broken(self) -> None:
        """Ask the connection holder for a new connection; a request found this one dead."""
        self.broken.set()

    async def get(self, context_id: str, session: TracingSession) -> tuple[ShoppingAgent, asyncio.Lock]:
        async with self.create_lock:
            if context_id in self.agents:
                self.agents.move_to_end(context_id)
                return self.agents[context_id]
            if self.use_claude:
                agent = MarketplaceShoppingAgent(session, self.directory, BASE_URL)
            else:
                agent = ShoppingAgent(session)
            await agent.start()
            self.agents[context_id] = (agent, asyncio.Lock())
            while len(self.agents) > MAX_CONVERSATIONS:
                self.agents.popitem(last=False)
            return self.agents[context_id]


async def keep_session(pool: ConversationPool) -> None:
    """Hold the connection to the product server for as long as the app runs.

    One task owns the connection for its whole life because an MCP session has to be closed by
    the task that opened it, so a request handler can neither open nor replace one. Connecting
    here rather than at startup also means a store that is asleep, as any free hosting tier is
    when idle, delays the first answer instead of stopping this agent from starting at all.
    """
    while True:
        try:
            async with mcp_session(pool.product_endpoint, pool.transport) as session:
                pool.attach(TracingSession(session, dst=base_of(pool.product_endpoint),
                                           dst_name=pool.product_name))
                print(f"Product server: {pool.product_source}")
                await pool.broken.wait()
                print("[a2a] reconnecting to the product server")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[a2a] product server unreachable ({type(e).__name__}: {e}); "
                  f"retrying in {RECONNECT_DELAY_SECONDS}s")
        pool.detach()
        await asyncio.sleep(RECONNECT_DELAY_SECONDS)


def is_connection_error(error: BaseException) -> bool:
    """Whether this failure is the connection to the store rather than the shopping itself.

    The MCP client runs in a task group, so a broken stream usually arrives wrapped in an
    exception group rather than on its own.
    """
    if isinstance(error, CONNECTION_ERRORS):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(is_connection_error(inner) for inner in error.exceptions)
    return False


async def cancel(task: asyncio.Task) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


class ShoppingAgentExecutor(AgentExecutor):
    def __init__(self, pool: ConversationPool) -> None:
        self.pool = pool

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = context.get_user_input()
        metadata = request_metadata(context)
        quote_only = bool(metadata.get("quote_only"))
        print(f"[a2a] {context.context_id[:8]} <- {text!r}{' (quote only)' if quote_only else ''}")
        trace: list[dict] = []
        token = current_trace.set(trace)
        sink = open_sink(metadata.get("trace_sink"))
        sink_token = current_sink.set(sink)
        negotiations: list = []
        try:
            if self.pool.product_source.startswith("NANDA"):
                record(kind="startup", src="agent", dst="index", phase="request",
                       label="earlier, at startup: shopping agent resolved its home store through NANDA",
                       detail={"product_server": self.pool.product_source})
            record_lane(self.pool.product_endpoint, "seller", self.pool.product_name)
            session = self.pool.session
            if session is None:
                reply = (f"I can't reach the store right now ({self.pool.product_source}). It may still be "
                         "starting up, so please send your request again in a few seconds.")
            else:
                try:
                    # Inside the guard: opening a conversation lists the store's tools, which is
                    # itself a call that a dropped connection fails.
                    agent, lock = await self.pool.get(context.context_id, session)
                    async with lock:
                        if quote_only and not isinstance(agent, MarketplaceShoppingAgent):
                            reply = ("Price negotiation needs Claude: start a2a_server.py with ANTHROPIC_API_KEY set. "
                                     "Nothing was negotiated.")
                        else:
                            if isinstance(agent, MarketplaceShoppingAgent):
                                agent.quote_only = quote_only
                            reply = await agent.handle(text) or "Send me a shopping request, e.g. 'recommend headphones under 150'."
                            negotiations = list(getattr(agent, "negotiations", []))
                except Exception as e:
                    if is_connection_error(e):
                        self.pool.mark_broken()
                        reply = ("The store's connection dropped. I'm opening a new one, so please send your "
                                 "request again in a few seconds.")
                    else:
                        reply = f"Something went wrong talking to the store: {e}"
        finally:
            current_trace.reset(token)
            current_sink.reset(sink_token)
            if sink is not None:
                await sink.close()
        print(f"[a2a] {context.context_id[:8]} -> {reply.splitlines()[0] if reply else ''}")
        message = new_text_message(reply, context_id=context.context_id, task_id=context.task_id)
        message.metadata.update({
            "trace": {
                "agent": "Claude" if self.pool.use_claude else "rule-based",
                "product_server": self.pool.product_source,
                "mcp_calls": trace,
            },
            "negotiations": negotiations,
        })
        await event_queue.enqueue_event(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Shopping requests finish immediately and can't be cancelled")


def build_app(endpoint: str, transport: str, source: str, name: str = "Product server") -> Starlette:
    pool = ConversationPool(endpoint, transport, source, name)
    card = build_agent_card()
    handler = DefaultRequestHandler(
        agent_executor=ShoppingAgentExecutor(pool), task_store=InMemoryTaskStore(), agent_card=card,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with AsyncExitStack() as stack:
            pool.directory = SellerDirectory()
            stack.push_async_callback(pool.directory.close)
            keeper = asyncio.create_task(keep_session(pool))
            stack.push_async_callback(cancel, keeper)
            mode = "Claude" if pool.use_claude else "rule-based"
            print(f"A2A agent card: {BASE_URL}/.well-known/agent-card.json ({mode} agent)")
            yield

    routes = create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/", enable_v0_3_compat=True)
    return Starlette(routes=routes, lifespan=lifespan)


def main() -> None:
    locator, query = os.environ.get("NANDA_LOCATOR"), os.environ.get("NANDA_SEARCH")
    try:
        if locator or query:
            server = asyncio.run(discover(locator, query))
            endpoint, transport = server.endpoint, server.transport
            name = (server.card.get("seller") or {}).get("name") or server.name
            source = f"NANDA Index: {server.describe()}"
        else:
            endpoint, transport, name = DEFAULT_SERVER_URL, "sse", "Product server"
            source = f"direct URL {endpoint}"
    except DiscoveryError as e:
        raise SystemExit(
            f"Could not find the product server through the NANDA Index: {e}\n"
            "Is the index running (curl http://localhost:3001/health) and the product server registered?"
        ) from None
    uvicorn.run(build_app(endpoint, transport, source, name), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
