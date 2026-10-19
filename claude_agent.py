"""Claude-powered shopping agent that uses the product catalog MCP server's tools.

Claude decides which catalog tools to call; the agent runs them over MCP and keeps
the cart, budget and preferences locally (same state as the rule-based ShoppingAgent).

Run the server first (python product_server.py), then:
    export ANTHROPIC_API_KEY=sk-ant-...
    python claude_agent.py                                  # interactive chat
    python claude_agent.py "wireless headphones under 300"  # one-shot commands

To find the server through the NANDA Index instead of a fixed URL:
    python claude_agent.py --nanda-search "product catalog"
    python claude_agent.py --nanda-locator "urn:air:host39.org:personal:you@example.com"
"""
import argparse
import asyncio
import json
import os
import re
from typing import Any, Optional

import anthropic
import httpx
from mcp import ClientSession

from nanda_discovery import DiscoveryError, choose_server, mcp_session
from shopping_agent import DEFAULT_SERVER_URL, ShoppingAgent

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_TOOL_ROUNDS = 10
MAX_TURNS_KEPT = 20
SELLER_ONLY_TOOLS = {"start_negotiation", "make_offer", "get_negotiation"}

# Cart buttons in the web UI send these exact commands; handling them without Claude
# keeps the buttons instant and free.
DIRECT_COMMAND_RE = re.compile(r"^(?:(?:add|remove)\s+P\d{3,5}(?:\s+x\d+)?|cart|checkout)$", re.IGNORECASE)

LOCAL_TOOLS = [
    {
        "name": "add_to_cart",
        "description": "Add a product to the shopper's cart after checking stock. Use the product ID (e.g. P003).",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1, "default": 1},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "remove_from_cart",
        "description": "Remove a product from the cart. Omit quantity to remove it entirely.",
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1},
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "view_cart",
        "description": "Show the items in the shopper's cart and the total.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_budget",
        "description": "Remember the shopper's maximum budget per item, in USD.",
        "input_schema": {
            "type": "object",
            "properties": {"amount": {"type": "number", "exclusiveMinimum": 0}},
            "required": ["amount"],
        },
    },
    {
        "name": "set_preferences",
        "description": "Remember things the shopper cares about (e.g. wireless, travel, waterproof).",
        "input_schema": {
            "type": "object",
            "properties": {"preferences": {"type": "array", "items": {"type": "string"}}},
            "required": ["preferences"],
        },
    },
    {
        "name": "checkout",
        "description": "Place the (simulated) order for everything in the cart. Only call this when the shopper explicitly asks to check out.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = """\
You are a friendly personal shopping assistant for an online store.
Use the catalog tools to look up real products; never invent products, prices or stock.
Always mention product IDs (like P003) and prices when you list products.
Respect the shopper's budget and preferences when searching or recommending.
Recommend only the kind of product asked for ("computer" means the laptops category; a tablet is
not a laptop). If nothing of that kind fits the budget, say so plainly and name the cheapest one,
without listing product IDs from other categories.
If a message asks for several products, handle each one separately.
Use the cart tools when the shopper wants to add, remove or buy items.
Reply in short plain text (no Markdown tables or headings); simple "-" lists are fine.

Shopper state when their latest message was sent (tool results after that are newer):
{state}"""


class ClaudeShoppingAgent(ShoppingAgent):
    system_prompt = SYSTEM_PROMPT

    def __init__(self, session: ClientSession, *, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        super().__init__(session)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.history: list[dict] = []
        self.turn_starts: list[int] = []
        self.claude_tools: list[dict] = []
        self.mcp_tool_names: set[str] = set()

    async def start(self) -> None:
        await super().start()
        # Seller-side negotiation tools are for a buyer's negotiator agent, not for the shopper's assistant.
        tools = [t for t in self.tools if t.name not in SELLER_ONLY_TOOLS]
        self.mcp_tool_names = {t.name for t in tools}
        self.claude_tools = [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
            for t in tools
        ] + LOCAL_TOOLS

    async def handle(self, message: str) -> str:
        text = message.strip()
        if not text:
            return ""
        if DIRECT_COMMAND_RE.match(text):
            return await super().handle(text)

        self.trim_history()
        start = len(self.history)
        self.turn_starts.append(start)
        self.history.append({"role": "user", "content": text})
        try:
            return await self.run_claude()
        except anthropic.APIError as e:
            # Roll back the unfinished turn so the history stays a valid conversation.
            del self.history[start:]
            self.turn_starts.pop()
            return f"Claude couldn't answer: {getattr(e, 'message', e)}"

    async def run_claude(self) -> str:
        # Snapshot once per turn: refreshing it between tool rounds made Claude read
        # its own cart changes as pre-existing items.
        system = self.system_prompt.format(state=self.describe_state())
        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system,
                tools=self.claude_tools,
                messages=self.history,
            )
            self.history.append({
                "role": "assistant",
                "content": [b.model_dump(mode="json", exclude_none=True) for b in response.content],
            })
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                return "".join(b.text for b in response.content if b.type == "text").strip()

            results = []
            for block in tool_uses:
                output, is_error = await self.run_tool(block.name, block.input)
                print(f"[tool] {block.name} {json.dumps(block.input)} -> {'error' if is_error else 'ok'}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error})
            self.history.append({"role": "user", "content": results})
        return "I had to stop after too many tool calls. Could you narrow down the request?"

    async def run_tool(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        try:
            if name in self.mcp_tool_names:
                result = await self.session.call_tool(name, args)
                return "".join(getattr(c, "text", "") for c in result.content), bool(result.isError)
            return await self.run_local_tool(name, args), False
        except Exception as e:
            return f"{name} failed: {e}", True

    async def run_local_tool(self, name: str, args: dict[str, Any]) -> str:
        pid = str(args.get("product_id", "")).upper()
        quantity = args.get("quantity")
        # The cart helpers parse quantities from command text ("add P003 x2").
        qty_text = f" x{int(quantity)}" if quantity else ""
        if name == "add_to_cart":
            return await self.add_to_cart([pid], f"add {pid}{qty_text}".lower())
        if name == "remove_from_cart":
            return self.remove_from_cart([pid], f"remove {pid}{qty_text}".lower())
        if name == "view_cart":
            return self.format_cart()
        if name == "set_budget":
            self.budget = float(args["amount"])
            return f"Budget set to ${self.budget:.2f} per item."
        if name == "set_preferences":
            for pref in args.get("preferences", []):
                pref = str(pref).strip().lower()
                if pref and pref not in self.preferences:
                    self.preferences.append(pref)
            return "Preferences: " + (", ".join(self.preferences) or "none")
        if name == "checkout":
            return await self.checkout()
        raise ValueError(f"Unknown tool '{name}'")

    def describe_state(self) -> str:
        budget = f"${self.budget:.2f} per item" if self.budget is not None else "not set"
        prefs = ", ".join(self.preferences) or "none"
        return f"Budget: {budget}\nPreferences: {prefs}\n{self.format_cart()}"

    def reset_conversation(self) -> None:
        self.history.clear()
        self.turn_starts.clear()

    def trim_history(self) -> None:
        # Drop whole turns so the history never starts with a dangling tool_result.
        while len(self.turn_starts) >= MAX_TURNS_KEPT:
            cut = self.turn_starts[1]
            del self.history[:cut]
            self.turn_starts = [s - cut for s in self.turn_starts[1:]]


async def run(endpoint: str, transport: str, source: str, model: str, commands: list[str]) -> None:
    async with mcp_session(endpoint, transport) as session:
        agent = ClaudeShoppingAgent(session, model=model)
        await agent.start()

        if commands:
            print(f"[server] {source}")
            for command in commands:
                print(f"you> {command}")
                print(f"agent> {await agent.handle(command)}\n")
            return

        print(f"Connected using {model} to {source}")
        print(f"Tools: {', '.join(agent.tool_names)}")
        print("Ask for anything, e.g. 'wireless headphones under 300 for travel'. Type 'quit' to exit.\n")
        loop = asyncio.get_running_loop()
        while True:
            try:
                message = await loop.run_in_executor(None, input, "you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if message.strip().lower() in {"quit", "exit", "bye"}:
                break
            try:
                reply = await agent.handle(message)
            except Exception as e:
                reply = f"Something went wrong talking to the store: {e}"
            if reply:
                print(f"agent> {reply}\n")
        print("Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude-powered shopping agent (MCP client)")
    parser.add_argument("commands", nargs="*", help="Run these messages and exit instead of starting a chat")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help="Product server SSE URL (when not using NANDA)")
    parser.add_argument("--nanda-locator", default=os.environ.get("NANDA_LOCATOR"),
                        help="Find the server in the NANDA Index by exact urn:air: identifier")
    parser.add_argument("--nanda-search", default=os.environ.get("NANDA_SEARCH"),
                        help="Find the server in the NANDA Index by keywords, e.g. 'product catalog'")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model ID")
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("set ANTHROPIC_API_KEY first, e.g. export ANTHROPIC_API_KEY=sk-ant-...")

    try:
        endpoint, transport, source = asyncio.run(choose_server(args.url, args.nanda_locator, args.nanda_search))
    except DiscoveryError as e:
        raise SystemExit(f"NANDA discovery failed: {e}") from None
    try:
        asyncio.run(run(endpoint, transport, source, args.model, args.commands))
    except BaseExceptionGroup as group:
        # The MCP clients wrap connection failures in an exception group.
        if group.subgroup(httpx.ConnectError) is None:
            raise
        hint = "" if args.nanda_locator or args.nanda_search else "\nStart it first in another terminal: python product_server.py"
        raise SystemExit(f"Could not connect to the MCP server at {endpoint}.{hint}") from None


if __name__ == "__main__":
    main()
