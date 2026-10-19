"""Web UI for the personal shopping agent.

Run the product server first (python product_server.py), then:
    python agent_ui.py
and open http://localhost:8082 in your browser.

If ANTHROPIC_API_KEY is set, chat messages are answered by Claude (claude_agent.py);
otherwise the rule-based agent (shopping_agent.py) is used.

Set NANDA_LOCATOR (exact urn:air: identifier) or NANDA_SEARCH (keywords) to find the
product server through the NANDA Index instead of PRODUCT_SERVER_URL.
"""
import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route
import uvicorn

from claude_agent import ClaudeShoppingAgent
from nanda_discovery import choose_server, mcp_session
from shopping_agent import DEFAULT_SERVER_URL, ShoppingAgent

PAGE_FILE = Path(__file__).parent / "ui.html"
MAX_HISTORY = 500
server_endpoint = DEFAULT_SERVER_URL


@asynccontextmanager
async def lifespan(app: Starlette):
    global server_endpoint
    # One MCP session is kept open for the life of the web app, so the agent's
    # cart, budget and preferences persist across browser requests.
    async with AsyncExitStack() as stack:
        server_endpoint, transport, source = await choose_server(
            DEFAULT_SERVER_URL, os.environ.get("NANDA_LOCATOR"), os.environ.get("NANDA_SEARCH"),
        )
        session = await stack.enter_async_context(mcp_session(server_endpoint, transport))
        if os.environ.get("ANTHROPIC_API_KEY"):
            agent = ClaudeShoppingAgent(session)
            mode = f"Claude ({agent.model})"
        else:
            agent = ShoppingAgent(session)
            mode = "rule-based"
        await agent.start()
        app.state.agent = agent
        app.state.lock = asyncio.Lock()
        app.state.history = []
        print(f"Connected to {source} using the {mode} agent")
        yield


def snapshot(agent: ShoppingAgent) -> dict:
    return {
        "server": server_endpoint,
        "tools": agent.tool_names,
        "categories": agent.categories,
        "budget": agent.budget,
        "preferences": agent.preferences,
        "cart": [
            {"id": pid, **item, "subtotal": round(item["unit_price"] * item["quantity"], 2)}
            for pid, item in agent.cart.items()
        ],
        "total": agent.cart_total(),
    }


async def api_state(request: Request) -> JSONResponse:
    return JSONResponse(snapshot(request.app.state.agent))


def record(history: list[dict], role: str, text: str) -> None:
    history.append({"role": role, "text": text, "time": datetime.now().isoformat(timespec="seconds")})
    del history[:-MAX_HISTORY]


async def api_chat(request: Request) -> JSONResponse:
    body = await request.json()
    message = str(body.get("message", ""))
    # Cart buttons send commands through the agent too, but those aren't part of the conversation.
    keep = body.get("record", True)
    agent: ShoppingAgent = request.app.state.agent
    async with request.app.state.lock:
        try:
            reply = await agent.handle(message)
        except Exception as e:
            reply = f"Something went wrong talking to the store: {e}"
        if keep and message.strip():
            record(request.app.state.history, "you", message)
            if reply:
                record(request.app.state.history, "agent", reply)
    return JSONResponse({"reply": reply, "state": snapshot(agent)})


async def api_history(request: Request) -> JSONResponse:
    return JSONResponse({"messages": request.app.state.history})


async def api_history_clear(request: Request) -> JSONResponse:
    request.app.state.history.clear()
    agent = request.app.state.agent
    if isinstance(agent, ClaudeShoppingAgent):
        agent.reset_conversation()
    return JSONResponse({"messages": []})


async def api_tool(request: Request) -> JSONResponse:
    body = await request.json()
    name = body.get("name")
    arguments = body.get("arguments") or {}
    agent: ShoppingAgent = request.app.state.agent
    if name not in agent.tool_names:
        return JSONResponse({"error": f"Unknown tool '{name}'"}, status_code=400)
    try:
        result = await agent.call(name, **arguments)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return JSONResponse(result)


async def homepage(request: Request) -> HTMLResponse:
    # Read on every request so edits to ui.html show up on browser refresh.
    return HTMLResponse(PAGE_FILE.read_text(encoding="utf-8"))


app = Starlette(
    debug=True,
    routes=[
        Route("/", endpoint=homepage),
        Route("/api/state", endpoint=api_state),
        Route("/api/chat", endpoint=api_chat, methods=["POST"]),
        Route("/api/history", endpoint=api_history),
        Route("/api/history/clear", endpoint=api_history_clear, methods=["POST"]),
        Route("/api/tool", endpoint=api_tool, methods=["POST"]),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("UI_PORT", 8082)))
