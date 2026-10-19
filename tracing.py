"""Per-request call traces that A2A agents return in their reply metadata ("trace").

Each entry is either a call record (one request + one response: {"kind", "src", "dst", "tool",
"arguments", "ok", "ms", "summary", "result_preview"}) or a one-way event with a "phase".
src/dst are lane names: "agent", "negotiator", "index", or a server's base URL (see base_of);
"lane" entries say which role a base URL plays.
flow_ui.py draws them as arrows.

Every entry has an "id". When a caller sends A2A metadata {"trace_sink": "http://localhost:.../"},
entries are also POSTed there as they happen (in order), so a viewer can draw them live and skip
them when the full trace arrives with the reply.
"""
import asyncio
import json
import time
import uuid
from contextvars import ContextVar
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession

TRACE_PREVIEW_CHARS = 600
SINK_HOSTS = {"localhost", "127.0.0.1", "::1"}

current_trace: ContextVar[list | None] = ContextVar("current_trace", default=None)


class TraceSink:
    """Posts trace entries, one at a time and in order, to a live viewer. Failures are ignored."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task = asyncio.create_task(self.run())

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=5) as client:
            while (entry := await self.queue.get()) is not None:
                try:
                    await client.post(self.url, json=[entry])
                except httpx.HTTPError:
                    pass

    def put(self, entry: dict) -> None:
        self.queue.put_nowait(entry)

    async def close(self) -> None:
        """Deliver what's queued (so live entries arrive before the reply), then stop."""
        self.queue.put_nowait(None)
        try:
            await asyncio.wait_for(self.task, 10)
        except asyncio.TimeoutError:
            self.task.cancel()


def open_sink(url: Any) -> Optional[TraceSink]:
    """A sink for a caller-supplied URL. Only local http(s) URLs: agents never post traces elsewhere."""
    if not isinstance(url, str):
        return None
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or parts.hostname not in SINK_HOSTS:
        return None
    return TraceSink(url)


current_sink: ContextVar[TraceSink | None] = ContextVar("current_sink", default=None)


def base_of(url: str) -> str:
    """Lane key for a server: scheme://host:port/ (its card and endpoints share it)."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/"


def record_lane(url: str, role: str, name: str) -> None:
    """Tell the viewer which server lives at a base URL ("seller" or "negotiator")."""
    record(kind="lane", key=base_of(url), role=role, name=name)


def record(**entry: Any) -> None:
    trace = current_trace.get()
    if trace is not None:
        entry.setdefault("id", uuid.uuid4().hex[:12])
        trace.append(entry)
        sink = current_sink.get()
        if sink is not None:
            sink.put(entry)


def summarize_result(text: str) -> str:
    """One-line description of an MCP tool result (the catalog tools return JSON objects)."""
    try:
        data = json.loads(text)
    except ValueError:
        return text.strip().splitlines()[0][:120] if text.strip() else ""
    if not isinstance(data, dict):
        return str(data)[:120]
    if "negotiation_id" in data:
        quote = data.get("quote") or {}
        reply = data.get("seller_reply") or {}
        head = f"{reply['action']} \u00b7 " if reply.get("action") else ""
        left = data.get("offers_left")
        said = f": \u201c{reply['message'][:80]}\u201d" if reply.get("message") else ""
        return (f"{head}${quote.get('unit_price')} ({data.get('status')}, "
                f"{left} offer{'' if left == 1 else 's'} left){said}") if not data.get("error") else f"error={data['error']}"
    parts = []
    for key, value in data.items():
        if isinstance(value, list):
            ids = [v["id"] for v in value if isinstance(v, dict) and v.get("id")]
            parts.append(f"{len(value)} {key}" + (f" ({', '.join(ids[:4])}{', …' if len(ids) > 4 else ''})" if ids else ""))
        elif isinstance(value, dict):
            parts.append(f"{len(value)} {key}")
        elif key in ("id", "name", "error", "stock", "available", "unit_price", "price", "total"):
            parts.append(f"{key}={value}")
    return "; ".join(parts[:5]) or ", ".join(list(data)[:6])


class TracingSession:
    """Wraps an MCP session and records each call into the current request's trace."""

    def __init__(self, session: ClientSession, *, src: str = "agent", dst: str = "", dst_name: str = "") -> None:
        self._session = session
        self.src, self.dst, self.dst_name = src, dst, dst_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    async def _traced(self, label: str, arguments: dict, call):
        started = time.perf_counter()
        ok, preview, summary = True, "", ""
        try:
            result = await call
            if hasattr(result, "content"):
                ok = not getattr(result, "isError", False)
                preview = "".join(getattr(c, "text", "") for c in result.content)
                summary = summarize_result(preview)
                ok = ok and not preview.lstrip().startswith('{"error"')
            elif hasattr(result, "tools"):
                preview = ", ".join(t.name for t in result.tools)
                summary = f"{len(result.tools)} tools"
            return result
        except Exception as e:
            ok, preview, summary = False, str(e), str(e)[:120]
            raise
        finally:
            record(kind="mcp", src=self.src, dst=self.dst, dst_name=self.dst_name, tool=label,
                   arguments=arguments, ok=ok, ms=round((time.perf_counter() - started) * 1000, 1),
                   summary=summary, result_preview=preview[:TRACE_PREVIEW_CHARS])

    async def call_tool(self, name: str, arguments: dict | None = None, *args, **kwargs):
        return await self._traced(name, arguments or {}, self._session.call_tool(name, arguments, *args, **kwargs))

    async def list_tools(self, *args, **kwargs):
        return await self._traced("tools/list", {}, self._session.list_tools(*args, **kwargs))


def traced_http_hooks(src: str) -> dict:
    """httpx event hooks that record NANDA Index lookups and card fetches made by an agent."""

    async def on_response(response) -> None:
        await response.aread()
        request = response.request
        started = request.extensions.get("trace_started", time.perf_counter())
        path = request.url.path
        try:
            data = response.json()
        except ValueError:
            data = None
        if "/api/v1/" in path:
            kind, dst = "nanda", "index"
            if isinstance(data, dict) and "candidates" in data:
                summary = f"{len(data['candidates'])} candidate(s): " + ", ".join(
                    f"{c.get('displayName')} {c.get('score', 0):.2f}" for c in data["candidates"])
            elif isinstance(data, dict) and "match" in data:
                summary = f"{data.get('match')} match \u2192 {(data.get('entry') or {}).get('url', '?')}"
            else:
                summary = f"HTTP {response.status_code}"
        else:
            kind, dst = "card", base_of(str(request.url))
            summary = f"{(data or {}).get('name', 'card')}" if isinstance(data, dict) else f"HTTP {response.status_code}"
        query = dict(request.url.params)
        record(kind=kind, src=src, dst=dst, tool=f"GET {path}", arguments=query, ok=response.status_code < 400,
               ms=round((time.perf_counter() - started) * 1000, 1), summary=summary,
               result_preview=json.dumps(data)[:TRACE_PREVIEW_CHARS] if data is not None else "")

    async def on_request(request) -> None:
        request.extensions["trace_started"] = time.perf_counter()

    return {"request": [on_request], "response": [on_response]}
