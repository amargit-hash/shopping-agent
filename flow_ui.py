"""Web page that runs the gift planner and shows every hop of the NANDA + A2A + MCP flow live.

    Gift planner --(1) task / URN / keywords--> NANDA Index
                 --(2) GET card----> shopping agent's A2A agent card
                 --(3) A2A---------> shopping agent --MCP--> seller(s)
                                                    --A2A--> negotiator --MCP--> sellers  (negotiate mode)
                 --(1..3) per request--> laptop / gift / travel specialist --MCP--> store  (advice mode)

Every HTTP exchange the planner makes is captured with httpx event hooks and streamed to
the browser (Server-Sent Events). Calls made inside the shopping agent and the negotiator are
reported in each reply's metadata ("trace", see tracing.py) and drawn on their own lanes; seller
and negotiator lanes appear as the trace names them.

Start the index, the seller(s), a2a_server.py (negotiator_agent.py for negotiation, specialist_agent.py
for advice) first, then:
    python flow_ui.py        # http://localhost:8084
"""
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("NANDA_INDEX_URL", "http://localhost:3001")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from claude_agent import DEFAULT_MODEL  # noqa: E402
from a2a_link import A2ALink  # noqa: E402
from gift_planner_agent import (  # noqa: E402
    ADVICE_TASK, JOB_NAMES, NEGOTIATE_TASK, NEGOTIATION_TIMEOUT, PLANNER_TASK, QUOTE_ONLY, ShoppingAgentLink,
    advice_section, advice_task, claude_plan, fixed_plan, negotiate_plan, pick_capable, require_skill, split_requests,
)
from nanda_discovery import (  # noqa: E402
    A2A_CARD_TYPE, NANDA_INDEX_URL, DiscoveredAgent, DiscoveryError, agentic_search_entries, read_agent_card,
    read_candidates, resolve, resolve_agent, resolve_entry, search_entries,
)
from tracing import base_of  # noqa: E402

PORT = int(os.environ.get("FLOW_UI_PORT", 8084))
AGENT_URN = os.environ.get("AGENT_URN", "urn:air:host39.org:personal:shopping-agent@nanda.local")
PRODUCT_URN = os.environ.get("PRODUCT_URN", "urn:air:host39.org:personal:demo@nanda.local")
SELLER_B_URN = os.environ.get("SELLER_B_URN", "urn:air:host39.org:personal:seller-b@nanda.local")
NEGOTIATOR_URN = os.environ.get("NEGOTIATOR_URN", "urn:air:host39.org:personal:negotiator@nanda.local")
SPECIALIST_URNS = {
    "Laptop specialist (A2A, advice mode)": os.environ.get(
        "LAPTOP_URN", "urn:air:host39.org:personal:laptop-advisor@nanda.local"),
    "Gift specialist (A2A, advice mode)": os.environ.get("GIFT_URN", "urn:air:host39.org:personal:gift-ideas@nanda.local"),
    "Travel specialist (A2A, advice mode)": os.environ.get(
        "TRAVEL_URN", "urn:air:host39.org:personal:travel-gear@nanda.local"),
}
SHOWN_LANES = ("planner", "index", "agent")
RUNS: dict[str, "FlowRecorder"] = {}  # run id -> recorder, while the run lasts
PAGE_FILE = Path(__file__).parent / "flow_ui.html"
MEDIA_NAMES = {A2A_CARD_TYPE: "A2A agent card", "application/mcp-server-card+json": "MCP server card"}


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return raw.decode(errors="replace")[:2000]


def message_text(message: dict) -> str:
    return "\n".join(p.get("text", "") for p in (message or {}).get("parts", []) if p.get("text"))


def whole_numbers(value: Any) -> Any:
    """A2A metadata is a protobuf Struct, which turns every number into a float (1 -> 1.0)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: whole_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [whole_numbers(v) for v in value]
    return value


def first_line(text: str, limit: int = 90) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


def call_label(kind: str, entry: dict) -> str:
    tool, args = entry.get("tool", ""), entry.get("arguments") or {}
    if kind == "nanda":
        for key in ("q", "locator"):
            if args.get(key):
                return f"{tool}  {key}=\u201c{args[key]}\u201d"
        return tool
    if kind == "card":
        return f"{tool} (read card)"
    if tool == "make_offer":
        message = first_line(str(args.get("message", "")), 60)
        return f"MCP make_offer ${args.get('offer_price')}" + (f": \u201c{message}\u201d" if message else "")
    compact = json.dumps(args, separators=(",", ":"))
    return f"MCP {tool} {compact if compact != '{}' else ''}".strip()


class FlowRecorder:
    """Turns the planner's HTTP traffic into diagram events: one arrow per request and per response."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self.queue = queue
        self.seq = 0
        self.started = time.perf_counter()
        self.reported_startup: set[str] = set()  # agents whose startup lookup was drawn
        self.agent_lane: str | None = None  # advice mode: the lane of the agent being asked, for "agent" traces
        self.warned_no_trace = False
        self.lane_ids: dict[str, str] = {}  # trace lane key (base URL or role name) -> diagram lane
        self.lane_roles: dict[str, str] = {}
        self.extra_lanes = 0
        self.seen: set[str] = set()

    def emit(self, kind: str, src: str, dst: str, label: str, detail: Any = None, **extra: Any) -> None:
        self.seq += 1
        self.queue.put_nowait({
            "seq": self.seq, "kind": kind, "from": src, "to": dst, "label": label, "detail": detail,
            "t": round((time.perf_counter() - self.started) * 1000), **extra,
        })

    def step(self, title: str) -> None:
        self.emit("step", "planner", "planner", title)

    def classify(self, request: httpx.Request) -> tuple[str, str]:
        url = str(request.url)
        if url.startswith(NANDA_INDEX_URL):
            return "index", "nanda"
        lane = self.lane_ids.get(base_of(url), "agent")
        if request.method == "GET" and "/.well-known/" in url:
            return lane, "card"
        return lane, "a2a"

    async def on_request(self, request: httpx.Request) -> None:
        lane, kind = self.classify(request)
        request.extensions["flow_started"] = time.perf_counter()
        body = parse_json(request.content)
        query = {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}
        if kind == "nanda" and request.url.path.endswith("/resolve"):
            label = f"GET /api/v1/resolve  locator={query.get('locator', '')}"
        elif kind == "nanda" and request.url.path.endswith("/agentic-search"):
            label = f"GET /api/v1/agentic-search  q=\u201c{query.get('q', '')}\u201d"
        elif kind == "nanda" and request.url.path.endswith("/search"):
            label = f"GET /api/v1/search  q=\u201c{query.get('q', '')}\u201d"
        elif kind == "card":
            label = f"GET {request.url.host}:{request.url.port}{request.url.path}"
        else:
            method = (body or {}).get("method", "POST") if isinstance(body, dict) else "POST"
            text = message_text(((body or {}).get("params") or {}).get("message")) if isinstance(body, dict) else ""
            label = f"A2A {method}: \u201c{first_line(text, 70)}\u201d"
        self.emit(kind, "planner", lane, label, {"method": request.method, "url": str(request.url), "body": body},
                  phase="request")

    async def on_response(self, response: httpx.Response) -> None:
        await response.aread()
        request = response.request
        lane, kind = self.classify(request)
        ms = round((time.perf_counter() - request.extensions.get("flow_started", time.perf_counter())) * 1000, 1)
        data = parse_json(response.content)
        status = response.status_code

        if kind == "nanda" and isinstance(data, dict) and "match" in data:
            entry = data.get("entry") or {}
            label = (f"{status} {data.get('match')} match \u2192 {MEDIA_NAMES.get(entry.get('type'), entry.get('type'))}"
                     f" at {entry.get('url', '?')}")
        elif kind == "nanda" and isinstance(data, dict) and "candidates" in data:
            ranked = ", ".join(f"{c.get('displayName')} [{MEDIA_NAMES.get(c.get('type'), c.get('type'))}] "
                               f"{c.get('score', 0):.2f}" for c in data["candidates"])
            label = f"{status} {len(data['candidates'])} candidate(s): {ranked or 'none'}"
        elif kind == "nanda" and isinstance(data, dict) and "results" in data:
            label = f"{status} {data.get('count', len(data['results']))} result(s)"
        elif kind == "card" and isinstance(data, dict):
            skills = ", ".join(s.get("name", s.get("id", "")) for s in data.get("skills", []))
            label = f"{status} {data.get('name', 'agent card')} \u00b7 skills: {skills}"
        elif kind == "a2a" and isinstance(data, dict):
            self.emit_trace(data)
            if data.get("error"):
                label = f"A2A error {data['error'].get('code')}: {data['error'].get('message')}"
            else:
                result = data.get("result") or {}
                message = result.get("message") or (result.get("task") or {}).get("status", {}).get("message") or result
                label = f"reply: {first_line(message_text(message)) or '(no text)'}"
        else:
            label = f"HTTP {status}"
        self.emit(kind, lane, "planner", label, {"status": status, "body": data}, phase="response", ms=ms)

    def add_lane(self, key: str, role: str, name: str) -> str:
        """Map a trace lane key to a diagram lane, announcing new lanes (and role upgrades) to the page.

        Live traces can mention a server before naming it (the negotiator's card is read during
        discovery), so an unknown server first gets an "other" lane that a later lane entry upgrades."""
        lane = self.lane_ids.get(key)
        if lane is None:
            lane = "agent" if role == "agent" else self.lane_ids.get(role) if role == "negotiator" else None
            if lane is None:
                self.extra_lanes += 1
                lane = f"svc{self.extra_lanes}"
            self.lane_ids[key] = lane
        if role == "negotiator":
            self.lane_ids.setdefault("negotiator", lane)
        if lane in SHOWN_LANES:
            return lane
        known = self.lane_roles.get(lane)
        if known is None or (known == "other" and role != "other"):
            self.lane_roles[lane] = role
            self.emit("lane", "planner", "planner", name, lane=lane, role=role, url=key)
        return lane

    def lane_for(self, ref: str) -> str:
        if ref == "agent" and self.agent_lane:
            return self.agent_lane
        if ref in SHOWN_LANES:
            return ref
        if ref in self.lane_ids:
            return self.lane_ids[ref]
        if ref == "negotiator":
            return self.add_lane(ref, "negotiator", "Negotiator Agent")
        if not ref:  # traces from an older a2a_server.py don't name the MCP server
            return self.add_lane("", "seller", "Product Server")
        return self.add_lane(ref, "other", ref.split("//")[-1].rstrip("/"))

    def emit_entries(self, entries: list, live: bool = False) -> None:
        """Draw trace entries, live (posted by the agents) or from a reply; each entry is drawn once."""
        fresh = [e for e in entries if isinstance(e, dict) and not (e.get("id") and e["id"] in self.seen)]
        self.seen.update(e["id"] for e in fresh if e.get("id"))
        # Within one batch, lane entries can come after the first call to that server, so register them first.
        for entry in fresh:
            if entry.get("kind") == "lane" and entry.get("key") is not None:
                self.add_lane(entry["key"], entry.get("role", "other"), entry.get("name") or entry["key"])
        for entry in fresh:
            kind = entry.get("kind", "mcp")
            if kind == "lane":
                continue
            if kind == "startup":
                if entry.get("src", "agent") in self.reported_startup:
                    continue
                self.reported_startup.add(entry.get("src", "agent"))
            src, dst = self.lane_for(entry.get("src", "agent")), self.lane_for(entry.get("dst", ""))
            if entry.get("phase"):
                self.emit(kind, src, dst, entry.get("label", ""), entry.get("detail"), phase=entry["phase"],
                          ms=entry.get("ms"), live=live)
                continue
            self.emit(kind, src, dst, call_label(kind, entry),
                      {"tool": entry.get("tool"), "arguments": entry.get("arguments")}, phase="request", live=live)
            summary = entry.get("summary") or first_line(entry.get("result_preview", ""), 70)
            self.emit(kind, dst, src, f"{'ok' if entry.get('ok') else 'error'} \u00b7 {summary}",
                      {"summary": summary, "result_preview": entry.get("result_preview")},
                      phase="response", ms=entry.get("ms"), live=live)

    def emit_trace(self, data: dict) -> None:
        message = (data.get("result") or {}).get("message") or {}
        metadata = whole_numbers(message.get("metadata") or {})
        trace = metadata.get("trace")
        if not trace:
            if not self.warned_no_trace:
                self.warned_no_trace = True
                self.emit("note", "agent", "agent",
                          "This shopping agent doesn't report its MCP calls; restart a2a_server.py to see them")
            return
        entries = trace.get("mcp_calls", [])
        has_startup = any(isinstance(e, dict) and e.get("kind") == "startup" for e in entries)
        if ("agent" not in self.reported_startup and not has_startup
                and str(trace.get("product_server", "")).startswith("NANDA")):
            self.reported_startup.add("agent")
            self.emit("startup", "agent", "index",
                      "earlier, at startup: shopping agent resolved its home store through NANDA",
                      {"product_server": trace.get("product_server"), "agent_mode": trace.get("agent")})
        self.emit_entries(entries)
        for outcome in metadata.get("negotiations") or []:
            self.emit("deal", "agent", "agent", "Negotiation result (seller-issued quotes)", outcome)


async def discover_by_task(rec: FlowRecorder, http: httpx.AsyncClient, task: str, gifts: list[str],
                           use_claude: bool, job: str = "buy", prefix: str = "") -> DiscoveredAgent:
    """Agentic search for the task, then a capable candidate. Advice mode gives every candidate its own lane."""
    advice = job == "advice"
    rec.step(f"{prefix}1 \u00b7 Gift planner describes " + ("this request" if advice else "its job")
             + " to the NANDA Index (it knows no agent name or URN)")
    entries = await agentic_search_entries(http, task, A2A_CARD_TYPE, NANDA_INDEX_URL)
    if advice:
        for entry in entries:
            rec.add_lane(base_of(entry["url"]), "a2a", entry.get("displayName") or entry["identifier"])
    rec.step(f"{prefix}2 \u00b7 Gift planner reads the A2A agent card of each candidate ({len(entries)} A2A agent(s))")
    candidates = await read_candidates(http, entries)
    rec.step(f"{prefix}3 \u00b7 Gift planner keeps the agents whose card has a skill to {JOB_NAMES[job]}, then "
             + ("Claude compares their descriptions and skills and picks one" if use_claude
                else "picks the best-scored one (no LLM available)"))
    agent, reason, skipped = await pick_capable(candidates, job, task, gifts, use_claude, DEFAULT_MODEL)
    for c in skipped:
        rec.emit("note", "planner", "planner", f"Skipped {c.agent.name}: its card has no skill to {JOB_NAMES[job]}",
                 {"skills": c.agent.skills})
    lane = rec.lane_ids.get(base_of(agent.endpoint), "agent") if advice else "agent"
    rec.emit("choice", "planner", "planner", f"Chose {agent.name}: {reason}", {
        "chosen": {"identifier": agent.identifier, "name": agent.name, "endpoint": agent.endpoint, "skills": agent.skills},
        "reason": reason,
        "candidates": [{"name": c.agent.name, "identifier": c.agent.identifier, "score": round(c.score, 3),
                        "description": c.description, "skills": c.agent.skills, "capable": c not in skipped}
                       for c in candidates],
    }, agent_name=agent.name, endpoint=agent.endpoint, lane=lane)
    return agent


async def discover_by_name(rec: FlowRecorder, http: httpx.AsyncClient, mode: str, target: str,
                           job: str = "buy") -> DiscoveredAgent:
    rec.step("1 \u00b7 Gift planner looks up the shopping agent in the NANDA Index "
             + ("by keywords" if mode == "search" else "by its URN"))
    if mode == "search":
        records = await search_entries(http, target, A2A_CARD_TYPE, NANDA_INDEX_URL)
        if not records:
            raise DiscoveryError(f"The NANDA Index has no active A2A agent matching '{target}'")
        identifier, name, card_url = records[0]["identifier"], records[0]["display_name"], records[0]["registry_url"]
    else:
        entry = await resolve_entry(http, target, A2A_CARD_TYPE, NANDA_INDEX_URL)
        identifier, name, card_url = entry.get("identifier", target), entry.get("displayName", target), entry["url"]
    rec.step("2 \u00b7 Gift planner reads the A2A agent card the index pointed to")
    agent = await read_agent_card(http, identifier, name, card_url)
    require_skill(agent, job)
    rec.emit("choice", "planner", "planner", f"Using {agent.name} at {agent.endpoint}",
             {"chosen": {"identifier": agent.identifier, "name": agent.name, "endpoint": agent.endpoint}},
             agent_name=agent.name, endpoint=agent.endpoint)
    return agent


async def run_advice(rec: FlowRecorder, http: httpx.AsyncClient, gifts: list[str], prefix: str, use_claude: bool,
                     sink_url: str | None) -> None:
    """Each request is described to the index on its own and goes to the specialist it ranks best."""
    sections, links = [], []
    try:
        for number, request in enumerate(split_requests(gifts), 1):
            rec.step(f"Request {number}: \u201c{request}\u201d")
            agent = await discover_by_task(rec, http, advice_task(prefix, request), [request], use_claude, "advice",
                                           prefix=f"{number}.")
            rec.agent_lane = rec.lane_ids.get(base_of(agent.endpoint))
            rec.step(f"{number}.4 \u00b7 Gift planner asks {agent.name} for advice over A2A (read-only)")
            link = A2ALink(agent, http, them=agent.name, metadata={"trace_sink": sink_url} if sink_url else None)
            links.append(link)
            await link.connect()
            reply, metadata = await link.send(request)
            rec.emit("advice", "planner", "planner", f"Advice from {agent.name}",
                     {"request": request, "agent": agent.name, "reply": reply,
                      "advice": whole_numbers(metadata.get("advice") or {})}, lane=rec.agent_lane)
            rec.agent_lane = None
            sections.append(advice_section(number, request, agent.name, reply))
    finally:
        # Closing an A2A client closes the shared httpx client, so links are closed only at the end.
        for link in links:
            await link.close()
    rec.emit("result", "planner", "planner", "Gift planner result", {
        "summary": "Specialist advice (read-only: nothing was added to a cart or bought)\n\n" + "\n\n".join(sections)})


async def run_flow(rec: FlowRecorder, gifts: list[str], mode: str, target: str, checkout: bool, use_claude: bool,
                   job: str = "buy", sink_url: str | None = None) -> None:
    hooks = {"request": [rec.on_request], "response": [rec.on_response]}
    negotiate = job == "negotiate"
    async with httpx.AsyncClient(timeout=NEGOTIATION_TIMEOUT if negotiate else 180, event_hooks=hooks) as http:
        if job == "advice":
            return await run_advice(rec, http, gifts, target, use_claude, sink_url)
        if mode == "task":
            agent = await discover_by_task(rec, http, target, gifts, use_claude, job)
        else:
            agent = await discover_by_name(rec, http, mode, target, job)
        rec.lane_ids[base_of(agent.endpoint)] = "agent"

        how = "Claude plans each message from the card" if use_claude else "fixed plan"
        number = 4 if mode == "task" else 3
        if negotiate:
            rec.step(f"{number} \u00b7 Gift planner asks for the best negotiated price over A2A, quote only ({how}). "
                     "The shopping agent finds the sellers and a negotiator through NANDA by itself")
        else:
            rec.step(f"{number} \u00b7 Gift planner delegates the gift request(s) over A2A ({how})")
        # trace_sink: the agents post each step here as it happens, so the diagram updates live.
        metadata = {**(QUOTE_ONLY if negotiate else {}), **({"trace_sink": sink_url} if sink_url else {})}
        link = ShoppingAgentLink(agent, http, metadata=metadata)
        await link.connect()
        try:
            if negotiate:
                summary = await negotiate_plan(link, " ".join(gifts), use_claude, DEFAULT_MODEL)
            elif use_claude:
                summary = await claude_plan(link, gifts, checkout, DEFAULT_MODEL)
            else:
                summary = await fixed_plan(link, gifts, checkout)
        finally:
            await link.close()
        rec.emit("result", "planner", "planner", "Gift planner result", {"summary": summary, "conversation": link.context_id})


async def homepage(request: Request) -> HTMLResponse:
    return HTMLResponse(PAGE_FILE.read_text())


async def api_status(request: Request) -> JSONResponse:
    async def check(name: str, coro) -> dict:
        try:
            return {"name": name, "ok": True, "detail": await coro}
        except Exception as e:
            return {"name": name, "ok": False, "detail": str(e)}

    async def index_health() -> str:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{NANDA_INDEX_URL}/health")
            response.raise_for_status()
            return f"{NANDA_INDEX_URL} healthy"

    async def product() -> str:
        server = await resolve(PRODUCT_URN)
        return f"registered; MCP endpoint {server.endpoint}"

    async def server(urn: str) -> str:
        found = await resolve(urn)
        return f"registered; MCP endpoint {found.endpoint}"

    async def agent(urn: str) -> str:
        found = await resolve_agent(urn)
        return f"registered; A2A endpoint {found.endpoint}"

    checks = await asyncio.gather(
        check("NANDA Index", index_health()),
        check("Seller A (MCP)", server(PRODUCT_URN)),
        check("Seller B (MCP, negotiate mode)", server(SELLER_B_URN)),
        check("Shopping agent (A2A)", agent(AGENT_URN)),
        check("Negotiator (A2A, negotiate mode)", agent(NEGOTIATOR_URN)),
        *(check(name, agent(urn)) for name, urn in SPECIALIST_URNS.items()),
    )
    return JSONResponse({
        "index_url": NANDA_INDEX_URL, "agent_urn": AGENT_URN, "product_urn": PRODUCT_URN, "planner_task": PLANNER_TASK,
        "negotiate_task": NEGOTIATE_TASK, "advice_task": ADVICE_TASK,
        "claude_available": bool(os.environ.get("ANTHROPIC_API_KEY")), "checks": checks,
    })


async def api_run(request: Request) -> StreamingResponse:
    params = request.query_params
    job = params.get("job") if params.get("job") in ("negotiate", "advice") else "buy"
    negotiate = job == "negotiate"
    gifts = [g.strip() for g in params.get("gifts", "").splitlines() if g.strip()][:6]
    # Advice is routed per request by task, so it ignores the URN and keyword modes.
    mode = params.get("mode") if params.get("mode") in ("urn", "search") and job != "advice" else "task"
    target = params.get("target", "").strip() or {
        "urn": AGENT_URN, "search": "personal shopping agent",
        "task": {"negotiate": NEGOTIATE_TASK, "advice": ADVICE_TASK}.get(job, PLANNER_TASK)}[mode]
    checkout = params.get("checkout") == "1" and job == "buy"
    use_claude = params.get("claude") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))

    queue: asyncio.Queue = asyncio.Queue()
    rec = FlowRecorder(queue)
    run_id = uuid.uuid4().hex
    RUNS[run_id] = rec
    sink_url = f"http://127.0.0.1:{PORT}/api/trace/{run_id}"

    async def worker() -> None:
        try:
            if not gifts:
                raise ValueError("Describe what to shop for" if negotiate else "Add at least one gift request")
            await run_flow(rec, gifts, mode, target, checkout, use_claude, job, sink_url)
        except DiscoveryError as e:
            rec.emit("error", "planner", "planner", f"Discovery failed: {e}")
        except Exception as e:
            rec.emit("error", "planner", "planner", f"{type(e).__name__}: {e}")
        finally:
            RUNS.pop(run_id, None)
            queue.put_nowait(None)

    task = asyncio.create_task(worker())

    async def stream():
        try:
            while (event := await queue.get()) is not None:
                yield f"data: {json.dumps(event, default=str)}\n\n"
            yield "event: done\ndata: {}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


async def api_trace(request: Request) -> Response:
    """Live trace entries from the agents of a running flow (see tracing.TraceSink)."""
    rec = RUNS.get(request.path_params["run_id"])
    if rec is None:
        return Response(status_code=404)
    try:
        entries = await request.json()
    except ValueError:
        return Response(status_code=400)
    if isinstance(entries, list):
        rec.emit_entries(whole_numbers(entries), live=True)
    return Response(status_code=204)


app = Starlette(routes=[
    Route("/", homepage),
    Route("/api/status", api_status),
    Route("/api/run", api_run),
    Route("/api/trace/{run_id}", api_trace, methods=["POST"]),
])

if __name__ == "__main__":
    print(f"Flow viewer: http://localhost:{PORT}  (NANDA Index {NANDA_INDEX_URL})")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
