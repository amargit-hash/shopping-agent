"""End-to-end run of the two-seller negotiation and the specialist advice over a real NANDA Index, A2A and MCP,
isolated from the demo.

It creates a throwaway index database (nanda_index_e2e) in the demo's PostgreSQL (:5434, must be
running), starts its own index on :3011 and every service on spare ports, registers and verifies
them, runs the gift planner's --negotiate and --advice modes and the flow viewer's negotiate and advice
runs, then stops
everything and drops the database. Claude is replaced by the scripted stand-in in
tests/scripted_claude, so no API key is used and the decisions are deterministic.

    cd shopping_agent && ../.venv/bin/python tests/e2e_negotiation.py
"""
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(APP))
from negotiation_report import CHAT_HEADER, COMPARISON_HEADER  # noqa: E402

INDEX_DIR = APP.parent / "nanda-index-v2" / "server"
PY = sys.executable
DB = "nanda_index_e2e"
PG = ["-h", "localhost", "-p", "5434", "-U", "nanda"]
INDEX_URL = "http://localhost:3011"
PORTS = {"seller_a": 8091, "seller_b": 8092, "agent": 8093, "flow": 8094, "negotiator": 8097,
         "laptop": 8095, "gift": 8096, "travel": 8098}
SPECIALISTS = {"laptop": "Laptop Advisor Agent", "gift": "Gift Ideas Agent", "travel": "Travel Gear Agent"}
EMAILS = {"laptop": "laptop-advisor@nanda.local", "gift": "gift-ideas@nanda.local", "travel": "travel-gear@nanda.local"}
# Each request should be routed to the specialist for it.
ADVICE = {"a laptop for video editing under 1200": "laptop", "a birthday gift for my mom under 50": "gift",
          "headphones for a long flight under 200": "travel"}
REQUEST = "the best gaming laptop under 1500"
URN = "urn:air:host39.org:personal:{}@nanda.local".format
LOGS = Path(tempfile.mkdtemp(prefix="nanda-e2e-"))
PROCESSES: list[subprocess.Popen] = []

ENV = {**os.environ, "PYTHONPATH": str(HERE / "scripted_claude"), "ANTHROPIC_API_KEY": "scripted-test-key",
       "NANDA_INDEX_URL": INDEX_URL, "PYTHONUNBUFFERED": "1"}
# The accounts this test registers live only in the throwaway database it drops at the end,
# so the password is generated per run rather than kept in the file.
PASSWORD = os.environ.get("E2E_NANDA_PASSWORD") or secrets.token_hex(12)


def psql(sql: str, db: str = "nanda_index") -> str:
    return subprocess.run(["psql", *PG, "-d", db, "-Atc", sql], check=True, capture_output=True, text=True).stdout.strip()


def start(name: str, args: list[str], cwd: Path = APP, **env: str) -> None:
    log = open(LOGS / f"{name}.log", "w")
    PROCESSES.append(subprocess.Popen(args, cwd=cwd, env={**ENV, **env}, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True))


def wait_for(url: str, name: str, seconds: int = 40) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"{name} did not start; see {LOGS}/{name}.log")


def register(kind: str, email: str, name: str, card_url: str) -> str:
    result = subprocess.run(
        [PY, "nanda_register.py", "--kind", kind, "--email", email, "--create-account", "--name", name,
         "--card-url", card_url],
        cwd=APP, env={**ENV, "NANDA_PASSWORD": PASSWORD}, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise SystemExit(f"Registering {name} failed:\n{result.stdout}\n{result.stderr}")
    token = psql(f"select verify_token from organizations where contact_email = '{email}' and status = 'pending'", DB)
    httpx.get(f"{INDEX_URL}/api/v1/verify-email", params={"token": token}, timeout=10).raise_for_status()
    match = httpx.get(f"{INDEX_URL}/api/v1/resolve", params={"locator": URN(email.split("@")[0])}, timeout=10).json()
    assert match.get("match") == "exact", match
    print(f"  registered {name} as {URN(email.split('@')[0])}")
    return URN(email.split("@")[0])


def check(condition: bool, message: str) -> None:
    print(("  PASS " if condition else "  FAIL ") + message)
    if not condition:
        raise SystemExit(f"End-to-end check failed: {message} (logs in {LOGS})")


def main() -> None:
    for port in [3011, *PORTS.values()]:
        try:
            httpx.get(f"http://localhost:{port}/", timeout=1)
            raise SystemExit(f"Port {port} is already in use; stop whatever runs there first")
        except httpx.ConnectError:
            pass
    print(f"Logs: {LOGS}")
    psql(f"drop database if exists {DB}")
    psql(f"create database {DB}")
    index_env = {"DATABASE_URL": f"postgresql://nanda@localhost:5434/{DB}", "PORT": "3011", "API_BASE_URL": INDEX_URL}
    subprocess.run(["node_modules/.bin/tsx", "--env-file=.env", "src/db/migrate.ts"], cwd=INDEX_DIR,
                   env={**ENV, **index_env}, check=True, capture_output=True)
    start("index", ["node_modules/.bin/tsx", "--env-file=.env", "src/server.ts"], cwd=INDEX_DIR, **index_env)
    wait_for(f"{INDEX_URL}/health", "index")

    print("1. Sellers and negotiator")
    card = "http://localhost:{}/.well-known/mcp-server-card.json".format
    # The same seller settings as start_nanda_demo.sh: own SKUs, exclusives and other configurations.
    start("seller_a", [PY, "product_server.py"], PORT=str(PORTS["seller_a"]), SELLER_ID="store-a",
          SELLER_NAME="Harbor Electronics", SELLER_CATALOG_SHARE="0.9")
    start("seller_b", [PY, "product_server.py"], PORT=str(PORTS["seller_b"]), SELLER_ID="store-b",
          SELLER_NAME="Summit Gadgets", SELLER_PRICE_SPREAD="0.08", SELLER_FLOOR_FRACTION="0.88",
          SELLER_SKU_PREFIX="SG-", SELLER_CATALOG_SHARE="0.85", SELLER_VARIANT_SHARE="0.35")
    start("negotiator", [PY, "negotiator_agent.py"], NEGOTIATOR_PORT=str(PORTS["negotiator"]))
    wait_for(card(PORTS["seller_a"]), "seller_a")
    wait_for(card(PORTS["seller_b"]), "seller_b")
    negotiator_card = f"http://localhost:{PORTS['negotiator']}/.well-known/agent-card.json"
    wait_for(negotiator_card, "negotiator")
    seller_a = register("seller", "seller-a@nanda.local", "Seller A", card(PORTS["seller_a"]))
    seller_b = register("seller", "seller-b@nanda.local", "Seller B", card(PORTS["seller_b"]))
    negotiator = register("negotiator", "negotiator@nanda.local", "Price Negotiator Agent", negotiator_card)

    print("2. Shopping agent (finds its home store through NANDA)")
    start("agent", [PY, "a2a_server.py"], A2A_PORT=str(PORTS["agent"]), NANDA_LOCATOR=seller_a)
    agent_card = f"http://localhost:{PORTS['agent']}/.well-known/agent-card.json"
    wait_for(agent_card, "agent")
    agent = register("a2a", "shopping-agent@nanda.local", "Personal Shopping Agent", agent_card)

    print("2b. Specialists (find their store through NANDA; the travel one runs without Claude)")
    specialist_card = "http://localhost:{}/.well-known/agent-card.json".format
    for profile in SPECIALISTS:
        extra = {"ANTHROPIC_API_KEY": ""} if profile == "travel" else {}
        start(profile, [PY, "specialist_agent.py", "--profile", profile, "--port", str(PORTS[profile])],
              NANDA_LOCATOR=seller_a, **extra)
    for profile in SPECIALISTS:
        wait_for(specialist_card(PORTS[profile]), profile)
        register("specialist", EMAILS[profile], SPECIALISTS[profile], specialist_card(PORTS[profile]))

    print("3. Gift planner --negotiate (discovers the shopping agent by task)")
    result = subprocess.run([PY, "gift_planner_agent.py", "--negotiate", REQUEST], cwd=APP, env=ENV,
                            capture_output=True, text=True, timeout=300)
    (LOGS / "gift_planner.log").write_text(result.stdout + result.stderr)
    out = result.stdout
    check(result.returncode == 0, "gift planner exited cleanly")
    check("Personal Shopping Agent" in out, "planner chose the shopping agent found through NANDA")
    check("Harbor Electronics" in out and "Summit Gadgets" in out, "result lists both sellers")
    check("Quote only: nothing was bought" in out, "result is quote only")
    check("(quote only)" in out, "requests were sent as quote only")
    final = out.split("=== Gift planner result ===", 1)[-1]
    check(final.count(COMPARISON_HEADER) == 1, "final answer shows the spec and price comparison once")
    check(bool(re.search(r"^\| SKU \| P\d+ \| SG-\d{5} \|$", final, re.M)), "each seller's candidate has its own SKU")
    check(bool(re.search(r"^\| ram_gb \|", final, re.M)), "comparison lists the specs")
    check(bool(re.search(r"^Negotiator's pick: (Harbor Electronics|Summit Gadgets), ", final, re.M)),
          "final answer names the negotiator's pick")
    check(bool(re.search(r"^Why: \S", final, re.M)), "the pick comes with a reason")
    check(final.count(CHAT_HEADER) == 1, "final answer shows the negotiation chat once")
    chat_lines = [l for l in final.splitlines() if l.startswith("- Negotiator -> ")]
    check(len(chat_lines) > 0 and {"Harbor Electronics", "Summit Gadgets"} <= {l.split(" -> ")[1].split(" (")[0]
                                                                                for l in chat_lines},
          f"chat has offers to both sellers ({len(chat_lines)} offers)")

    print("4. Flow viewer, negotiate run")
    start("flow", [PY, "flow_ui.py"], FLOW_UI_PORT=str(PORTS["flow"]), AGENT_URN=agent, PRODUCT_URN=seller_a,
          SELLER_B_URN=seller_b, NEGOTIATOR_URN=negotiator)
    # (the specialist URNs are the flow viewer's defaults: the same emails)
    flow = f"http://localhost:{PORTS['flow']}"
    wait_for(flow + "/", "flow")
    status = httpx.get(flow + "/api/status", timeout=20).json()
    check(all(c["ok"] for c in status["checks"]) and len(status["checks"]) == 8,
          "status page: index, both sellers, shopping agent, negotiator and three specialists registered")
    events = []
    params = {"job": "negotiate", "gifts": REQUEST, "mode": "task", "target": "", "claude": "1"}
    with httpx.stream("GET", flow + "/api/run", params=params, timeout=300) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: {}":
                events.append(json.loads(line[6:]))
    (LOGS / "flow_events.json").write_text(json.dumps(events, indent=1))
    lanes = {e["lane"]: (e["role"], e["label"]) for e in events if e["kind"] == "lane"}
    roles = sorted(role for role, _ in lanes.values())
    check(roles == ["negotiator", "seller", "seller"], f"diagram has a negotiator lane and two seller lanes {roles}")
    offers = [e for e in events if e["kind"] == "mcp" and e["phase"] == "request"
              and (e.get("detail") or {}).get("tool") == "make_offer"]
    seller_lanes = {lanes[e["to"]][1] for e in offers}
    check(bool(offers) and all(lanes[e["from"]][0] == "negotiator" for e in offers),
          f"{len(offers)} offers drawn from the negotiator lane")
    check(seller_lanes == {"Harbor Electronics", "Summit Gadgets"}, "offers went to both sellers")
    check(all(e.get("live") for e in offers), "offers were drawn live, as they happened")
    shortlisted = {lanes.get(e["to"], ("", ""))[1] for e in events if e["kind"] == "mcp" and e["phase"] == "request"
                   and (e.get("detail") or {}).get("tool") == "compare_products"
                   and lanes.get(e["from"], ("", ""))[0] == "negotiator"}
    check(shortlisted == {"Harbor Electronics", "Summit Gadgets"}, "negotiator shortlisted products at both sellers")
    deal = next((e for e in events if e["kind"] == "deal"), None)
    check(deal is not None and len(deal["detail"]["offers"]) == 2 and all(o["available"] for o in deal["detail"]["offers"]),
          "result card with a quote from each seller")
    detail = deal["detail"]
    skus = [o["product"]["sku"] for o in detail["offers"]]
    check(skus[0].startswith("P") and skus[1].startswith("SG-"), f"candidates use each seller's SKUs {skus}")
    check(all(o["product"]["specs"] for o in detail["offers"]), "candidates carry the sellers' specs")
    check(detail["winner"] in ("Harbor Electronics", "Summit Gadgets") and detail["decided_by"] == "negotiator"
          and detail["winner_reason"], f"negotiator picked {detail['winner']}: {detail['winner_reason']}")
    check(all(o["offers_made"] > 0 for o in detail["offers"]), "the negotiator made offers to every seller")
    check(any(o["savings"] > 0 for o in detail["offers"]), "at least one seller came down from its list price")
    made = sum(o["offers_made"] for o in deal["detail"]["offers"])
    check(len(offers) == made, f"each offer drawn once ({len(offers)} drawn, {made} made)")
    finals = {o["seller"]: (o["list_price"], o["final_price"]) for o in deal["detail"]["offers"]}
    check(all(final <= listed for listed, final in finals.values()), f"final prices at or below list: {finals}")
    check(len(deal["detail"].get("chat") or []) == made, "result card carries the negotiation chat")
    check(not any(e["kind"] == "error" for e in events), "no errors in the flow")
    result_event = next((e for e in events if e["kind"] == "result"), None)
    check(result_event is not None, "planner result shown")
    summary = result_event["detail"]["summary"]
    check(CHAT_HEADER in summary and COMPARISON_HEADER in summary,
          "flow viewer's final answer includes the comparison and the chat")
    advice_checks(flow, specialist_card(PORTS["laptop"]))
    print("All end-to-end checks passed.")
    if "--keep" in sys.argv:
        print(f"Services stay up until Ctrl+C; flow viewer: {flow}/?job=negotiate&claude=1&run=1", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


def advice_checks(flow: str, laptop_card: str) -> None:
    print("5. Gift planner --advice (each request routed to its specialist by task)")
    result = subprocess.run([PY, "gift_planner_agent.py", "--advice", *ADVICE], cwd=APP, env=ENV,
                            capture_output=True, text=True, timeout=180)
    (LOGS / "gift_planner_advice.log").write_text(result.stdout + result.stderr)
    check(result.returncode == 0, "advice run exited cleanly")
    final = result.stdout.split("=== Gift planner result ===", 1)[-1]
    for number, (request, profile) in enumerate(ADVICE.items(), 1):
        check(f"{number}. \u201c{request}\u201d, answered by {SPECIALISTS[profile]}:" in final,
              f"\u201c{request}\u201d went to {SPECIALISTS[profile]}")
    check(bool(re.search(r"Travel Gear Agent's advice .*\n1\. P\d+", final)), "travel specialist used the rule-based advisor")
    check("Laptop Advisor Agent recommends:" in final, "laptop specialist used (scripted) Claude")
    check("read-only" in final, "advice is read-only")

    print("6. Capability checks")
    result = subprocess.run([PY, "gift_planner_agent.py", "--card-url", laptop_card, "--checkout", "headphones under 100"],
                            cwd=APP, env=ENV, capture_output=True, text=True, timeout=60)
    check(result.returncode != 0 and "can't build a cart and check out" in result.stderr,
          "a specialist is refused for checkout")
    result = subprocess.run([PY, "gift_planner_agent.py", "--no-claude", "books under 30"], cwd=APP, env=ENV,
                            capture_output=True, text=True, timeout=120)
    (LOGS / "gift_planner_buy.log").write_text(result.stdout + result.stderr)
    check(result.returncode == 0 and "Personal Shopping Agent" in result.stdout.split("[planner]")[0],
          "buying by task still goes to the shopping agent")

    print("7. Flow viewer, advice run")
    events = []
    params = {"job": "advice", "gifts": "\n".join(ADVICE), "mode": "task", "target": "", "claude": "1"}
    with httpx.stream("GET", flow + "/api/run", params=params, timeout=180) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: {}":
                events.append(json.loads(line[6:]))
    (LOGS / "flow_advice_events.json").write_text(json.dumps(events, indent=1))
    lanes = {e["lane"]: (e["role"], e["label"]) for e in events if e["kind"] == "lane"}
    check(not any(e["kind"] == "error" for e in events), "no errors in the advice flow")
    check(set(SPECIALISTS.values()) <= {name for role, name in lanes.values() if role == "a2a"},
          "each specialist has its own lane")
    advice = [e for e in events if e["kind"] == "advice"]
    check([e["detail"]["agent"] for e in advice] == [SPECIALISTS[p] for p in ADVICE.values()],
          "each request was answered by its specialist")
    check(all(e["detail"]["advice"].get("picks") for e in advice), "every answer carries the specialist's picks")
    by_lane = {lane: name for lane, (_, name) in lanes.items()}
    calls = [e for e in events if e["kind"] == "mcp" and e["phase"] == "request"]
    check({by_lane.get(e["from"]) for e in calls} == set(SPECIALISTS.values())
          and all(lanes[e["to"]][0] == "seller" for e in calls),
          f"{len(calls)} read-only MCP calls drawn from the specialist lanes to the store")
    check(all((e.get("detail") or {}).get("tool") in ("list_categories", "recommend_products", "compare_products",
                                                      "search_products", "get_product", "check_availability", "tools/list")
              for e in calls), "specialists only used read-only catalog tools")
    a2a = [e for e in events if e["kind"] == "a2a" and e["phase"] == "request"]
    check(len(a2a) == 3 and {by_lane.get(e["to"]) for e in a2a} == set(SPECIALISTS.values()),
          "one A2A message to each specialist")
    result_event = next((e for e in events if e["kind"] == "result"), None)
    check(result_event is not None and "Specialist advice" in result_event["detail"]["summary"], "advice result shown")


def cleanup() -> None:
    for process in PROCESSES:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in PROCESSES:
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    try:
        psql(f"drop database if exists {DB}")
    except subprocess.CalledProcessError as e:
        print(f"Could not drop {DB}: {e.stderr}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
