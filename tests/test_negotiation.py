"""Unit tests for the multi-seller negotiation pieces. No network and no Claude: model calls are mocked.

    cd shopping_agent && ../.venv/bin/python -m unittest discover -s tests -v
"""
import asyncio
import json
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_ui  # noqa: E402
import negotiator_agent  # noqa: E402
import product_server  # noqa: E402
from marketplace import MarketplaceShoppingAgent, is_seller_card  # noqa: E402
from nanda_discovery import DiscoveredAgent  # noqa: E402
from negotiation_report import CHAT_HEADER, COMPARISON_HEADER, attach_report, format_chat  # noqa: E402
from shopping_agent import PRODUCT_ID_RE  # noqa: E402
from tracing import base_of, current_trace, record_lane  # noqa: E402

URN_A = "urn:air:host39.org:personal:seller-a@nanda.local"
URN_B = "urn:air:host39.org:personal:seller-b@nanda.local"


def negotiation(ask=100_000, floor=90_000):
    return {"ask_cents": ask, "floor_cents": floor}


class ApplyDecisionTests(unittest.TestCase):
    """The seller's limits are enforced in code, whatever the model says."""

    def test_counter_below_floor_is_raised_to_floor(self):
        out = product_server.apply_decision(negotiation(), 80_000, {"action": "counter", "price": 700})
        self.assertEqual(out, {"action": "counter", "price_cents": 90_000})

    def test_counter_above_ask_becomes_hold(self):
        out = product_server.apply_decision(negotiation(), 85_000, {"action": "counter", "price": 1200})
        self.assertEqual(out, {"action": "hold", "price_cents": 100_000})

    def test_counter_at_or_below_offer_accepts_the_offer(self):
        out = product_server.apply_decision(negotiation(), 95_000, {"action": "counter", "price": 940})
        self.assertEqual(out, {"action": "accept", "price_cents": 95_000})

    def test_accept_below_floor_becomes_counter(self):
        out = product_server.apply_decision(negotiation(), 50_000, {"action": "accept"})
        self.assertEqual(out["action"], "counter")
        self.assertGreaterEqual(out["price_cents"], 90_000)

    def test_offer_at_ask_is_accepted_at_ask(self):
        out = product_server.apply_decision(negotiation(), 120_000, {"action": "hold"})
        self.assertEqual(out, {"action": "accept", "price_cents": 100_000})

    def test_unknown_action_holds(self):
        out = product_server.apply_decision(negotiation(), 95_000, {"action": "give it away", "price": 1})
        self.assertEqual(out, {"action": "hold", "price_cents": 100_000})

    def test_bad_counter_price_holds(self):
        out = product_server.apply_decision(negotiation(), 95_000, {"action": "counter", "price": "cheap"})
        self.assertEqual(out, {"action": "hold", "price_cents": 100_000})


class SellerServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        product_server.NEGOTIATIONS.clear()
        self.pid = next(p["id"] for p in product_server.PRODUCTS if p["stock"] > 2)

    def test_public_view_hides_the_floor(self):
        view = product_server.start_negotiation(self.pid)
        self.assertNotIn("floor", json.dumps(view))
        self.assertEqual(view["quote"]["unit_price"], view["list_price"])

    def test_start_negotiation_validates_input(self):
        self.assertIn("error", product_server.start_negotiation("P99999"))
        self.assertIn("error", product_server.start_negotiation(self.pid, quantity=0))

    async def test_three_offers_then_final(self):
        view = product_server.start_negotiation(self.pid)
        low = round(view["list_price"] * 0.5, 2)
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
                mock.patch.object(product_server, "seller_decision",
                                  mock.AsyncMock(return_value={"action": "counter", "price": 0.01, "message": "hm"})):
            for _ in range(3):
                reply = await product_server.make_offer(view["negotiation_id"], low)
            refused = await product_server.make_offer(view["negotiation_id"], low)
        self.assertEqual(reply["status"], "final")
        self.assertIn("error", refused)
        neg = product_server.NEGOTIATIONS[view["negotiation_id"]]
        self.assertEqual(neg["ask_cents"], neg["floor_cents"])  # the model's 1-cent counter was clamped

    async def test_api_error_does_not_use_an_offer(self):
        view = product_server.start_negotiation(self.pid)
        failing = mock.AsyncMock(side_effect=product_server.anthropic.APIError("down", request=mock.Mock(), body=None))
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}), \
                mock.patch.object(product_server, "seller_decision", failing):
            reply = await product_server.make_offer(view["negotiation_id"], 1.0)
        self.assertIn("error", reply)
        self.assertEqual(product_server.NEGOTIATIONS[view["negotiation_id"]]["offers_made"], 0)

class CatalogTests(unittest.TestCase):
    base = [{"id": f"P{n:03d}", "name": f"Laptop {n}", "brand": "Acme", "category": "laptops", "price": 1000.0,
             "specs": {"ram_gb": 16, "storage_gb": 512}} for n in range(1, 201)]

    def test_default_seller_keeps_the_catalog(self):
        catalog = product_server.build_catalog(self.base, "store-a")
        self.assertEqual([p["id"] for p in catalog], [p["id"] for p in self.base])
        self.assertEqual({p["price"] for p in catalog}, {1000.0})
        self.assertEqual(self.base[0]["specs"], {"ram_gb": 16, "storage_gb": 512})  # base left untouched

    def test_own_skus_exclusives_and_variants(self):
        a = {p["name"]: p for p in product_server.build_catalog(self.base, "store-a", share=0.9)}
        b = {p["name"]: p for p in product_server.build_catalog(self.base, "store-b", "SG-", 0.08, 0.85, 0.35)}
        self.assertTrue(all(re.fullmatch(r"SG-1\d{4}", p["id"]) for p in b.values()))
        self.assertEqual(len({p["id"] for p in b.values()}), len(b))
        both = set(a) & set(b)
        self.assertTrue(set(a) - set(b) and set(b) - set(a), "each seller has models the other doesn't carry")
        same = [n for n in both if a[n]["mpn"] == b[n]["mpn"]]
        variants = [n for n in both if a[n]["mpn"] != b[n]["mpn"]]
        self.assertTrue(same and variants)
        self.assertTrue(all(a[n]["specs"] == b[n]["specs"] for n in same))
        self.assertTrue(all(a[n]["specs"] != b[n]["specs"] for n in variants))
        self.assertTrue(all(920 <= b[n]["price"] <= 1080 for n in same))
        again = product_server.build_catalog(self.base, "store-b", "SG-", 0.08, 0.85, 0.35)
        self.assertEqual([p["id"] for p in again], [p["id"] for p in b.values()])  # deterministic

    def test_product_ids_cover_seller_skus(self):
        self.assertEqual(PRODUCT_ID_RE.findall("compare P123 and sg-10042, not SG-1"), ["P123", "sg-10042"])


class ParseRequestTests(unittest.TestCase):
    def test_valid_request(self):
        req = negotiator_agent.parse_request(
            'please: {"request": "best gaming laptop", "category": "Laptops", "keywords": "gaming", '
            '"sellers": ["%s", "%s"], "max_budget": 900, "quantity": 2.0}' % (URN_A, URN_B))
        self.assertEqual((req["request"], req["category"], req["keywords"]), ("best gaming laptop", "laptops", "gaming"))
        self.assertEqual(req["quantity"], 2)
        self.assertEqual(req["max_budget"], 900.0)

    def test_rejects_bad_requests(self):
        bad = [
            "no json here",
            '{"sellers": ["%s", "%s"]}' % (URN_A, URN_B),
            '{"request": "laptop", "sellers": ["%s"]}' % URN_A,
            '{"request": "laptop", "sellers": ["%s", "%s"]}' % (URN_A, URN_A),
            '{"request": "laptop", "sellers": ["http://localhost:8081", "%s"]}' % URN_B,
            '{"request": "laptop", "sellers": ["%s", "%s"], "quantity": 11}' % (URN_A, URN_B),
            '{"request": "laptop", "sellers": ["%s", "%s"], "max_budget": -5}' % (URN_A, URN_B),
            '{"request": "laptop", "category": 5, "sellers": ["%s", "%s"]}' % (URN_A, URN_B),
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                negotiator_agent.parse_request(text)


REQUEST = {"request": "best laptop", "category": "laptops", "keywords": "", "quantity": 1, "max_budget": 950,
           "notes": "", "sellers": [URN_A, URN_B]}


def seller_state(name, list_price, final, sku="P001", mpn="MPN-1", ram=16, offers=2, status="final"):
    return {"name": name, "identifier": f"urn:air:x:{name}", "list_price": list_price, "offers_made": offers,
            "status": status, "quote": {"unit_price": final, "total": final, "quote_id": f"Q-{name}"},
            "candidate": {"sku": sku, "name": "Book", "brand": "Acme", "category": "laptops", "mpn": mpn,
                          "rating": 4.5, "specs": {"ram_gb": ram}}}


class OutcomeTests(unittest.TestCase):
    def test_negotiators_pick_wins_even_if_not_cheapest(self):
        states = [seller_state("A", 1000, 930, mpn="MPN-1", ram=32), seller_state("B", 980, 910, "SG-10001", "MPN-2")]
        out = negotiator_agent.build_outcome(REQUEST, states, pick=(1, "$20 more for twice the RAM."))
        self.assertEqual((out["winner"], out["best_price"], out["decided_by"]), ("A", 930, "negotiator"))
        self.assertEqual(out["lowest_price"], 910)
        self.assertFalse(out["same_model"])
        self.assertEqual(out["offers"][0]["savings"], 70)
        text = negotiator_agent.format_outcome(out)
        self.assertIn("| ram_gb | 32 | 16 |", text)
        self.assertIn("| SKU | P001 | SG-10001 |", text)
        self.assertIn("Negotiator's pick: A, Book (P001) at $930.00", text)
        self.assertIn("Why: $20 more for twice the RAM.", text)

    def test_pick_over_budget_or_missing_falls_back_to_lowest_price(self):
        states = [seller_state("A", 1000, 960), seller_state("B", 980, 940),
                  {"name": "C", "identifier": "urn:air:x:C", "error": "down"}]
        for pick in ((1, "A is nicer"), None, (3, "C")):
            with self.subTest(pick=pick):
                out = negotiator_agent.build_outcome(REQUEST, states, pick=pick)
                self.assertEqual(out["winner"], "B")
                self.assertEqual(out["decided_by"], negotiator_agent.FALLBACK)
        self.assertTrue(out["same_model"])
        self.assertFalse(out["offers"][2]["available"])

    def test_no_pick_and_nothing_within_budget(self):
        out = negotiator_agent.build_outcome(REQUEST, [seller_state("A", 1000, 990), seller_state("B", 999, 970)],
                                             pick=(0, "Nothing fits."))
        self.assertIsNone(out["winner"])
        self.assertIn("Negotiator's pick: none. Nothing fits.", negotiator_agent.format_outcome(out))


class ChatReportTests(unittest.TestCase):
    outcome = {"chat": [
        {"seller": "Summit Gadgets", "round": 1.0, "buyer_offer": 820.0, "buyer_message": "Can you do $820?",
         "seller_action": "counter", "seller_price": 865.0, "seller_message": "Best I can do is $865."},
        {"seller": "Harbor Electronics", "round": 1.0, "buyer_offer": 830.0, "buyer_message": "",
         "seller_action": "hold", "seller_price": 905.0, "seller_message": ""},
    ]}

    def test_format_chat(self):
        self.assertEqual(format_chat(self.outcome).splitlines(), [
            CHAT_HEADER,
            '- Negotiator -> Summit Gadgets (offer 1): $820.00 "Can you do $820?"',
            '  Summit Gadgets: countered at $865.00 "Best I can do is $865."',
            "- Negotiator -> Harbor Electronics (offer 1): $830.00",
            "  Harbor Electronics: held at $905.00",
        ])
        self.assertEqual(format_chat({"chat": []}), "")

    def test_attach_report_once(self):
        text = attach_report("Summit Gadgets wins at $865.", [self.outcome])
        self.assertTrue(text.startswith("Summit Gadgets wins at $865.\n\n" + CHAT_HEADER))
        self.assertEqual(attach_report(text, [self.outcome]), text)
        self.assertEqual(attach_report("No deal.", [{"chat": []}]), "No deal.")

    def test_attach_report_adds_the_missing_comparison(self):
        outcome = negotiator_agent.build_outcome(REQUEST, [seller_state("A", 1000, 930), seller_state("B", 980, 910)],
                                                 self.outcome["chat"], (2, "Cheapest for the same model."))
        reply = negotiator_agent.format_outcome(outcome)
        self.assertEqual(attach_report(reply, [outcome]), reply)  # already complete
        shortened = attach_report("B wins.\n\n" + format_chat(outcome), [outcome])
        self.assertEqual((shortened.count(COMPARISON_HEADER), shortened.count(CHAT_HEADER)), (1, 1))


def laptop(sku, name, price, mpn, ram=16, category="laptops", stock=5, rating=4.5):
    return {"id": sku, "name": name, "brand": "Acme", "category": category, "mpn": mpn, "price": price,
            "rating": rating, "reviews": 10, "stock": stock, "specs": {"ram_gb": ram}}


class FakeSellerSession:
    """Just enough of a seller's MCP server for Negotiation: a catalog, a fixed floor and code-enforced replies."""

    def __init__(self, name, products, floor_fraction=0.9):
        self.name, self.products, self.floor_fraction = name, {p["id"]: p for p in products}, floor_fraction
        self.ask = self.list_cents = self.floor = 0
        self.offers, self.status, self.offer_prices, self.started = 0, "open", [], []

    def view(self):
        price = self.ask / 100
        return {"negotiation_id": f"N-{self.name}", "list_price": self.list_cents / 100, "status": self.status,
                "offers_made": self.offers, "offers_left": 3 - self.offers,
                "quote": {"unit_price": price, "total": price, "quote_id": f"Q-{self.name}-{self.ask}"}}

    async def call_tool(self, tool, arguments):
        if tool == "search_products":
            hits = [p for p in self.products.values()
                    if (not arguments.get("category") or p["category"] == arguments["category"])
                    and (not arguments.get("max_price") or p["price"] <= arguments["max_price"])
                    and arguments.get("query", "").lower() in p["name"].lower()]
            data = {"products": [{"id": p["id"], "name": p["name"]} for p in hits][: arguments.get("limit", 10)]}
        elif tool == "compare_products":
            data = {"comparison": [{**self.products[i], "in_stock": self.products[i]["stock"] > 0}
                                   for i in arguments["product_ids"] if i in self.products]}
        elif tool == "start_negotiation":
            product = self.products.get(arguments["product_id"])
            if product is None:
                data = {"error": "not found"}
            else:
                self.started.append(product["id"])
                self.ask = self.list_cents = int(round(product["price"] * 100))
                self.floor, self.offers, self.status = int(self.list_cents * self.floor_fraction), 0, "open"
                data = self.view()
        elif tool == "get_negotiation":
            data = self.view()
        elif tool == "make_offer":
            offer = int(round(arguments["offer_price"] * 100))
            self.offer_prices.append(arguments["offer_price"])
            out = product_server.apply_decision({"ask_cents": self.ask, "floor_cents": self.floor}, offer,
                                                {"action": "counter", "price": 0})
            self.offers += 1
            self.ask = out["price_cents"]
            self.status = "accepted" if out["action"] == "accept" else ("final" if self.offers >= 3 else "open")
            data = {"seller_reply": {"action": out["action"], "message": "ok"}, **self.view()}
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))])


def tool_use(name, **arguments):
    return SimpleNamespace(type="tool_use", id=f"t{id(arguments)}", name=name, input=arguments,
                           model_dump=lambda **_: {"type": "tool_use", "name": name, "input": arguments})


async def negotiate(states, script, request=None):
    """Shortlist and bargain with a scripted Claude; returns the Negotiation and the mocked create()."""
    responses = [SimpleNamespace(content=c, stop_reason="tool_use") for c in script]
    with mock.patch.object(negotiator_agent.anthropic, "AsyncAnthropic") as client:
        create = client.return_value.messages.create = mock.AsyncMock(side_effect=responses)
        neg = negotiator_agent.Negotiation(dict(request or REQUEST))
        for st in states:
            await neg.shortlist(st)
        await neg.bargain(states)
    return neg, create


class NegotiationLoopTests(unittest.IsolatedAsyncioTestCase):
    def sellers(self):
        a = FakeSellerSession("A", [laptop("P001", "Book 14", 1000, "MPN-1"),
                                    laptop("P002", "Phone X", 500, "MPN-9", category="phones")])
        b = FakeSellerSession("B", [laptop("SG-10001", "Book 14", 980, "MPN-2", ram=32), laptop("SG-10002", "Tab", 2000, "MPN-3")],
                              floor_fraction=0.908)
        states = [{"identifier": URN_A, "name": "A", "session": a}, {"identifier": URN_B, "name": "B", "session": b}]
        return a, b, states

    async def test_rules_are_enforced_and_prices_come_from_sellers(self):
        a, b, states = self.sellers()
        script = [
            [tool_use("make_offer", seller=1, price=900, message="early")],                  # refused: no candidate
            [tool_use("choose_candidate", seller=1, sku="SG-10001", reason="wrong seller")],  # refused: not A's SKU
            [tool_use("choose_candidate", seller=1, sku="P002", reason="a phone")],           # refused: not shortlisted
            [tool_use("choose_candidate", seller=1, sku="P001", reason="fits")],
            [tool_use("choose_candidate", seller=2, sku="SG-10001", reason="more RAM")],
            [tool_use("make_offer", seller=1, price=1200, message="above the ask")],          # refused: not below price
            *[[tool_use("make_offer", seller=1, price=500, message="lowball")]] * 4,          # 4th refused: 3 max
            [tool_use("choose_candidate", seller=1, sku="P001", reason="again")],             # refused: offers made
            [tool_use("make_offer", seller=2, price=850, message="try")],
            [tool_use("finish", winner=2, reason="")],                                         # refused: no reason
            [tool_use("finish", winner=2, reason="B's twice-the-RAM model costs $10 less. I claim it is $1.")],
        ]
        neg, _ = await negotiate(states, script)
        self.assertEqual((a.started, b.started), (["P001"], ["SG-10001"]))
        self.assertEqual(a.offer_prices, [500, 500, 500])
        self.assertEqual(b.offer_prices, [850])
        out = negotiator_agent.build_outcome(REQUEST, states, neg.chat, neg.pick)
        self.assertEqual({o["seller"]: o["final_price"] for o in out["offers"]}, {"A": 900, "B": 889.84})
        self.assertEqual((out["winner"], out["decided_by"]), ("B", "negotiator"))
        self.assertEqual(out["offers"][1]["product"]["specs"], {"ram_gb": 32})  # specs come from the seller
        self.assertIn("I claim it is $1", out["winner_reason"])  # Claude's text is the reason, never a price
        self.assertEqual(out["best_price"], 889.84)
        # Only offers that reached a seller are in the chat, with the seller's own price.
        self.assertEqual([(t["seller"], t["sku"], t["round"], t["buyer_offer"]) for t in neg.chat],
                         [("A", "P001", 1, 500), ("A", "P001", 2, 500), ("A", "P001", 3, 500), ("B", "SG-10001", 1, 850)])
        self.assertEqual(neg.chat[-1]["seller_price"], 889.84)
        self.assertIn('- Negotiator -> B (offer 1): $850.00 "try"', negotiator_agent.format_outcome(out))

    async def test_shortlist_respects_category_and_budget_headroom(self):
        _, _, states = self.sellers()
        neg, create = await negotiate(states, [[tool_use("finish", winner=0, reason="nothing")]])
        self.assertEqual([p["sku"] for p in states[0]["shortlist"]], ["P001"])
        self.assertEqual([p["sku"] for p in states[1]["shortlist"]], ["SG-10001"])  # $2000 is above 950 x 1.25
        opening = create.call_args_list[0].kwargs["messages"][0]["content"]
        self.assertIn("  - SG-10001 | Book 14 | Acme | MPN-2 | $980.00 | rating 4.5 | ram_gb 32", opening)
        self.assertIn("2. B: no candidate yet", opening)

    async def test_winner_over_budget_is_refused(self):
        a, b, states = self.sellers()
        script = [[tool_use("choose_candidate", seller=1, sku="P001", reason="fits")],
                  [tool_use("finish", winner=1, reason="nicer")],
                  [tool_use("finish", winner=0, reason="Nothing within budget.")]]
        neg, create = await negotiate(states, script)
        # The list is shared and ends with Claude's last answer; the message before it is what Claude saw.
        self.assertIn("above the budget", create.call_args.kwargs["messages"][-2]["content"][0]["content"])
        self.assertEqual(neg.pick, (0, "Nothing within budget."))

    async def test_failing_seller_stops_taking_offers(self):
        a, b, states = self.sellers()
        original = a.call_tool

        async def broken(tool, arguments):
            if tool == "make_offer":
                return SimpleNamespace(content=[SimpleNamespace(text="Error executing tool make_offer: boom")])
            return await original(tool, arguments)

        a.call_tool = broken
        script = [[tool_use("choose_candidate", seller=1, sku="P001", reason="fits")]] + \
                 [[tool_use("make_offer", seller=1, price=800, message="hi")]] * 3 + \
                 [[tool_use("finish", winner=0, reason="done")]]
        _, create = await negotiate(states, script)
        self.assertTrue(states[0]["stalled"])
        last_prompt = create.call_args.kwargs["messages"][-2]["content"][-1]["text"]
        self.assertIn("1. A: candidate P001 Book 14 (MPN-1), list $1000.00, current price $1000.00, final, 0 offer(s) left",
                      last_prompt)


class MarketplaceTests(unittest.IsolatedAsyncioTestCase):
    def test_is_seller_card(self):
        self.assertTrue(is_seller_card({"tools": [{"name": n} for n in product_server.NEGOTIATION_TOOLS]}))
        self.assertFalse(is_seller_card({"tools": [{"name": "search_products"}]}))

    def make_agent(self):
        agent = MarketplaceShoppingAgent(mock.Mock(), mock.Mock(), "http://localhost:8083/", api_key="test")
        agent.base_tools = [{"name": n} for n in ("search_products", "add_to_cart", "checkout", "remove_from_cart",
                                                   "delegate_negotiation")]
        agent.mcp_tool_names = {"search_products"}
        return agent

    async def test_quote_only_blocks_the_cart(self):
        agent = self.make_agent()
        agent.quote_only = True
        self.assertIn("quote-only", await agent.handle("checkout"))
        self.assertIn("quote-only", await agent.handle("add P001"))
        output, is_error = await agent.run_tool("add_to_cart", {"product_id": "P001"})
        self.assertTrue(is_error)
        self.assertEqual(agent.cart, {})
        with mock.patch.object(agent, "run_claude", mock.AsyncMock(return_value="ok")):
            await agent.handle("best laptop under 1500")
        self.assertEqual({t["name"] for t in agent.claude_tools}, {"search_products", "delegate_negotiation"})

    async def test_delegation_requires_discovered_agent_and_sellers(self):
        agent = self.make_agent()
        args = {"agent": "urn:air:evil", "request": "best laptop"}
        self.assertIn("Unknown agent", json.loads((await agent.run_tool("delegate_negotiation", args))[0])["error"])
        agent.found_agents["urn:air:neg"] = DiscoveredAgent(
            "urn:air:neg", "Negotiator", "http://localhost:8087/.well-known/agent-card.json", "http://localhost:8087/",
            "1.0", [], {})
        args["agent"] = "urn:air:neg"
        self.assertIn("find_sellers", json.loads((await agent.run_tool("delegate_negotiation", args))[0])["error"])


class FlowRecorderTests(unittest.TestCase):
    def test_trace_lanes_and_negotiation(self):
        queue = asyncio.Queue()
        rec = flow_ui.FlowRecorder(queue)
        rec.lane_ids[base_of("http://localhost:8083/")] = "agent"
        trace = []
        token = current_trace.set(trace)
        try:
            # The negotiator's card is read before its lane entry, as during find_agents.
            trace.append({"kind": "card", "src": "agent", "dst": "http://localhost:8087/", "tool": "GET /card",
                          "arguments": {}, "ok": True, "ms": 1.0, "summary": "Price Negotiator Agent"})
            record_lane("http://localhost:8081/sse", "seller", "Harbor Electronics")
            record_lane("http://localhost:8088/sse", "seller", "Summit Gadgets")
            record_lane("http://localhost:8087/", "negotiator", "Price Negotiator Agent")
            trace.append({"kind": "a2a", "src": "agent", "dst": "negotiator", "phase": "request", "label": "negotiate"})
            trace.append({"kind": "mcp", "src": "negotiator", "dst": "http://localhost:8088/", "tool": "make_offer",
                          "arguments": {"offer_price": 850.0, "message": "deal?"}, "ok": True, "ms": 3.0,
                          "summary": "counter"})
        finally:
            current_trace.reset(token)
        outcome = {"winner": "Summit Gadgets", "offers": []}
        rec.emit_trace({"result": {"message": {"metadata": {
            "trace": {"agent": "Claude", "product_server": "direct URL", "mcp_calls": trace},
            "negotiations": [outcome]}}}})
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        lanes = {e["lane"]: e["label"] for e in events if e["kind"] == "lane"}
        self.assertEqual(sorted(lanes.values()), ["Harbor Electronics", "Price Negotiator Agent", "Summit Gadgets"])
        card = next(e for e in events if e["kind"] == "card")
        self.assertEqual(lanes[card["to"]], "Price Negotiator Agent")
        offer = next(e for e in events if e["kind"] == "mcp" and e["phase"] == "request")
        self.assertEqual((lanes[offer["from"]], lanes[offer["to"]]), ("Price Negotiator Agent", "Summit Gadgets"))
        self.assertIn("make_offer $850", offer["label"])
        self.assertEqual(next(e for e in events if e["kind"] == "deal")["detail"], outcome)
        self.assertFalse(any(e["kind"] == "startup" for e in events))

    def test_live_entries_upgrade_lanes_and_are_not_drawn_twice(self):
        queue = asyncio.Queue()
        rec = flow_ui.FlowRecorder(queue)
        card = {"id": "c1", "kind": "card", "src": "agent", "dst": "http://localhost:8087/", "tool": "GET /card",
                "arguments": {}, "ok": True, "ms": 1.0, "summary": "Price Negotiator Agent"}
        lane = {"id": "l1", "kind": "lane", "key": "http://localhost:8087/", "role": "negotiator",
                "name": "Price Negotiator Agent"}
        offer = {"id": "o1", "kind": "mcp", "src": "negotiator", "dst": "http://localhost:8088/", "tool": "make_offer",
                 "arguments": {"offer_price": 800}, "ok": True, "ms": 2.0, "summary": "counter"}
        for entry in (card, lane, offer):  # one POST each, as the agents send them
            rec.emit_entries([entry], live=True)
        rec.emit_trace({"result": {"message": {"metadata": {"trace": {"mcp_calls": [card, lane, offer]}}}}})
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        lane_events = [e for e in events if e["kind"] == "lane"]
        self.assertEqual([(e["lane"], e["role"]) for e in lane_events[:2]], [("svc1", "other"), ("svc1", "negotiator")])
        arrows = [e for e in events if e["kind"] in ("card", "mcp")]
        self.assertEqual(len(arrows), 4)  # request + response for the card and the offer, once each
        self.assertTrue(all(e["live"] for e in arrows))
        offer_request = next(e for e in arrows if e["kind"] == "mcp" and e["phase"] == "request")
        self.assertEqual(offer_request["from"], "svc1")


if __name__ == "__main__":
    unittest.main()
