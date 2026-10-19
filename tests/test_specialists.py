"""Unit tests for the specialist advisor agents and the planner's per-request routing. No network, no Claude.

    cd shopping_agent && ../.venv/bin/python -m unittest discover -s tests -v
"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flow_ui  # noqa: E402
import gift_planner_agent as planner  # noqa: E402
import nanda_register  # noqa: E402
import product_server  # noqa: E402
import specialist_agent  # noqa: E402
from google.protobuf.json_format import MessageToDict  # noqa: E402
from nanda_discovery import AgentCandidate, DiscoveredAgent, DiscoveryError  # noqa: E402
from specialist_agent import PROFILES, Advisor  # noqa: E402

PRODUCTS = {p["id"]: p for p in product_server.PRODUCTS}


class CatalogSession:
    """An MCP session that runs the product server's tools in-process and records every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(self, name, arguments=None):
        self.calls.append(name)
        data = getattr(product_server, name)(**(arguments or {}))
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(data))], isError=False)

    async def list_tools(self):
        names = ["list_categories", "search_products", "get_product", "compare_products", "recommend_products",
                 "check_availability", "start_negotiation", "make_offer", "get_negotiation"]
        return SimpleNamespace(tools=[SimpleNamespace(name=n, description="", inputSchema={"type": "object"})
                                      for n in names])


def advise(profile: str, text: str, use_claude: bool = False) -> tuple[str, dict, CatalogSession]:
    session = CatalogSession()
    reply, advice = asyncio.run(Advisor(PROFILES[profile], session, use_claude).answer(text))
    return reply, advice, session


def card_of(profile: str) -> dict:
    return MessageToDict(specialist_agent.build_agent_card(PROFILES[profile], "http://localhost:9999"))


def agent(name: str, card: dict, url: str = "http://localhost:9999/") -> DiscoveredAgent:
    skills = [s.get("name", "") for s in card.get("skills", [])]
    return DiscoveredAgent(f"urn:air:{name}", name, url + ".well-known/agent-card.json", url, "1.0", skills, card)


SHOPPING_CARD = MessageToDict(__import__("a2a_server").build_agent_card())
NEGOTIATOR_CARD = MessageToDict(__import__("negotiator_agent").build_agent_card())


class RuleAdviceTests(unittest.TestCase):
    def test_laptop_advice_stays_in_laptops_and_budget(self):
        reply, advice, session = advise("laptop", "a laptop for video editing under 1200")
        self.assertTrue(advice["picks"])
        for pick in advice["picks"]:
            self.assertEqual(PRODUCTS[pick["id"]]["category"], "laptops")
            self.assertLessEqual(pick["price"], 1200)
            self.assertIn(pick["id"], reply)
        self.assertEqual(advice["budget"], 1200)
        self.assertTrue(advice["read_only"])
        self.assertTrue(set(session.calls) <= specialist_agent.READ_ONLY_TOOLS, session.calls)

    def test_travel_advice_uses_the_category_named_in_the_request(self):
        _, advice, _ = advise("travel", "headphones for a long flight under 200")
        self.assertTrue(advice["picks"])
        self.assertTrue(all(PRODUCTS[p["id"]]["category"] == "headphones" for p in advice["picks"]))

    def test_travel_advice_never_leaves_its_categories(self):
        _, advice, _ = advise("travel", "something for my trip under 80")
        self.assertTrue(all(PRODUCTS[p["id"]]["category"] in PROFILES["travel"].categories for p in advice["picks"]))

    def test_gift_advice_can_use_any_category(self):
        _, advice, _ = advise("gift", "a birthday gift for my mom under 50")
        self.assertTrue(advice["picks"])
        self.assertTrue(all(p["price"] <= 50 for p in advice["picks"]))

    def test_nothing_fits_says_so(self):
        reply, advice, _ = advise("laptop", "a laptop under 5")
        self.assertEqual(advice["picks"], [])
        self.assertIn("nothing in stock", reply)

    def test_cart_and_negotiation_requests_are_refused_without_tool_calls(self):
        for text in ("add P001 to my cart", "please check out", "negotiate the price of P001"):
            reply, advice, session = advise("laptop", text)
            self.assertIn("read-only", reply)
            self.assertIn("refused", advice)
            self.assertEqual(session.calls, [])


def response(*blocks, stop="tool_use"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def tool_use(name, **arguments):
    return SimpleNamespace(type="tool_use", id=f"t-{name}", name=name, input=arguments,
                           model_dump=lambda **_: {"type": "tool_use", "id": f"t-{name}", "name": name, "input": arguments})


def text(value):
    return SimpleNamespace(type="text", text=value, model_dump=lambda **_: {"type": "text", "text": value})


class ClaudeAdviceTests(unittest.TestCase):
    def run_script(self, script):
        client = mock.MagicMock()
        client.messages.create = mock.AsyncMock(side_effect=script)
        with mock.patch.object(specialist_agent.anthropic, "AsyncAnthropic", return_value=client):
            reply, advice, session = advise("laptop", "a laptop under 900", use_claude=True)
        return reply, advice, session, client.messages.create.call_args_list

    def test_only_read_only_tools_are_offered(self):
        _, _, _, calls = self.run_script([response(text("No laptops."), stop="end_turn")])
        offered = {t["name"] for t in calls[0].kwargs["tools"]}
        self.assertEqual(offered, specialist_agent.READ_ONLY_TOOLS)

    def test_tools_outside_the_allowlist_are_refused(self):
        _, _, session, calls = self.run_script([
            response(tool_use("make_offer", negotiation_id="x", offer_price=1)),
            response(text("Done."), stop="end_turn"),
        ])
        self.assertNotIn("make_offer", session.calls)
        result = calls[1].kwargs["messages"][2]["content"][0]  # the list grows after the call: index it
        self.assertTrue(result["is_error"])

    def test_picks_are_only_products_the_catalog_returned(self):
        laptop = product_server.recommend_products(900, "laptops", limit=3)["recommendations"][0]
        reply, advice, _, _ = self.run_script([
            response(tool_use("recommend_products", budget=900, category="laptops", limit=3)),
            response(text(f"Get {laptop['id']}, or P9999 which I made up."), stop="end_turn"),
        ])
        ids = [p["id"] for p in advice["picks"]]
        self.assertIn(laptop["id"], ids)
        self.assertNotIn("P9999", ids)
        self.assertEqual(advice["mode"], "Claude")


class CapabilityTests(unittest.TestCase):
    def test_specialists_can_advise_but_not_buy_or_negotiate(self):
        for profile in PROFILES:
            specialist = agent(profile, card_of(profile))
            self.assertTrue(planner.can_do(specialist, "advice"), profile)
            self.assertFalse(planner.can_do(specialist, "buy"), profile)
            self.assertFalse(planner.can_do(specialist, "negotiate"), profile)

    def test_shopping_agent_can_do_every_job_and_negotiator_none(self):
        shopping, negotiator = agent("shop", SHOPPING_CARD), agent("neg", NEGOTIATOR_CARD)
        for job in planner.JOB_SKILL_TAGS:
            self.assertTrue(planner.can_do(shopping, job), job)
            self.assertFalse(planner.can_do(negotiator, job), job)

    def test_checkout_job_skips_a_better_scored_specialist(self):
        candidates = [AgentCandidate(agent("Laptop Advisor Agent", card_of("laptop")), 0.9, "", []),
                      AgentCandidate(agent("Personal Shopping Agent", SHOPPING_CARD), 0.4, "", [])]
        chosen, _, skipped = asyncio.run(planner.pick_capable(candidates, "buy", "task", ["laptop"], False, "m"))
        self.assertEqual(chosen.name, "Personal Shopping Agent")
        self.assertEqual([c.agent.name for c in skipped], ["Laptop Advisor Agent"])

    def test_no_capable_candidate_names_the_skipped_ones(self):
        candidates = [AgentCandidate(agent("Gift Ideas Agent", card_of("gift")), 0.9, "", [])]
        with self.assertRaisesRegex(DiscoveryError, "Gift Ideas Agent.*none can build a cart"):
            asyncio.run(planner.pick_capable(candidates, "buy", "task", ["x"], False, "m"))

    def test_named_specialist_is_refused_for_checkout(self):
        with self.assertRaisesRegex(DiscoveryError, "can't build a cart"):
            planner.require_skill(agent("Travel Gear Agent", card_of("travel")), "buy")

    def test_advice_task_fits_the_index_limit(self):
        task = planner.advice_task(planner.ADVICE_TASK, "x" * 300)
        self.assertEqual(len(task), planner.MAX_TASK_CHARS)
        self.assertTrue(task.startswith(planner.ADVICE_TASK + ": x"))


class RegistrationTests(unittest.TestCase):
    def test_specialist_listing_comes_from_the_card(self):
        card = card_of("travel")
        description, tags = nanda_register.card_listing(card, nanda_register.KINDS["specialist"])
        self.assertEqual(description, PROFILES["travel"].description)
        self.assertIn("travel", tags)
        self.assertIn("specialist", tags)
        self.assertEqual(len(tags), len(set(tags)))
        self.assertLessEqual(len(tags), 20)

    def test_other_kinds_keep_their_own_listing(self):
        spec = nanda_register.KINDS["seller"]
        self.assertEqual(nanda_register.card_listing({"description": "x"}, spec), (spec["description"], spec["tags"]))


class FlowRecorderAdviceTests(unittest.TestCase):
    def test_agent_traces_and_requests_go_to_the_chosen_agents_lane(self):
        rec = flow_ui.FlowRecorder(asyncio.Queue())
        lane = rec.add_lane("http://localhost:8085/", "a2a", "Laptop Advisor Agent")
        rec.agent_lane = lane
        self.assertEqual(rec.lane_for("agent"), lane)
        request = SimpleNamespace(url="http://localhost:8085/", method="POST")
        self.assertEqual(rec.classify(request), (lane, "a2a"))
        card = SimpleNamespace(url="http://localhost:8085/.well-known/agent-card.json", method="GET")
        self.assertEqual(rec.classify(card), (lane, "card"))
        rec.agent_lane = None
        self.assertEqual(rec.lane_for("agent"), "agent")

    def test_each_agents_startup_lookup_is_drawn_once(self):
        rec = flow_ui.FlowRecorder(asyncio.Queue())
        startup = lambda src, i: {"id": i, "kind": "startup", "src": src, "dst": "index", "phase": "request",
                                  "label": "startup"}
        rec.emit_entries([startup("http://localhost:8085/", "a"), startup("http://localhost:8085/", "b"),
                          startup("http://localhost:8089/", "c")])
        drawn = []
        while not rec.queue.empty():
            event = rec.queue.get_nowait()
            if event["kind"] == "startup":
                drawn.append(event)
        self.assertEqual(len(drawn), 2)


if __name__ == "__main__":
    unittest.main()
