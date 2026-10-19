"""A gift-planner agent that delegates shopping to the shopping agent over A2A.

It has no catalog access of its own, and by default it doesn't know the shopping agent's name
either: it describes its job to the NANDA Index (agentic search), reads the A2A agent card of
each candidate, picks one, and sends it tasks; that agent in turn uses the product server's MCP tools:

    gift planner --A2A--> shopping agent (a2a_server.py) --MCP--> product server

    export NANDA_INDEX_URL=http://localhost:3001
    python gift_planner_agent.py "headphones under 150" "books under 30"          # discover by task
    python gift_planner_agent.py --task "buy birthday presents" "watch under 200"  # custom job description
    python gift_planner_agent.py --shopping-agent "urn:air:host39.org:personal:shopping-agent@nanda.local" "books under 30"
    python gift_planner_agent.py --search "personal shopping agent" "watch under 200" --checkout

With ANTHROPIC_API_KEY set, Claude chooses among the candidates and plans what to ask the
chosen agent; otherwise the best-scored candidate is used and a fixed plan runs:
recommend -> add the top pick -> show the cart.

Advice mode routes each request to a specialist: the planner describes that request to the index
("expert specialist advice: a laptop for video editing under 1200"), so the laptop, gift or travel
specialist (specialist_agent.py) that fits it ranks first. Specialists are read-only:

    python gift_planner_agent.py --advice "a laptop for video editing under 1200" "a birthday gift for my mom under 50"

Every job checks the chosen agent's card for the skill it needs (a cart for buying, best-price
quotes for --negotiate, recommendations for --advice), so a read-only specialist is never asked to
check out.
"""
import argparse
import asyncio
import os
import re
from typing import Optional

import anthropic
import httpx
from a2a.client import A2AClientError

from a2a_link import A2ALink
from claude_agent import DEFAULT_MODEL
from nanda_discovery import (
    A2A_CARD_TYPE, NANDA_INDEX_URL, AgentCandidate, DiscoveredAgent, DiscoveryError, agentic_search_entries,
    discover_agent, read_agent_card, read_candidates,
)
from negotiation_report import attach_report
from shopping_agent import PRODUCT_ID_RE

# The planner's own job, which is all it tells the index when discovering by task.
PLANNER_TASK = "help me buy gifts within a budget: recommend products, add them to a cart and check out"
NEGOTIATE_TASK = "shopping agent: find the best product for me and get the best price for it from several sellers"
# Advice mode appends each request to this, so the index ranks the specialist for that request first.
ADVICE_TASK = "expert specialist advice"
MAX_TASK_CHARS = 128  # the index's limit for an agentic-search query
# The skill tag an agent card needs for each job. Specialists advise but have no cart, and the
# negotiator only takes JSON, so neither is handed a job it can't do.
JOB_SKILL_TAGS = {"buy": "cart", "negotiate": "price-comparison", "advice": "recommendations"}
JOB_NAMES = {"buy": "build a cart and check out", "negotiate": "get negotiated quotes from several sellers",
             "advice": "recommend products"}
# Sent with every message in negotiation mode; the shopping agent then disables its cart tools.
QUOTE_ONLY = {"quote_only": True}
NEGOTIATION_TIMEOUT = 900
MAX_TOOL_ROUNDS = 12
REQUEST_SPLIT_RE = re.compile(r"\s*(?:[,;]|\band also\b|\balso\b|\bplus\b|\band\b)\s*", re.IGNORECASE)
PRICE_HINT_RE = re.compile(r"\$\s*\d|\d\s*\$|\b(?:under|below|less than|up to|within|max)\s*\$?\s*\d", re.IGNORECASE)
REQUEST_VERB_RE = re.compile(r"^(?:i\s+(?:want|need|would like)|show|find|get|buy|recommend|suggest)\b", re.IGNORECASE)

ASK_TOOL = {
    "name": "ask_shopping_agent",
    "description": "Send one message to the Personal Shopping Agent over A2A and get its reply. "
                   "It remembers this conversation, including the cart.",
    "input_schema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}

CHOOSE_TOOL = {
    "name": "choose_agent",
    "description": "Pick the candidate agent to delegate the job to. Use index 0 if none can do it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "minimum": 0, "description": "1-based candidate number, or 0 for none"},
            "reason": {"type": "string", "description": "One sentence on why"},
        },
        "required": ["index", "reason"],
    },
}

CHOOSER_PROMPT = """\
You are a gift-planning agent with no shopping ability of your own. The NANDA Index returned these
A2A agents for your job description. Using their descriptions and the skills on their agent cards,
pick the one that can best do your job, or none if no candidate fits.
Always answer by calling the choose_agent tool exactly once; don't reply with plain text."""

PLANNER_PROMPT = """\
You are a gift-planning agent. You have no product catalog yourself: delegate all shopping to the
Personal Shopping Agent with the ask_shopping_agent tool. It was discovered through the NANDA Index;
its agent card lists these skills (with example requests it understands):
{skills}

For each gift request, ask the shopping agent for recommendations, pick one in-stock product
that fits the budget, and add it to the cart by product ID. Then show the cart.
Only add a product of the kind that was asked for (a tablet is not a laptop). If nothing of that
kind fits the budget, add nothing for that request and say so, with the cheapest option found.
{checkout}Finish with a short plain-text summary: which product (ID, name, price) was chosen for each request
and the cart total."""


# One A2A conversation with the shopping agent (a single contextId keeps the cart).
ShoppingAgentLink = A2ALink


def split_requests(gifts: list[str]) -> list[str]:
    """Split lines like 'a laptop under 200 and a book under 50' into one request per product.

    A line is split only when every part looks like a request of its own (has a price limit or a
    request verb), so 'black and white headphones' stays whole.
    """
    out = []
    for gift in gifts:
        parts = [p.strip(" .") for p in REQUEST_SPLIT_RE.split(gift) if p.strip(" .")]
        if len(parts) > 1 and all(PRICE_HINT_RE.search(p) or REQUEST_VERB_RE.search(p) for p in parts):
            out += parts
        else:
            out.append(gift.strip())
    return out


async def fixed_plan(link: ShoppingAgentLink, gifts: list[str], checkout: bool) -> str:
    picks = []
    for gift in split_requests(gifts):
        reply = await link.ask(f"recommend {gift}")
        ids = PRODUCT_ID_RE.findall(reply)
        if not ids:
            picks.append(f"- {gift}: nothing added. {reply}")
            continue
        added = await link.ask(f"add {ids[0].upper()}")
        picks.append(f"- {gift}: {added}")
    cart = await link.ask("checkout" if checkout else "cart")
    return "Gift plan:\n" + "\n".join(picks) + "\n\n" + cart


async def claude_plan(link: ShoppingAgentLink, gifts: list[str], checkout: bool, model: str) -> str:
    skills = "\n".join(
        f"- {s.get('name')}: {s.get('description')} Examples: {', '.join(s.get('examples', []))}"
        for s in link.agent.card.get("skills", [])
    )
    system = PLANNER_PROMPT.format(
        skills=skills,
        checkout="When the cart is complete, ask the shopping agent to check out.\n" if checkout
        else "Do not check out.\n",
    )
    requests = split_requests(gifts)
    return await tool_loop(link, system, "Gift requests:\n" + "\n".join(f"{i}. {g}" for i, g in enumerate(requests, 1)), model)


NEGOTIATE_PROMPT = """\
You are a gift-planning agent with no catalog of your own. Delegate with the ask_shopping_agent tool to the
agent below (discovered through the NANDA Index); its agent card lists these skills:
{skills}

Ask it to get the best deal for the request from several sellers: a negotiator agent picks a product at
each seller, negotiates every price and recommends the best value. This is quote only: never ask it to add
anything to the cart or to check out. Ask follow-up questions only if its answer is missing a seller's
negotiated price or the winner.
Finish with a short plain-text summary: the winning seller and product and why, and each seller's product
and negotiated price. Use only numbers and specs the shopping agent reported."""


def negotiation_message(request: str) -> str:
    return (f"{request}. Get me the best deal from several sellers: have a negotiator agent pick the best "
            "matching product at each seller, negotiate every price, and recommend the best value for specs and "
            "price. Quote only: don't add anything to the cart.")


async def negotiate_plan(link: ShoppingAgentLink, request: str, use_claude: bool, model: str) -> str:
    """Quote-only flow: the shopping agent picks the model and delegates the bargaining.

    The negotiation chat is attached from the reply metadata, since a Claude summary would shorten it."""
    if not use_claude:
        summary = await link.ask(negotiation_message(request))
    else:
        skills = "\n".join(
            f"- {s.get('name')}: {s.get('description')} Examples: {', '.join(s.get('examples', []))}"
            for s in link.agent.card.get("skills", [])
        )
        summary = await tool_loop(link, NEGOTIATE_PROMPT.format(skills=skills), f"Request: {request}", model)
    return attach_report(summary, link.negotiations[-1:])


async def tool_loop(link: ShoppingAgentLink, system: str, content: str, model: str) -> str:
    client = anthropic.AsyncAnthropic()
    messages = [{"role": "user", "content": content}]
    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.messages.create(
            model=model, max_tokens=2000, system=system, tools=[ASK_TOOL], messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text").strip()
        results = []
        for block in response.content:
            if block.type == "tool_use":
                reply = await link.ask(str(block.input.get("message", "")))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": reply})
        messages.append({"role": "user", "content": results})
    return "Stopped after too many steps; see the exchanges above."


def describe_candidate(number: int, candidate: AgentCandidate) -> str:
    lines = [f"{number}. {candidate.agent.name} (index score {candidate.score:.2f})",
             f"   Description: {candidate.description or '(none)'}",
             f"   Tags: {', '.join(candidate.tags) or '(none)'}"]
    for skill in candidate.agent.card.get("skills", []):
        examples = "; ".join(skill.get("examples", []))
        lines.append(f"   Skill {skill.get('name') or skill.get('id')}: {skill.get('description', '')}"
                     + (f" Examples: {examples}" if examples else ""))
    return "\n".join(lines)


async def choose_agent(candidates: list[AgentCandidate], task: str, gifts: list[str], use_claude: bool,
                       model: str) -> tuple[DiscoveredAgent, str]:
    """Return the agent to delegate to and why. Claude decides when available."""
    if not candidates:
        raise DiscoveryError(f"The NANDA Index found no usable A2A agent for the task '{task}'")
    if not use_claude:
        best = candidates[0]
        return best.agent, f"highest index score ({best.score:.2f}); no LLM available to compare candidates"
    listing = "\n".join(describe_candidate(i, c) for i, c in enumerate(candidates, 1))
    prompt = (f"My job: {task}\nRequests:\n" + "\n".join(f"- {g}" for g in gifts)
              + f"\n\nCandidates from the NANDA Index:\n{listing}")
    # No forced tool_choice: some models reject {"type": "tool"} / {"type": "any"}, so the prompt asks for the call.
    response = await anthropic.AsyncAnthropic().messages.create(
        model=model, max_tokens=2000, system=CHOOSER_PROMPT, tools=[CHOOSE_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )
    choice = next((b.input for b in response.content if b.type == "tool_use"), None)
    if choice is None:
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        raise DiscoveryError(f"Claude didn't call choose_agent; it said: {text[:300] or '(nothing)'}")
    number, reason = int(choice.get("index", 0) or 0), str(choice.get("reason", "")).strip()
    if not 1 <= number <= len(candidates):
        raise DiscoveryError(f"Claude found no suitable agent among {len(candidates)} candidate(s): {reason}")
    return candidates[number - 1].agent, f"Claude chose candidate {number}: {reason}"


def skill_tags(agent: DiscoveredAgent) -> set[str]:
    return {str(t).lower() for s in agent.card.get("skills", []) for t in s.get("tags", [])}


def can_do(agent: DiscoveredAgent, job: str) -> bool:
    return JOB_SKILL_TAGS[job] in skill_tags(agent)


def require_skill(agent: DiscoveredAgent, job: str) -> None:
    if not can_do(agent, job):
        raise DiscoveryError(f"{agent.name} can't {JOB_NAMES[job]}: its agent card has no skill tagged "
                             f"'{JOB_SKILL_TAGS[job]}'")


def split_capable(candidates: list[AgentCandidate], job: str) -> tuple[list[AgentCandidate], list[AgentCandidate]]:
    """(candidates whose card has the job's skill, the others), each in index order."""
    return [c for c in candidates if can_do(c.agent, job)], [c for c in candidates if not can_do(c.agent, job)]


async def pick_capable(candidates: list[AgentCandidate], job: str, task: str, gifts: list[str], use_claude: bool,
                       model: str) -> tuple[DiscoveredAgent, str, list[AgentCandidate]]:
    """Choose among the candidates that can do the job; returns (agent, reason, the skipped candidates)."""
    capable, skipped = split_capable(candidates, job)
    if candidates and not capable:
        names = ", ".join(c.agent.name for c in skipped)
        raise DiscoveryError(f"The NANDA Index found {len(skipped)} agent(s) ({names}), but none can {JOB_NAMES[job]}")
    agent, reason = await choose_agent(capable, task, gifts, use_claude, model)
    return agent, reason, skipped


def advice_task(prefix: str, request: str) -> str:
    return f"{prefix.strip()}: {request.strip()}"[:MAX_TASK_CHARS]


def job_of(args: argparse.Namespace) -> str:
    return "negotiate" if args.negotiate else "advice" if args.advice else "buy"


def print_candidates(candidates: list[AgentCandidate], skipped: list[AgentCandidate], job: str) -> None:
    for i, c in enumerate(candidates, 1):
        print(f"[discovery] candidate {i}: {c.agent.name} ({c.agent.identifier}) score {c.score:.2f}"
              + (f" - skipped, can't {JOB_NAMES[job]}" if c in skipped else ""))


async def discover_for_task(http: httpx.AsyncClient, task: str, job: str, gifts: list[str], use_claude: bool,
                            model: str) -> tuple[DiscoveredAgent, str]:
    print(f"[discovery] asking the NANDA Index for agents that can: {task}")
    candidates = await read_candidates(http, await agentic_search_entries(http, task, A2A_CARD_TYPE, NANDA_INDEX_URL))
    agent, reason, skipped = await pick_capable(candidates, job, task, gifts, use_claude, model)
    print_candidates(candidates, skipped, job)
    return agent, f"NANDA agentic search, {reason}"


async def find_named_agent(http: httpx.AsyncClient, args: argparse.Namespace) -> Optional[tuple[DiscoveredAgent, str]]:
    """The agent given by --card-url, --shopping-agent or --search, if any."""
    if args.card_url:
        return await read_agent_card(http, "", "shopping agent", args.card_url), f"direct card URL {args.card_url}"
    if args.shopping_agent or args.search:
        return await discover_agent(args.shopping_agent, args.search), "NANDA Index"
    return None


async def find_agent(http: httpx.AsyncClient, args: argparse.Namespace, use_claude: bool) -> tuple[DiscoveredAgent, str]:
    job = job_of(args)
    named = await find_named_agent(http, args)
    if named is not None:
        require_skill(named[0], job)
        return named
    return await discover_for_task(http, args.task, job, args.gifts, use_claude, args.model)


def advice_section(number: int, request: str, agent_name: str, reply: str) -> str:
    return f"{number}. \u201c{request}\u201d, answered by {agent_name}:\n{reply.strip()}"


async def advice_plan(http: httpx.AsyncClient, args: argparse.Namespace, use_claude: bool) -> str:
    """Each request goes to the agent the index ranks best for it (or to the one named agent)."""
    named = await find_named_agent(http, args)
    if named is not None:
        require_skill(named[0], "advice")
    sections, links = [], []
    try:
        for number, request in enumerate(split_requests(args.gifts), 1):
            if named is not None:
                agent, source = named
            else:
                agent, source = await discover_for_task(http, advice_task(args.task, request), "advice", [request],
                                                        use_claude, args.model)
            print(f"[discovery] {source}: {agent.describe()}")
            link = A2ALink(agent, http, them=agent.name)
            links.append(link)
            await link.connect()
            sections.append(advice_section(number, request, agent.name, await link.ask(request)))
    finally:
        # Closing an A2A client closes the shared httpx client, so links are closed only at the end.
        for link in links:
            await link.close()
    return "Specialist advice (read-only: nothing was added to a cart or bought)\n\n" + "\n\n".join(sections)


async def run(args: argparse.Namespace) -> None:
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY")) and not args.no_claude
    async with httpx.AsyncClient(timeout=NEGOTIATION_TIMEOUT if args.negotiate else 120) as http:
        if args.advice:
            summary = await advice_plan(http, args, use_claude)
            print("\n=== Gift planner result ===")
            print(summary)
            return
        agent, source = await find_agent(http, args, use_claude)
        print(f"[discovery] {source}: {agent.describe()}")
        if agent.skills:
            print(f"[discovery] skills: {', '.join(agent.skills)}")
        print(f"[planner] {'Claude (' + args.model + ')' if use_claude else 'fixed plan'}")

        link = ShoppingAgentLink(agent, http, metadata=QUOTE_ONLY if args.negotiate else None)
        await link.connect()
        print(f"[a2a] conversation {link.context_id}{' (quote only)' if args.negotiate else ''}")
        try:
            if args.negotiate:
                summary = await negotiate_plan(link, args.negotiate, use_claude, args.model)
            elif use_claude:
                summary = await claude_plan(link, args.gifts, args.checkout, args.model)
            else:
                summary = await fixed_plan(link, args.gifts, args.checkout)
        finally:
            await link.close()
    print("\n=== Gift planner result ===")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gift-planner agent that delegates shopping over A2A")
    parser.add_argument("gifts", nargs="*", help="Gift requests, e.g. 'headphones under 150' (asked for if omitted)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--shopping-agent", default=os.environ.get("SHOPPING_AGENT_LOCATOR"),
                       help="NANDA identifier of the shopping agent (urn:air:...)")
    where.add_argument("--search", help="Find the shopping agent in the NANDA Index by keywords")
    where.add_argument("--card-url", help="Skip NANDA and use this A2A agent card URL directly")
    where.add_argument("--task", help="Job description sent to the index's agentic search (the default "
                                      f"discovery mode; default: '{PLANNER_TASK}'). With --advice, each request "
                                      f"is appended to it (default '{ADVICE_TASK}')")
    job = parser.add_mutually_exclusive_group()
    job.add_argument("--negotiate", metavar="REQUEST",
                     help="Quote only: e.g. 'the best laptop under 900'. A negotiator agent picks a product at "
                          "each seller, negotiates and recommends the best deal (needs Claude)")
    job.add_argument("--advice", action="store_true",
                     help="Read-only: send each request to the specialist the NANDA Index ranks best for it")
    parser.add_argument("--checkout", action="store_true", help="Place the (simulated) order at the end")
    parser.add_argument("--no-claude", action="store_true", help="Use the fixed plan even if ANTHROPIC_API_KEY is set")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model ID for planning")
    args = parser.parse_args()
    if not (args.shopping_agent or args.search or args.card_url or args.task):
        args.task = NEGOTIATE_TASK if args.negotiate else ADVICE_TASK if args.advice else PLANNER_TASK
    if args.advice and args.checkout:
        parser.error("--advice is read-only: don't combine it with --checkout")
    if args.negotiate:
        if args.gifts or args.checkout:
            parser.error("--negotiate is quote only: don't combine it with gift requests or --checkout")
        args.gifts = [args.negotiate]
    elif not args.gifts:
        print("Gift requests, one per line (e.g. 'headphones under 150'). Empty line to start:")
        while (line := input("gift> ").strip()):
            args.gifts.append(line)
        if not args.gifts:
            parser.error("no gift requests given")

    try:
        asyncio.run(run(args))
    except DiscoveryError as e:
        raise SystemExit(f"Finding an agent failed: {e}") from None
    except (httpx.ConnectError, A2AClientError) as e:
        raise SystemExit(f"Could not talk to the agent: {e}. Is it running (a2a_server.py, "
                         "specialist_agent.py)?") from None


if __name__ == "__main__":
    main()
