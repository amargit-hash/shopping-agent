"""Register the product server (MCP) or the shopping agent (A2A) in the NANDA Index as a Personal entry.

Product server (MCP server card):
    python nanda_register.py --card-url https://<your-app>/.well-known/mcp-server-card.json
Shopping agent (A2A agent card, served by a2a_server.py):
    python nanda_register.py --a2a --card-url https://<your-app>/.well-known/agent-card.json
A second seller (product_server.py with SELLER_ID etc.) or the negotiator (negotiator_agent.py):
    python nanda_register.py --kind seller --name "Seller B" --card-url http://localhost:8088/.well-known/mcp-server-card.json
    python nanda_register.py --kind negotiator --card-url http://localhost:8087/.well-known/agent-card.json
A specialist advisor (specialist_agent.py); its name, description and tags come from its card:
    python nanda_register.py --kind specialist --card-url http://localhost:8085/.well-known/agent-card.json

For scripts: --email skips the email prompt, NANDA_PASSWORD (environment) skips the password
prompt and --create-account creates the account without asking.

You'll be asked for your NANDA Index email and password (an account is created if you
don't have one). After registering, click the link in the verification email; the entry
goes live once the email is verified.

Check an entry later with:
    python nanda_register.py --status you@example.com          (add --a2a for an agent entry)

Note: Personal identifiers are built from your email, which is therefore public in the index.
Each email has one Personal identifier, so register the server and the agent under different emails.
"""
import argparse
import asyncio
import getpass
import os
import re
import sys
from urllib.parse import urlparse

import httpx

from nanda_discovery import (
    A2A_CARD_TYPE, MCP_CARD_TYPE, NANDA_INDEX_URL, DiscoveryError, read_agent_card, read_card,
    resolve, resolve_agent,
)

KINDS = {
    "mcp": {
        "media_type": MCP_CARD_TYPE,
        "name": "Product Catalog",
        "org_prefix": "product-catalog",
        "description": "Product catalog MCP server for a personal shopping agent: search, compare, recommend "
                       "and check stock across electronics, laptops, home, fitness, beauty and more.",
        "tags": ["shopping", "product-catalog", "ecommerce", "mcp", "recommendations"],
        "read_card": read_card,
        "resolve": resolve,
        "use": 'python claude_agent.py --nanda-locator "{urn}"',
    },
    "a2a": {
        "media_type": A2A_CARD_TYPE,
        "name": "Personal Shopping Agent",
        "org_prefix": "shopping-agent",
        "description": "A2A personal shopping agent: finds, compares and recommends products within a budget, "
                       "keeps a cart per conversation and places simulated orders.",
        "tags": ["shopping", "a2a", "agent", "recommendations", "cart"],
        "read_card": read_agent_card,
        "resolve": resolve_agent,
        "use": 'python gift_planner_agent.py --shopping-agent "{urn}"',
    },
    "seller": {
        "media_type": MCP_CARD_TYPE,
        "name": "Seller",
        "org_prefix": "seller",
        "description": "Product seller MCP server: search the catalog, check stock, give price quotes and "
                       "negotiate prices (start_negotiation, make_offer). Quotes only; the seller uses its own SKUs, "
                       "and the model number (mpn) identifies the same model at other sellers.",
        "tags": ["shopping", "seller", "product-catalog", "price-negotiation", "quotes", "mcp"],
        "read_card": read_card,
        "resolve": resolve,
        "use": "it is found by the shopping agent's seller search (find_sellers)",
    },
    "negotiator": {
        "media_type": A2A_CARD_TYPE,
        "name": "Price Negotiator Agent",
        "org_prefix": "negotiator",
        "description": "A2A price negotiator agent: finds the best deal for a shopping request across several "
                       "sellers' MCP servers on a buyer's behalf. Picks a product at each seller, negotiates every "
                       "price and recommends the best value for specs and price. Quote only.",
        "tags": ["negotiation", "price", "quotes", "a2a", "agent", "shopping"],
        "read_card": read_agent_card,
        "resolve": resolve_agent,
        "use": "it is found by the shopping agent's agent search (find_agents)",
    },
    # Specialists (specialist_agent.py) describe themselves: the index entry copies the card's text.
    "specialist": {
        "media_type": A2A_CARD_TYPE,
        "name": None,
        "org_prefix": "specialist",
        "description": None,
        "tags": ["a2a", "agent", "specialist", "advice"],
        "read_card": read_agent_card,
        "resolve": resolve_agent,
        "use": 'python gift_planner_agent.py --advice "a laptop for video editing under 1200"',
    },
}


def card_listing(card: dict, spec: dict) -> tuple[str, list[str]]:
    """Index description and tags for a card: the kind's own, or (specialists) the card's."""
    if spec["description"] is not None:
        return spec["description"], spec["tags"]
    skill_tags = [t for s in card.get("skills", []) for t in s.get("tags", [])]
    tags = list(dict.fromkeys(t.lower() for t in skill_tags + spec["tags"]))[:20]
    return str(card.get("description") or "")[:1000], tags


def personal_locator(email: str) -> str:
    return f"urn:air:host39.org:personal:{email.strip().lower()}"


def is_local(url: str) -> bool:
    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


def default_org_id(email: str, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", email.split("@")[0].lower()).strip("-")
    return f"{prefix}-{slug}"[:64].rstrip("-")


def fail(response: httpx.Response, action: str) -> None:
    try:
        detail = response.json()
    except ValueError:
        detail = response.text[:300]
    sys.exit(f"{action} failed (HTTP {response.status_code}): {detail}")


async def login_or_signup(client: httpx.AsyncClient, email: str, create: bool = False) -> str:
    password = os.environ.get("NANDA_PASSWORD") or getpass.getpass("NANDA Index password (input hidden, min 8 characters): ")
    response = await client.post(f"{NANDA_INDEX_URL}/auth/login", json={"email": email, "password": password})
    if response.status_code == 200:
        return response.json()["token"]
    if response.status_code not in (400, 401, 404):
        fail(response, "Login")
    if not create and input("No account with that email/password. Create a new account? [y/N] ").strip().lower() != "y":
        sys.exit("Stopped. If you signed up with Google or GitHub, set a password on nandaindex.org first.")
    response = await client.post(f"{NANDA_INDEX_URL}/auth/register", json={"email": email, "password": password})
    if response.status_code not in (200, 201):
        fail(response, "Sign-up")
    return response.json()["token"]


async def register(card_url: str, display_name: str | None, org_id: str | None, kind: str,
                   email: str | None = None, create: bool = False) -> None:
    spec = KINDS[kind]
    async with httpx.AsyncClient(timeout=30) as client:
        is_agent = spec["media_type"] == A2A_CARD_TYPE
        print(f"Checking {'agent' if is_agent else 'server'} card {card_url} ...")
        try:
            found = await spec["read_card"](client, "", display_name or spec["name"] or "agent", card_url)
        except DiscoveryError as e:
            sys.exit(f"The card isn't usable yet: {e}")
        print(f"  OK: {found.describe().split(' -> ', 1)[1]}")
        display_name = display_name or spec["name"] or found.card.get("name") or "Agent"
        description, tags = card_listing(found.card, spec)

        email = email or input("Email for the NANDA Index (becomes part of your public identifier): ").strip()
        token = await login_or_signup(client, email, create)
        body = {
            "org_id": org_id or default_org_id(email, spec["org_prefix"]),
            "display_name": display_name,
            "hosting_path": "personal",
            "contact_email": email,
            "registry_url": card_url,
            "media_type": spec["media_type"],
            "description": description,
            "tags": tags,
        }
        response = await client.post(
            f"{NANDA_INDEX_URL}/api/v1/orgs", json=body, headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code != 201:
            fail(response, "Registration")
        record = response.json()

    flag = "" if kind == "mcp" else f" --kind {kind}"
    print("\nRegistered (status: pending until you verify your email).")
    print(f"  org_id:     {record.get('org_id')}")
    print(f"  identifier: {record.get('identifier')}")
    print("\nNext steps:")
    if is_local(NANDA_INDEX_URL):
        print("  1. A local index prints the verification link in its log instead of emailing it:")
        print("     open the '[email] verification link' URL from nanda-index-v2/logs/index.log.")
    else:
        print(f"  1. Open the verification email sent to {email} and click the link.")
    print(f"  2. Check it's live:  python nanda_register.py{flag} --status {email}")
    print(f"  3. Use it:           {spec['use'].format(urn=record.get('identifier'))}")


async def status(email: str, kind: str) -> None:
    locator = personal_locator(email)
    try:
        found = await KINDS[kind]["resolve"](locator)
    except DiscoveryError as e:
        sys.exit(f"{locator}: not resolvable yet ({e}). If you just registered, verify your email first.")
    print(f"Live in the NANDA Index:\n  {found.describe()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the product server or shopping agent in the NANDA Index")
    parser.add_argument("--card-url", help="URL of the card (.../mcp-server-card.json, or .../agent-card.json with --a2a)")
    parser.add_argument("--a2a", action="store_true", help="Register an A2A agent card (same as --kind a2a)")
    parser.add_argument("--kind", choices=sorted(KINDS), help="What to register (default mcp): sets the index "
                        "description and tags")
    parser.add_argument("--email", help="NANDA Index account email (asked for if omitted)")
    parser.add_argument("--create-account", action="store_true", help="Create the account if it doesn't exist")
    parser.add_argument("--name", help="Display name in the index")
    parser.add_argument("--org-id", help="Index org_id (lowercase letters, digits, dashes); derived from your email by default")
    parser.add_argument("--status", metavar="EMAIL", help="Check whether the entry for EMAIL resolves")
    args = parser.parse_args()
    kind = args.kind or ("a2a" if args.a2a else "mcp")
    if args.status:
        asyncio.run(status(args.status, kind))
    elif args.card_url:
        if not args.card_url.startswith("https://") and not (
            is_local(NANDA_INDEX_URL) and is_local(args.card_url)
        ):
            parser.error("--card-url must be a public https:// URL "
                         "(http://localhost is only allowed with a local NANDA_INDEX_URL)")
        asyncio.run(register(args.card_url, args.name, args.org_id, kind, args.email, args.create_account))
    else:
        parser.error("give --card-url to register, or --status EMAIL to check an entry")


if __name__ == "__main__":
    main()
