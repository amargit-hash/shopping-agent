"""Find MCP servers and A2A agents through the NANDA Index and connect to them.

Resolution chain (NANDA Index README):
    1. NANDA Index    GET /api/v1/resolve?locator=<urn>,  GET /api/v1/search?q=<words>
                      or GET /api/v1/agentic-search?q=<task description>  (no name needed)
    2. Card           MCP server card: {"transport": {"type": "sse", "url": ".../sse"}, ...}
                      A2A agent card:  {"supportedInterfaces": [{"url": ..., "protocolBinding": "JSONRPC"}], ...}
    3. Runtime        connect to the URL the card names
"""
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx
import truststore
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

# Verify HTTPS against the OS trust store instead of certifi's bundle, so corporate
# TLS-inspecting proxies (whose root CA lives only in the system keychain) still verify.
truststore.inject_into_ssl()

NANDA_INDEX_URL = os.environ.get("NANDA_INDEX_URL", "https://api.nandaindex.org").rstrip("/")
MCP_CARD_TYPE = "application/mcp-server-card+json"
A2A_CARD_TYPE = "application/a2a-agent-card+json"
TRANSPORT_ALIASES = {"sse": "sse", "streamable-http": "streamable-http", "streamable_http": "streamable-http", "http": "streamable-http"}


class DiscoveryError(Exception):
    pass


@dataclass
class DiscoveredServer:
    identifier: str
    name: str
    card_url: str
    endpoint: str
    transport: str
    card: dict = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.name} ({self.identifier}) via {self.card_url} -> {self.transport} {self.endpoint}"


@dataclass
class DiscoveredAgent:
    identifier: str
    name: str
    card_url: str
    endpoint: str
    protocol_version: str
    skills: list[str]
    card: dict

    def describe(self) -> str:
        return (f"{self.name} ({self.identifier}) via {self.card_url} -> "
                f"A2A {self.protocol_version} JSON-RPC {self.endpoint}")


async def get_json(client: httpx.AsyncClient, url: str, **params) -> dict:
    try:
        response = await client.get(url, params=params or None, headers={"Accept": "application/json"})
    except httpx.HTTPError as e:
        raise DiscoveryError(f"Could not reach {url}: {e}") from e
    if response.status_code >= 400:
        raise DiscoveryError(f"{url} returned HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as e:
        raise DiscoveryError(f"{url} did not return JSON (a proxy or firewall may be blocking it)") from e


async def read_card(client: httpx.AsyncClient, identifier: str, name: str, card_url: str) -> DiscoveredServer:
    card = await get_json(client, card_url)
    transport = card.get("transport") or {}
    endpoint = transport.get("url") or transport.get("endpoint") or card.get("url")
    kind = TRANSPORT_ALIASES.get(str(transport.get("type", "sse")).lower())
    auth = (card.get("authentication") or {}).get("type", "none")
    if not endpoint:
        raise DiscoveryError(f"Server card {card_url} has no transport URL")
    if kind is None:
        raise DiscoveryError(f"Server card {card_url} uses unsupported transport '{transport.get('type')}'")
    if auth not in ("none", None):
        raise DiscoveryError(f"Server card {card_url} requires '{auth}' authentication, which this client doesn't support")
    return DiscoveredServer(identifier, card.get("name") or name, card_url, endpoint, kind, card)


async def read_agent_card(client: httpx.AsyncClient, identifier: str, name: str, card_url: str) -> DiscoveredAgent:
    card = await get_json(client, card_url)
    # A2A 1.0 cards list interfaces; 0.3 cards have a single url + preferredTransport.
    interfaces = card.get("supportedInterfaces") or []
    if not interfaces and card.get("url"):
        interfaces = [{"url": card["url"], "protocolBinding": card.get("preferredTransport", "JSONRPC"),
                       "protocolVersion": card.get("protocolVersion", "0.3")}]
    jsonrpc = [i for i in interfaces if str(i.get("protocolBinding", "")).upper() == "JSONRPC" and i.get("url")]
    if not jsonrpc:
        raise DiscoveryError(f"Agent card {card_url} has no JSON-RPC interface")
    chosen = max(jsonrpc, key=lambda i: str(i.get("protocolVersion", "")))
    if card.get("securityRequirements") or card.get("security"):
        raise DiscoveryError(f"Agent card {card_url} requires authentication, which this client doesn't support")
    skills = [s.get("name") or s.get("id", "") for s in card.get("skills", [])]
    return DiscoveredAgent(identifier, card.get("name") or name, card_url, chosen["url"],
                           str(chosen.get("protocolVersion", "")), skills, card)


async def resolve_entry(client: httpx.AsyncClient, locator: str, media_type: str, index_url: str) -> dict:
    data = await get_json(client, f"{index_url}/api/v1/resolve", locator=locator)
    entry = data.get("entry") or {}
    if data.get("match") != "exact":
        raise DiscoveryError(f"No entry is registered under exactly '{locator}'")
    if entry.get("type") != media_type:
        raise DiscoveryError(f"'{locator}' is a {entry.get('type')}, not a {media_type}")
    if not entry.get("url"):
        raise DiscoveryError(f"'{locator}' has no card URL")
    return entry


async def search_entries(client: httpx.AsyncClient, query: str, media_type: str, index_url: str) -> list[dict]:
    data = await get_json(client, f"{index_url}/api/v1/search", q=query)
    return [r for r in data.get("results", [])
            if r.get("media_type") == media_type and r.get("status") == "active" and r.get("registry_url")]


async def resolve(locator: str, *, index_url: str = NANDA_INDEX_URL) -> DiscoveredServer:
    """Resolve an exact urn:air: identifier to a connectable MCP server."""
    async with httpx.AsyncClient(timeout=20) as client:
        entry = await resolve_entry(client, locator, MCP_CARD_TYPE, index_url)
        return await read_card(client, entry.get("identifier", locator), entry.get("displayName", locator), entry["url"])


async def search(query: str, *, index_url: str = NANDA_INDEX_URL) -> list[DiscoveredServer]:
    """Keyword-search the index and return every active MCP server whose card this client can use."""
    found = []
    async with httpx.AsyncClient(timeout=20) as client:
        for record in await search_entries(client, query, MCP_CARD_TYPE, index_url):
            try:
                found.append(await read_card(client, record["identifier"], record["display_name"], record["registry_url"]))
            except DiscoveryError:
                continue  # unreachable card, OAuth-only server, etc.
    return found


async def resolve_agent(locator: str, *, index_url: str = NANDA_INDEX_URL) -> DiscoveredAgent:
    """Resolve an exact urn:air: identifier to an A2A agent this client can call."""
    async with httpx.AsyncClient(timeout=20) as client:
        entry = await resolve_entry(client, locator, A2A_CARD_TYPE, index_url)
        return await read_agent_card(client, entry.get("identifier", locator), entry.get("displayName", locator), entry["url"])


async def search_agents(query: str, *, index_url: str = NANDA_INDEX_URL) -> list[DiscoveredAgent]:
    """Keyword-search the index for active A2A agents whose cards this client can use."""
    found = []
    async with httpx.AsyncClient(timeout=20) as client:
        for record in await search_entries(client, query, A2A_CARD_TYPE, index_url):
            try:
                found.append(await read_agent_card(client, record["identifier"], record["display_name"], record["registry_url"]))
            except DiscoveryError:
                continue
    return found


@dataclass
class AgentCandidate:
    agent: DiscoveredAgent
    score: float
    description: str
    tags: list[str]


async def agentic_search_entries(client: httpx.AsyncClient, task: str, media_type: str, index_url: str,
                                 limit: int = 10) -> list[dict]:
    """Natural-language task search: the index ranks entries by what they do, so no name is needed."""
    data = await get_json(client, f"{index_url}/api/v1/agentic-search", q=task.strip()[:128], limit=limit)
    return [c for c in data.get("candidates", []) if c.get("type") == media_type and c.get("url")]


async def read_candidates(client: httpx.AsyncClient, entries: list[dict]) -> list[AgentCandidate]:
    found = []
    for entry in entries:
        try:
            agent = await read_agent_card(client, entry["identifier"], entry.get("displayName", entry["identifier"]), entry["url"])
        except DiscoveryError:
            continue  # card unreachable, needs auth, or no JSON-RPC interface
        found.append(AgentCandidate(agent, float(entry.get("score") or 0), entry.get("description") or "",
                                    list(entry.get("tags") or [])))
    return sorted(found, key=lambda c: c.score, reverse=True)


async def find_agent_candidates(client: httpx.AsyncClient, task: str, index_url: str = NANDA_INDEX_URL) -> list[AgentCandidate]:
    """A2A agents the index considers able to do `task`, with their cards, best score first."""
    return await read_candidates(client, await agentic_search_entries(client, task, A2A_CARD_TYPE, index_url))


async def find_server_candidates(client: httpx.AsyncClient, task: str, index_url: str = NANDA_INDEX_URL,
                                 ) -> list[tuple[DiscoveredServer, float, str]]:
    """MCP servers the index considers able to do `task`: (server, score, description), best first."""
    found = []
    for entry in await agentic_search_entries(client, task, MCP_CARD_TYPE, index_url):
        try:
            server = await read_card(client, entry["identifier"], entry.get("displayName", entry["identifier"]), entry["url"])
        except DiscoveryError:
            continue
        found.append((server, float(entry.get("score") or 0), entry.get("description") or ""))
    return sorted(found, key=lambda f: f[1], reverse=True)


async def discover_agent(locator: Optional[str] = None, query: Optional[str] = None) -> DiscoveredAgent:
    if locator:
        return await resolve_agent(locator)
    if query:
        agents = await search_agents(query)
        if not agents:
            raise DiscoveryError(f"The NANDA Index has no usable A2A agent matching '{query}'")
        return agents[0]
    raise DiscoveryError("Give a NANDA locator or a search query")


async def discover(locator: Optional[str] = None, query: Optional[str] = None) -> DiscoveredServer:
    if locator:
        return await resolve(locator)
    if query:
        servers = await search(query)
        if not servers:
            raise DiscoveryError(f"The NANDA Index has no usable MCP server matching '{query}'")
        return servers[0]
    raise DiscoveryError("Give a NANDA locator or a search query")


@asynccontextmanager
async def mcp_session(endpoint: str, transport: str = "sse") -> AsyncIterator[ClientSession]:
    if transport == "streamable-http":
        async with streamablehttp_client(endpoint) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    else:
        async with sse_client(endpoint) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


async def choose_server(direct_url: str, locator: Optional[str], query: Optional[str]) -> tuple[str, str, str]:
    """Return (endpoint, transport, how it was found), preferring NANDA discovery when asked for."""
    if locator or query:
        server = await discover(locator, query)
        return server.endpoint, server.transport, f"NANDA Index: {server.describe()}"
    return direct_url, "sse", f"direct URL {direct_url}"


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Look up MCP servers or A2A agents in the NANDA Index")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--locator", help="Exact identifier, e.g. urn:air:host39.org:personal:you@example.com")
    group.add_argument("--search", help="Keywords, e.g. 'product catalog'")
    parser.add_argument("--agents", action="store_true", help="Look for A2A agents instead of MCP servers")
    args = parser.parse_args()

    async def show() -> None:
        try:
            if args.agents:
                found = [await resolve_agent(args.locator)] if args.locator else await search_agents(args.search)
            else:
                found = [await resolve(args.locator)] if args.locator else await search(args.search)
        except DiscoveryError as e:
            raise SystemExit(f"Discovery failed: {e}")
        if not found:
            raise SystemExit(f"No usable {'A2A agents' if args.agents else 'MCP servers'} found.")
        for item in found:
            print(item.describe())
            if args.agents and item.skills:
                print(f"  skills: {', '.join(item.skills)}")

    asyncio.run(show())
