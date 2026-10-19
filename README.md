# Shopping Agent

A personal shopping agent that finds, compares and negotiates product prices, and the
product catalog it shops from. Agents find each other through the
[NANDA Index](https://nandaindex.org), a directory that maps an agent identifier to the
agent's card, which in turn points at the agent's runtime endpoint.

```
Other agent --A2A message/send--> shopping agent --MCP tools--> product catalog (seller)
                                        |
                                        +--NANDA Index--> other sellers, negotiator agent
```

## Components

| File | What it is | Card |
|------|-----------|------|
| `a2a_server.py` | The shopping agent as an A2A server | `/.well-known/agent-card.json` |
| `product_server.py` | A seller's product catalog as an MCP server, including price negotiation | `/.well-known/mcp-server-card.json` |
| `negotiator_agent.py` | A2A agent that negotiates across several sellers | `/.well-known/agent-card.json` |
| `specialist_agent.py` | Advisor agents (laptop, gift, travel) | `/.well-known/agent-card.json` |
| `claude_agent.py` | Command-line shopping agent driven by Claude | — |
| `agent_ui.py`, `flow_ui.py` | Browser UIs for chatting with the agent and watching the message flow | — |
| `nanda_register.py` | Registers a card in the NANDA Index | — |
| `nanda_discovery.py` | Resolves an index identifier to a usable endpoint | — |

## Run it locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # optional: without it a rule-based agent answers

python product_server.py                   # catalog on :8081
PRODUCT_SERVER_URL=http://localhost:8081/sse python a2a_server.py   # agent on :8083
```

Then ask it something:

```bash
curl -X POST http://localhost:8083/ -H 'Content-Type: application/json' -d '{
  "jsonrpc": "2.0", "id": 1, "method": "message/send",
  "params": {"message": {"role": "user", "messageId": "1",
             "parts": [{"kind": "text", "text": "recommend headphones under 150"}]}}}'
```

## Deploy

`render.yaml` is a Render blueprint for both services. Deploy the catalog first, then set
`PRODUCT_SERVER_URL` on the agent to the catalog's `/sse` URL: the agent opens its MCP
session at startup and will not start without it.

Each service needs a public URL in its own card. Render supplies `RENDER_EXTERNAL_URL` and
Railway supplies `RAILWAY_PUBLIC_DOMAIN` automatically; anywhere else, set `A2A_PUBLIC_URL`
(agent) or `PUBLIC_BASE_URL` (catalog), or the card will advertise `localhost` and callers
will have nowhere to send their messages.

## Register in a NANDA Index

The index needs a public `https://` card URL; it rejects `localhost`. Registration is a
Personal entry, so no domain is required, but the email becomes part of the public
identifier and each email can hold only one entry.

```bash
# Shopping agent
python nanda_register.py --a2a --card-url https://<your-app>/.well-known/agent-card.json

# Product catalog, under a different email
python nanda_register.py --card-url https://<your-catalog>/.well-known/mcp-server-card.json

# Check whether an entry is live
python nanda_register.py --a2a --status you@example.com
```

`NANDA_INDEX_URL` selects the index (default `https://api.nandaindex.org`). Verify the
emailed link afterwards; an entry stays pending until then.

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | agent, catalog, negotiator | Claude-driven answers and price negotiation |
| `PRODUCT_SERVER_URL` | agent, specialists | Catalog SSE endpoint when not discovering through the index |
| `NANDA_INDEX_URL` | all | Which index to search and register in |
| `NANDA_LOCATOR` / `NANDA_SEARCH` | agent, specialists | Find the catalog through the index instead of a fixed URL |
| `A2A_PUBLIC_URL`, `A2A_PORT`, `A2A_HOST` | agent | Public URL and bind address |
| `PUBLIC_BASE_URL`, `PORT` | catalog | Public URL and bind port |
| `SELLER_ID`, `SELLER_NAME`, `SELLER_PRICE_SPREAD`, `SELLER_SKU_PREFIX`, `SELLER_CATALOG_SHARE`, `SELLER_VARIANT_SHARE`, `SELLER_FLOOR_FRACTION` | catalog | Run several instances as competing sellers with their own SKUs and prices |

## Tests

```bash
python -m pytest tests
```

## Note

Orders and price quotes are simulated. Nothing is reserved, sold or charged.
