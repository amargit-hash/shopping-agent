"""Personal shopping agent that talks to the product catalog MCP server over SSE.

Run the server first (python product_server.py), then:
    python shopping_agent.py                      # interactive chat
    python shopping_agent.py "find headphones under 300" "add P003"   # one-shot commands
"""
import argparse
import asyncio
import json
import os
import re
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

DEFAULT_SERVER_URL = os.environ.get("PRODUCT_SERVER_URL", "http://localhost:8081/sse")

PRODUCT_ID_RE = re.compile(r"\b(?:P\d{3,5}|[A-Z]{2}-\d{5})\b", re.IGNORECASE)  # P123 or a seller SKU like SG-10042
MAX_PRICE_RE = re.compile(r"(?:under|below|less than|cheaper than|max(?:imum)?|up to|within)\s*\$?\s*(\d+(?:\.\d+)?)")
MIN_PRICE_RE = re.compile(r"(?:over|above|more than|at least|min(?:imum)?)\s*\$?\s*(\d+(?:\.\d+)?)")
BETWEEN_RE = re.compile(r"between\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|-|to)\s*\$?\s*(\d+(?:\.\d+)?)")
RATING_RE = re.compile(r"(?:rating|rated)\s*(?:of\s*)?(?:above|over|at least)?\s*(\d(?:\.\d)?)")
QUANTITY_RE = re.compile(r"(?:\bx\s*(\d+)\b|\b(\d+)\s*(?:x\b|units?\b|pcs\b|of\b))")
DESCRIPTIVE_CATEGORIES = {"gaming", "home", "fitness", "beauty"}
MONEY_RE = re.compile(r"\$?\s*(\d+(?:\.\d+)?)")
CATEGORY_SYNONYMS = {
    "computer": "laptops", "computers": "laptops", "pc": "laptops", "notebook": "laptops",
    "notebooks": "laptops", "macbook": "laptops", "chromebook": "laptops",
    "novel": "books", "novels": "books", "cookbook": "books",
    "smartphone": "phones", "smartphones": "phones", "mobile": "phones",
    "earbuds": "headphones", "earphones": "headphones", "headset": "headphones",
    "sneakers": "shoes", "smartwatch": "watches", "ipad": "tablets",
}

STOPWORDS = {
    "a", "an", "the", "me", "i", "my", "some", "any", "for", "with", "and", "or", "to", "of", "in", "on",
    "find", "search", "show", "look", "looking", "want", "need", "get", "buy", "please", "can", "you",
    "good", "nice", "great", "best", "top", "rated", "cheap", "cheapest", "popular", "affordable",
    "under", "below", "less", "than", "over", "above", "more", "between", "up", "max", "maximum",
    "min", "minimum", "at", "least", "within", "rating", "stock", "available", "is", "are", "that",
    "which", "what", "do", "have", "something", "recommend", "suggest", "products", "items", "price",
}

HELP_TEXT = """\
I'm your personal shopping agent. Try things like:
  categories                         - see what the store sells
  find wireless headphones under 300 - search the catalog
  cheapest laptops in stock          - search with sorting
  show P005                          - full product details
  compare P001 P002 P003             - side-by-side comparison
  recommend a laptop                 - picks based on your budget and preferences
  budget 500                         - set your per-item budget
  prefer wireless, travel            - remember things you care about
  add P003 x2 / remove P003          - manage your cart
  cart / checkout                    - review or place your order
  quit                               - exit"""


class ShoppingAgent:
    def __init__(self, session: ClientSession):
        self.session = session
        self.cart: dict[str, dict] = {}
        self.budget: Optional[float] = None
        self.preferences: list[str] = []
        self.categories: dict[str, dict] = {}
        self.category_aliases: dict[str, str] = {}

    async def call(self, tool: str, **arguments: Any) -> dict:
        arguments = {k: v for k, v in arguments.items() if v is not None}
        result = await self.session.call_tool(tool, arguments)
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            raise RuntimeError(f"{tool} failed: {text}")
        return json.loads(text)

    async def start(self) -> None:
        self.tools = (await self.session.list_tools()).tools
        self.tool_names = [t.name for t in self.tools]
        self.categories = (await self.call("list_categories"))["categories"]
        for name in self.categories:
            self.category_aliases[name] = name
            # Singular forms: "laptop", "watch", "toy" ("fitness"/"gaming" stay as-is).
            if name.endswith("es") and name[:-2].endswith(("ch", "sh", "x")):
                self.category_aliases[name[:-2]] = name
            elif name.endswith("s") and not name.endswith("ss"):
                self.category_aliases[name[:-1]] = name
        for word, name in CATEGORY_SYNONYMS.items():
            if name in self.categories:
                self.category_aliases.setdefault(word, name)

    # ------------------------------------------------------------------
    # Intent handling
    # ------------------------------------------------------------------

    async def handle(self, message: str) -> str:
        text = message.strip()
        lower = text.lower()
        ids = [i.upper() for i in PRODUCT_ID_RE.findall(text)]
        words = re.findall(r"[a-z0-9\-]+", lower)
        first = words[0] if words else ""

        if not text:
            return ""
        if first in {"help", "?"} or lower == "what can you do":
            return HELP_TEXT
        if first in {"categories", "category"} or "what do you sell" in lower:
            return self.format_categories()
        if first == "budget" or "my budget" in lower:
            return self.set_budget(lower)
        if first in {"prefer", "preferences", "like"} or lower.startswith("i like") or lower.startswith("i prefer"):
            return self.set_preferences(lower)
        if first == "cart":
            return self.format_cart()
        if first == "checkout" or "place order" in lower:
            return await self.checkout()
        if first == "remove" and ids:
            return self.remove_from_cart(ids, lower)
        if first in {"add", "buy"} and ids:
            return await self.add_to_cart(ids, lower)
        if first == "compare" or len(ids) >= 2:
            return await self.compare(ids)
        if ids:
            return await self.show_product(ids[0])
        if first in {"recommend", "suggest"} or "recommend" in lower or "suggest" in lower:
            return await self.recommend(lower, words)
        return await self.search(lower, words)

    # ------------------------------------------------------------------
    # Preferences & budget
    # ------------------------------------------------------------------

    def set_budget(self, lower: str) -> str:
        match = MONEY_RE.search(lower)
        if not match:
            return f"Your budget is {'not set' if self.budget is None else f'${self.budget:.2f}'}. Try 'budget 500'."
        self.budget = float(match.group(1))
        return f"Got it, I'll keep recommendations under ${self.budget:.2f} per item."

    def set_preferences(self, lower: str) -> str:
        body = re.sub(r"^(i\s+)?(prefer|preferences|like)\s*:?\s*", "", lower)
        new = [p.strip() for p in re.split(r",|\band\b", body) if p.strip()]
        for pref in new:
            if pref not in self.preferences:
                self.preferences.append(pref)
        if not self.preferences:
            return "No preferences saved yet. Try 'prefer wireless, travel'."
        return "I'll prioritize: " + ", ".join(self.preferences)

    # ------------------------------------------------------------------
    # Catalog queries
    # ------------------------------------------------------------------

    def extract_filters(self, lower: str, words: list[str]) -> dict:
        filters: dict[str, Any] = {}
        if m := BETWEEN_RE.search(lower):
            filters["min_price"], filters["max_price"] = float(m.group(1)), float(m.group(2))
        else:
            if m := MAX_PRICE_RE.search(lower):
                filters["max_price"] = float(m.group(1))
            if m := MIN_PRICE_RE.search(lower):
                filters["min_price"] = float(m.group(1))
        if m := RATING_RE.search(lower):
            filters["min_rating"] = float(m.group(1))

        # Categories like "gaming" or "home" often describe another product
        # ("gaming laptop", "home speaker"), so a noun category wins when both appear.
        matched = [w for w in words if w in self.category_aliases]
        nouns = [w for w in matched if self.category_aliases[w] not in DESCRIPTIVE_CATEGORIES]
        category_word = (nouns or matched or [None])[0]
        if category_word:
            filters["category"] = self.category_aliases[category_word]

        if "cheap" in lower or "lowest price" in lower or "affordable" in lower:
            filters["sort_by"] = "price_asc"
        elif "expensive" in lower or "premium" in lower:
            filters["sort_by"] = "price_desc"
        elif "popular" in lower:
            filters["sort_by"] = "popularity"
        else:
            filters["sort_by"] = "rating"
        if "in stock" in lower or "available" in lower:
            filters["in_stock_only"] = True

        keywords = [
            w for w in words
            if w not in STOPWORDS
            and w != category_word
            and not re.fullmatch(r"\d+(\.\d+)?", w)
            and w not in {"stock", "expensive", "premium"}
        ]
        filters["query"] = " ".join(keywords)
        return filters

    async def search(self, lower: str, words: list[str]) -> str:
        filters = self.extract_filters(lower, words)
        result = await self.call("search_products", **filters)
        note = ""
        if result["count"] == 0 and filters["query"] and "category" in filters:
            # The keyword may live in another category ("home speaker" -> electronics).
            anywhere = {k: v for k, v in filters.items() if k != "category"}
            result = await self.call("search_products", **anywhere)
            if result["count"]:
                note = f"No {filters['category']} matched '{filters['query']}', but I found these in other categories:\n"
        has_other_filters = any(k in filters for k in ("category", "min_price", "max_price", "min_rating"))
        if result["count"] == 0 and filters["query"] and has_other_filters:
            relaxed = {**filters, "query": ""}
            result = await self.call("search_products", **relaxed)
            if result["count"]:
                note = f"Nothing matched '{filters['query']}' exactly, so here are the closest results:\n"
        if result["count"] == 0:
            return "I couldn't find anything matching that. Try 'categories' to see what's available."
        return note + self.format_product_list(result["products"], result.get("total"))

    async def show_product(self, product_id: str) -> str:
        result = await self.call("get_product", product_id=product_id)
        if "error" in result:
            return result["error"]
        p = result["product"]
        specs = ", ".join(f"{k}={v}" for k, v in p["specs"].items())
        stock = f"{p['stock']} in stock" if p["stock"] else "OUT OF STOCK"
        budget_note = ""
        if self.budget is not None and p["price"] > self.budget:
            budget_note = f"\n  Note: this is ${p['price'] - self.budget:.2f} over your ${self.budget:.2f} budget."
        return (
            f"{p['id']} - {p['name']} ({p['brand']})\n"
            f"  ${p['price']:.2f} | {p['rating']} stars from {p['reviews']:,} reviews | {stock}\n"
            f"  {p['description']}\n"
            f"  Specs: {specs}\n"
            f"  Tags: {', '.join(p['tags'])}"
            f"{budget_note}"
        )

    async def compare(self, ids: list[str]) -> str:
        if len(ids) < 2:
            return "Give me at least two product IDs to compare, e.g. 'compare P001 P002'."
        result = await self.call("compare_products", product_ids=ids)
        lines = []
        for p in result["comparison"]:
            stock = "in stock" if p["in_stock"] else "out of stock"
            lines.append(f"  {p['id']}  ${p['price']:>8.2f}  {p['rating']} stars ({p['reviews']:,})  {stock}  {p['name']}")
        if result["not_found"]:
            lines.append(f"  Not found: {', '.join(result['not_found'])}")
        if summary := result["summary"]:
            names = {p["id"]: p["name"] for p in result["comparison"]}
            lines.append(f"Cheapest: {names[summary['cheapest']]}")
            lines.append(f"Highest rated: {names[summary['highest_rated']]}")
            if summary["cheapest"] == summary["highest_rated"]:
                lines.append(f"My pick: {names[summary['cheapest']]}, it's both the cheapest and the best rated.")
        return "Comparison:\n" + "\n".join(lines)

    async def recommend(self, lower: str, words: list[str]) -> str:
        filters = self.extract_filters(lower, words)
        budget = filters.get("max_price", self.budget)
        if budget is None:
            budget = 10_000.0
        prefs = list(self.preferences)
        prefs += [w for w in filters["query"].split() if w not in prefs]
        result = await self.call(
            "recommend_products",
            budget=budget,
            category=filters.get("category"),
            preferences=prefs or None,
            limit=3,
        )
        recs = result["recommendations"]
        if not recs:
            what = filters.get("category", "anything")
            cheapest = await self.call(
                "search_products", category=filters.get("category"), in_stock_only=True, sort_by="price_asc", limit=1
            )
            if cheapest["products"]:
                p = cheapest["products"][0]
                return (f"No {what} in stock under ${budget:.2f}. The cheapest is "
                        f"{p['name']} at ${p['price']:.2f}.")
            return f"I couldn't find {what} in stock under ${budget:.2f}. Try raising your budget."
        header = f"Top picks under ${budget:.2f}" if budget < 10_000 else "Top picks"
        if prefs:
            header += f" (prioritizing {', '.join(prefs)})"
        lines = [header + ":"]
        for i, p in enumerate(recs, 1):
            why = f" - matches {', '.join(p['matched_preferences'])}" if p["matched_preferences"] else ""
            lines.append(f"  {i}. {p['id']} {p['name']}  ${p['price']:.2f}  {p['rating']} stars{why}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cart
    # ------------------------------------------------------------------

    async def add_to_cart(self, ids: list[str], lower: str) -> str:
        qty_match = QUANTITY_RE.search(PRODUCT_ID_RE.sub("", lower))
        quantity = int(next(g for g in qty_match.groups() if g)) if qty_match else 1
        messages = []
        for pid in ids:
            current = self.cart.get(pid, {}).get("quantity", 0)
            check = await self.call("check_availability", product_id=pid, quantity=current + quantity)
            if "error" in check:
                messages.append(check["error"])
                continue
            if check["stock"] == 0:
                messages.append(f"Sorry, {check['name']} is out of stock.")
                continue
            if not check["available"]:
                messages.append(f"Sorry, only {check['stock']} of {check['name']} available.")
                continue
            self.cart[pid] = {
                "name": check["name"],
                "category": check.get("category"),
                "unit_price": check["unit_price"],
                "quantity": current + quantity,
            }
            msg = f"Added {quantity} x {check['name']} (${check['unit_price']:.2f} each) to your cart."
            if self.budget is not None and check["unit_price"] > self.budget:
                msg += f" Heads up: it's over your ${self.budget:.2f} budget."
            messages.append(msg)
        return "\n".join(messages)

    def remove_from_cart(self, ids: list[str], lower: str = "") -> str:
        qty_match = QUANTITY_RE.search(PRODUCT_ID_RE.sub("", lower))
        quantity = int(next(g for g in qty_match.groups() if g)) if qty_match else None
        removed = []
        for pid in ids:
            item = self.cart.get(pid)
            if item is None:
                continue
            if quantity is None or quantity >= item["quantity"]:
                self.cart.pop(pid)
                removed.append(item["name"])
            else:
                item["quantity"] -= quantity
                removed.append(f"{quantity} x {item['name']} ({item['quantity']} left)")
        if not removed:
            return "None of those items are in your cart."
        return "Removed: " + ", ".join(removed)

    def cart_total(self) -> float:
        return round(sum(i["unit_price"] * i["quantity"] for i in self.cart.values()), 2)

    def format_cart(self) -> str:
        if not self.cart:
            return "Your cart is empty."
        lines = ["Your cart:"]
        for pid, item in self.cart.items():
            subtotal = item["unit_price"] * item["quantity"]
            lines.append(f"  {pid}  {item['quantity']} x {item['name']}  ${subtotal:.2f}")
        lines.append(f"Total: ${self.cart_total():.2f}")
        return "\n".join(lines)

    async def checkout(self) -> str:
        if not self.cart:
            return "Your cart is empty, nothing to check out."
        problems = []
        for pid, item in self.cart.items():
            check = await self.call("check_availability", product_id=pid, quantity=item["quantity"])
            if not check.get("available"):
                problems.append(f"{item['name']} (only {check.get('stock', 0)} left)")
        if problems:
            return "Can't check out, these items are no longer available: " + "; ".join(problems)
        summary = self.format_cart()
        self.cart.clear()
        return summary + "\nOrder placed (simulated). Thanks for shopping!"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_categories(self) -> str:
        lines = ["Categories:"]
        for name, info in self.categories.items():
            lines.append(f"  {name:<12} {info['count']} products  ${info['min_price']:.2f} - ${info['max_price']:.2f}")
        return "\n".join(lines)

    def format_product_list(self, products: list[dict], total: Optional[int] = None) -> str:
        if total is not None and total > len(products):
            lines = [f"Found {total} products, here are the top {len(products)}. Narrow it down with a price or keyword:"]
        else:
            lines = [f"Found {len(products)} product(s):"]
        for p in products:
            stock = "" if p["in_stock"] else "  (out of stock)"
            over = ""
            if self.budget is not None and p["price"] > self.budget:
                over = "  (over budget)"
            lines.append(f"  {p['id']}  ${p['price']:>8.2f}  {p['rating']} stars  {p['name']}{stock}{over}")
        return "\n".join(lines)


async def run(server_url: str, commands: list[str]) -> None:
    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            agent = ShoppingAgent(session)
            await agent.start()

            if commands:
                for command in commands:
                    print(f"you> {command}")
                    print(f"agent> {await agent.handle(command)}\n")
                return

            print(f"Connected to {server_url} (tools: {', '.join(agent.tool_names)})")
            print(HELP_TEXT + "\n")
            loop = asyncio.get_running_loop()
            while True:
                try:
                    message = await loop.run_in_executor(None, input, "you> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if message.strip().lower() in {"quit", "exit", "bye"}:
                    break
                try:
                    reply = await agent.handle(message)
                except Exception as e:
                    reply = f"Something went wrong talking to the store: {e}"
                if reply:
                    print(f"agent> {reply}\n")
            print("Goodbye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal shopping agent (MCP client)")
    parser.add_argument("commands", nargs="*", help="Run these commands and exit instead of starting a chat")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help="Product server SSE URL")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.commands))


if __name__ == "__main__":
    main()
