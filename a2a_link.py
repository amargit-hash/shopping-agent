"""One A2A conversation with another agent (a single contextId keeps its state, e.g. the cart)."""
import uuid
from typing import Any, Optional

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict, ParseDict

from nanda_discovery import DiscoveredAgent


def request_metadata(context: Any) -> dict:
    """Metadata of the incoming A2A message (server side), as plain JSON."""
    message = getattr(context, "message", None)
    try:
        return MessageToDict(message.metadata) if message is not None else {}
    except Exception:
        return {}


class A2ALink:
    def __init__(self, agent: DiscoveredAgent, http: httpx.AsyncClient, *, me: str = "gift-planner",
                 them: str = "shopping-agent", metadata: Optional[dict] = None, quiet: bool = False) -> None:
        self.agent = agent
        self.http = http
        self.me, self.them, self.quiet = me, them, quiet
        self.metadata = metadata or {}
        self.context_id = f"{me}-{uuid.uuid4().hex[:8]}"
        self.client = None
        self.negotiations: list[dict] = []  # every "negotiations" outcome the other agent reported

    async def connect(self) -> None:
        card = ParseDict(self.agent.card, AgentCard(), ignore_unknown_fields=True)
        self.client = await create_client(card, client_config=ClientConfig(httpx_client=self.http, streaming=False))

    async def send(self, text: str) -> tuple[str, dict[str, Any]]:
        """Send one message; return the reply text and the reply message's metadata."""
        if not self.quiet:
            print(f"\n[{self.me} -> {self.them}] {text}")
        message = new_text_message(text, role=Role.ROLE_USER, context_id=self.context_id)
        if self.metadata:
            message.metadata.update(self.metadata)
        replies, metadata = [], {}
        async for event in self.client.send_message(SendMessageRequest(message=message)):
            if event.HasField("message"):
                replies.append(get_message_text(event.message))
                metadata.update(MessageToDict(event.message.metadata))
            elif event.HasField("task"):
                task = event.task
                if task.status.HasField("message"):
                    replies.append(get_message_text(task.status.message))
                    metadata.update(MessageToDict(task.status.message.metadata))
                replies += [get_artifact_text(a) for a in task.artifacts]
        reply = "\n".join(r for r in replies if r) or "(no reply)"
        self.negotiations += [n for n in metadata.get("negotiations") or [] if isinstance(n, dict)]
        if not self.quiet:
            print(f"[{self.them} -> {self.me}]")
            print("  " + reply.replace("\n", "\n  "))
        return reply, metadata

    async def ask(self, text: str) -> str:
        return (await self.send(text))[0]

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
