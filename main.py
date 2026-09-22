"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse
import json
import os
import asyncio
import boto3
import logging
import uuid
import re
from typing import Dict

from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
    AgentInitializedEvent,
)
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-ghhrqrxtkm.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "NWTCXWG5KH"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-lAo1dN6cdW"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "us.amazon.nova-pro-v1:0"
model = BedrockModel(model_id=model_id)

memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {
        strategy["type"]: strategy["namespaces"][0]
        for strategy in strategies
    }


# ── Helper: safe memory role ──────────────────────────────────────────────────
def safe_role(msg) -> str:
    """Normalize to USER | ASSISTANT | TOOL | OTHER."""
    if isinstance(msg, str):
        r = msg.strip().lower()
        if r in ("user", "human"):
            return "USER"
        if r in ("assistant", "ai", "model"):
            return "ASSISTANT"
        if r in ("tool", "function"):
            return "TOOL"
        return "OTHER"

    if isinstance(msg, dict):
        r = msg.get("role")
        if isinstance(r, str):
            r = r.strip().lower()
            if r in ("user", "human"):
                return "USER"
            if r in ("assistant", "ai", "model"):
                return "ASSISTANT"
            if r in ("tool", "function"):
                return "TOOL"
        if "content" in msg and "role" not in msg:
            return "ASSISTANT"
    return "OTHER"


def extract_text(msg) -> str:
    """Pull plain text out of a Strands / dict message."""
    if not isinstance(msg, dict):
        return str(msg)

    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        ).strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────

    """
    Automatically loads recent conversation turns when the agent starts
    and saves new messages after they are added.
    """

    def __init__(self, mem_client: MemoryClient, memory_id: str, actor_id: str, session_id: str):
        self.mem_client = mem_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id

    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load the last few turns from memory into the system prompt."""
        try:
            turns = self.mem_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=5,
            )
            if turns:
                context_lines = []
                for turn in turns:
                    for msg in turn:
                        role = msg.get("role", "unknown")
                        text = msg.get("content", {}).get("text", str(msg.get("content", "")))
                        context_lines.append(f"{role}: {text}")
                if context_lines:
                    event.agent.system_prompt += (
                        "\n\nRecent conversation history:\n" + "\n".join(context_lines)
                    )
        except Exception as e:
            logger.warning(f"Could not load memory turns: {e}")

    def on_agent_initialized(self, event: AgentInitializedEvent):
     """Load recent turns + long-term facts/preferences into the system prompt."""
    try:
        # 1. Short-term: last few turns of this session
        turns = self.mem_client.get_last_k_turns(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            k=5,
        )
        if turns:
            context_lines = []
            for turn in turns:
                for msg in turn:
                    role = msg.get("role", "unknown")
                    text = msg.get("content", {}).get("text", str(msg.get("content", "")))
                    context_lines.append(f"{role}: {text}")
            if context_lines:
                event.agent.system_prompt += (
                    "\n\nRecent conversation history:\n" + "\n".join(context_lines)
                )
    except Exception as e:
        logger.warning(f"Could not load memory turns: {e}")

    # 2. Long-term: user preferences + facts
    try:
        namespaces = [
            f"/cs_agent/{self.actor_id}/preferences/",
            f"/cs_agent/{self.actor_id}/facts/",
        ]

        long_term_bits = []
        for ns in namespaces:
            try:
                records = self.mem_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=ns,
                    query="name preference communication style customer facts",
                    top_k=5,
                )
                for r in records or []:
                    # Handle different possible shapes returned by the SDK
                    text = None
                    if isinstance(r, dict):
                        text = (
                            r.get("content", {}).get("text")
                            or r.get("text")
                            or r.get("memoryRecord", {}).get("content", {}).get("text")
                        )
                    if text:
                        long_term_bits.append(text)
            except Exception as e:
                logger.warning(f"Could not retrieve from namespace {ns}: {e}")

        if long_term_bits:
            event.agent.system_prompt += (
                "\n\nKnown facts and preferences about this customer:\n"
                + "\n".join(f"- {bit}" for bit in long_term_bits)
            )
    except Exception as e:
        logger.warning(f"Could not load long-term memories: {e}")

    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(MessageAddedEvent, self.on_message_added)
class MemoryHook(HookProvider):
    """Short-term turns + long-term facts/preferences."""

    def __init__(self, mem_client: MemoryClient, memory_id: str, actor_id: str, session_id: str):
        self.mem_client = mem_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id

    def on_agent_initialized(self, event: AgentInitializedEvent):
        # Short-term
        try:
            turns = self.mem_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=5,
            )
            if turns:
                lines = []
                for turn in turns:
                    for msg in turn:
                        role = msg.get("role", "unknown")
                        text = msg.get("content", {}).get("text", str(msg.get("content", "")))
                        lines.append(f"{role}: {text}")
                if lines:
                    event.agent.system_prompt += (
                        "\n\nRecent conversation history:\n" + "\n".join(lines)
                    )
        except Exception as e:
            logger.warning(f"Could not load memory turns: {e}")

        # Long-term
        try:
            namespaces = [
                f"/cs_agent/{self.actor_id}/preferences/",
                f"/cs_agent/{self.actor_id}/facts/",
            ]
            bits = []
            for ns in namespaces:
                try:
                    records = self.mem_client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=ns,
                        query="name preference communication style customer facts",
                        top_k=5,
                    )
                    for r in records or []:
                        if isinstance(r, dict):
                            text = (
                                r.get("content", {}).get("text")
                                or r.get("text")
                                or r.get("memoryRecord", {}).get("content", {}).get("text")
                            )
                            if text:
                                bits.append(text)
                except Exception as e:
                    logger.warning(f"Could not retrieve from {ns}: {e}")

            if bits:
                event.agent.system_prompt += (
                    "\n\nKnown facts and preferences about this customer:\n"
                    + "\n".join(f"- {b}" for b in bits)
                )
        except Exception as e:
            logger.warning(f"Could not load long-term memories: {e}")

    def on_message_added(self, event: MessageAddedEvent):
        try:
            messages = event.agent.messages
            if not messages:
                return

            last = messages[-1]
            logger.warning(f"Last message raw: {last}")
            logger.warning(f"Last message type: {type(last)}")

            # Handle both dict and object-style messages
            if not isinstance(last, dict):
                last = getattr(last, "__dict__", None) or {}
                if hasattr(messages[-1], "role"):
                    last = {
                        "role": getattr(messages[-1], "role", None),
                        "content": getattr(messages[-1], "content", ""),
                    }

            role = safe_role(last)
            text = extract_text(last)
            if not text:
                return

            # CRITICAL: use tuple format — dict nested content caused Invalid role 'content'
            self.mem_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(text, role)],
            )
        except Exception as e:
            logger.warning(f"Could not save message to memory: {e}")

    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(MessageAddedEvent, self.on_message_added)

# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."

    resp = _bedrock_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
    )

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    texts = [
        result["content"]["text"]
        for result in results
        if result.get("content") and result["content"].get("text")
    ]
    return "\n---\n".join(texts)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f'''
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

point_value = 0.01
max_redeem_value = order_total * 0.5
max_points_by_value = int(max_redeem_value / point_value)

points_available = (loyalty_points // 500) * 500
max_redeemable = (max_points_by_value // 500) * 500
points_redeemed = min(points_available, max_redeemable)

points_discount = points_redeemed * point_value
subtotal_after_points = order_total - points_discount

tier_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * tier_rate

final_total = subtotal_after_points - tier_discount
total_savings = points_discount + tier_discount

points_earned = int(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount": round(tier_discount, 2),
    "tier_discount_pct": tier_rate * 100,
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
    "tier": tier,
    "product_category": product_category
}}
print(json.dumps(result))
'''

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response["stream"]:
                return json.dumps(event["result"])

    except Exception as e:
        logger.exception("Code Interpreter execution failed")
        raise

        


# System prompt used by the agent
# System prompt used by the agent
system_prompt = """
You are an Amazon Customer Support AI Agent with a working browser tool.

Browser rules (must follow):
1. ALWAYS call init_session first with a session_name.
2. Wait for init_session to succeed before navigate.
3. Never call init_session and navigate in the same step.
4. Then navigate to the URL.
5. Then get the page title (get_text on selector "title", or evaluate document.title).
6. If navigate times out once, retry navigate once on the same session.
7. Never say the browser is unavailable.

Order tracking rules (VERY IMPORTANT):
- When tracking an order, use the tool named orderTrackerGetOrder___get_order
- Pass ONLY the parameter "order_id" (example: {"order_id": "ORD-001"})
- NEVER pass "basePath", "url", or any other parameter to this tool
- Do not invent extra parameters

Other tools: search_knowledge_base, calculate_loyalty_discount, gateway order tools.
"""


# ── Helper: create MCP transport ──────────────────────────────────────────────
def create_streamable_http_transport():
    """Create the transport for the AgentCore Gateway."""
    return streamable_http_client(GATEWAY_URL)


# ── Helper: sanitize tool names ───────────────────────────────────────────────
def sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        # 1. Extract inputs
        user_input = payload.get("prompt") or payload.get("user_input") or ""
        actor_id = payload.get("customer_id") or payload.get("actor_id") or "default_customer"
        session_id = payload.get("session_id") or str(uuid.uuid4())

        logger.info(f"Invoke | actor={actor_id} | session={session_id}")

        # 2. Memory hook
        memory_hook = MemoryHook(
            mem_client=memory_client,
            memory_id=MEMORY_ID,
            actor_id=actor_id,
            session_id=session_id,
        )

        # 3. Browser tool
        
                # 3. Browser tool
                # 3. Browser tool
        browser_tool = None
        try:
            agent_core_browser = AgentCoreBrowser(region=REGION,
            session_timeout=3600,  # 1 hour (default may be shorter in practice)
        )
            browser_tool = agent_core_browser.browser
            logger.warning(f"Browser tool loaded successfully: {type(browser_tool)}")
        except Exception as e:
            logger.warning(f"Browser tool init failed: {e}")

        # 4. Base tools
        base_tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
        ]
        if browser_tool is not None:
            base_tools.append(browser_tool)

        logger.warning(
            "Base tools: %s",
            [getattr(t, "name", str(t)) for t in base_tools],
        )

        # 5. tools always defined
        tools = base_tools

        try:
            with MCPClient(create_streamable_http_transport) as mcp_client:
                gateway_tools = mcp_client.list_tools_sync()

                logger.warning(
                    "Gateway tools loaded: %s",
                    [getattr(t, "name", str(t)) for t in gateway_tools],
                )

                for t in gateway_tools:
                    if hasattr(t, "name"):
                        t.name = sanitize_tool_name(t.name)

                tools = base_tools + list(gateway_tools)

                agent = Agent(
                    model=model,
                    system_prompt=system_prompt,
                    tools=tools,
                    hooks=[memory_hook],
                )
                response = agent(user_input)

        except Exception as gateway_err:
            logger.warning(f"Gateway / agent error: {gateway_err}")
            agent = Agent(
                model=model,
                system_prompt=system_prompt,
                tools=tools,
                hooks=[memory_hook],
            )
            response = agent(user_input)

        # 6. Extract clean text from the response
        if hasattr(response, "message") and response.message:
            content = response.message.get("content", response.message)
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    return first.get("text", str(first))
                return str(first)
            return str(content)

        return str(response)

    except Exception as e:
        logger.exception("Error in invoke")
        return f"An error occurred while processing your request: {str(e)}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload), {}))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()