from __future__ import annotations
import asyncio, logging
from typing import Annotated, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from api.search import hybrid_search
from api.cart import create_checkout_session, complete_checkout
from shared.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    messages:     Annotated[list, add_messages]
    session_id:   str
    last_results: list[dict]   # products from last search, kept for follow-ups
    last_query:   Optional[str]


# ── In-memory session store (replace with Redis for production) ────────────────
_sessions: dict[str, ChatState] = {}


# ── LLM factory ───────────────────────────────────────────────────────────────
def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.3,
        google_api_key=settings.google_api_key,
    )


SYSTEM_PROMPT = """You are a helpful Würth product search assistant.
You help users find products, compare items, and complete purchases.

You have access to two tools:
- ucp_add_to_cart: call this when a user wants to add a specific product to their cart.
- ucp_checkout:    call this when the user wants to pay / complete their order.

When a user asks to search for products, present the results clearly and ask if they
want to add any to their cart. When a user says things like "add the first one" or
"buy it", call ucp_add_to_cart with the correct product details.
Keep responses concise and product-focused."""

SEARCH_KEYWORDS = frozenset([
    "find", "search", "show", "looking for",
    "need", "want", "buy", "suggest", "recommend",
])


# ── UCP Tools ─────────────────────────────────────────────────────────────────
@tool
def ucp_add_to_cart(product_id: str, title: str, price_cents: int) -> str:
    """Add a product to a UCP checkout session.
    price_cents is the price in cents (e.g. 1999 = €19.99).
    """
    session = create_checkout_session(product_id, title, price_cents)
    total = sum(li.item.price * li.quantity for li in session.line_items) / 100

    return (
        f"✅ Added '{title}' to cart.\n"
        f"UCP Session ID: {session.id}\n"
        f"Cart total: €{total:.2f}"
    )


@tool
def ucp_checkout(session_id: str) -> str:
    """Complete a UCP checkout session and place the order."""
    order = complete_checkout(session_id)
    return (
        f"🎉 Order confirmed!\n"
        f"Order ID : {order.id}\n"
        f"Status   : {order.status}\n"
        f"Total    : €{order.total:.2f}"
    )


TOOLS = [ucp_add_to_cart, ucp_checkout]
_tool_node = ToolNode(TOOLS)


# ── Chat node ─────────────────────────────────────────────────────────────────
async def chat_node(
    state: ChatState,
    client: OpenSearch,
    model: SentenceTransformer,
) -> ChatState:
    llm = _get_llm().bind_tools(TOOLS)

    # Most recent human message
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    search_results = list(state.get("last_results", []))
    extra_context: list = []

    # ── Auto-trigger hybrid search on intent keywords ──────────────────────────
    if any(kw in last_human.lower() for kw in SEARCH_KEYWORDS):
        try:
            vector = await asyncio.to_thread(
                lambda: model.encode(last_human, normalize_embeddings=True).tolist()
            )
            result = await hybrid_search(
                client=client,
                query=last_human,
                query_vector=vector,
                index_name=settings.index_name,
                size=5,
            )
            search_results = [h.model_dump() for h in result.hits]

            if search_results:
                rows = "\n".join(
                    f"{i+1}. [{h['id']}] {h['title']} "
                    f"— €{h['price'] or 'N/A'} "
                    f"(⭐ {h['average_rating'] or 'N/A'})"
                    for i, h in enumerate(search_results)
                )
                # Inject search results as a system context message (not shown to user)
                extra_context = [
                    HumanMessage(
                        content=(
                            f"[SYSTEM: search results for «{last_human}»]\n"
                            f"{rows}\n"
                            f"Present these results to the user and ask if they want "
                            f"to add any to their cart. Use the product id and title "
                            f"when calling ucp_add_to_cart."
                        )
                    )
                ]
        except Exception as exc:
            logger.warning("Hybrid search failed inside chatbot: %s", exc)

    # ── Invoke LLM ────────────────────────────────────────────────────────────
    full_messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + list(state["messages"])
        + extra_context
    )
    response = await asyncio.to_thread(llm.invoke, full_messages)

    return {
        **state,
        "messages": [response],                       # add_messages appends automatically
        "last_results": search_results,
        "last_query": last_human if search_results else state.get("last_query"),
    }


# ── Routing: go to tools if LLM made a tool call, otherwise end ───────────────
def _route(state: ChatState) -> Literal["tools", "end"]:
    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ── Graph factory (rebuilt per request — cheap, stateless compile) ─────────────
def _build_graph(client: OpenSearch, model: SentenceTransformer):
    async def _chat(state): return await chat_node(state, client, model)

    g = StateGraph(ChatState)
    g.add_node("chat", _chat)
    g.add_node("tools", _tool_node)

    g.set_entry_point("chat")
    g.add_conditional_edges("chat", _route, {"tools": "tools", "end": END})
    g.add_edge("tools", "chat")   # after tool execution → back to chat to process result

    return g.compile()


# ── Public API called by api/main.py ──────────────────────────────────────────
async def chat_turn(
    session_id: str,
    user_message: str,
    client: OpenSearch,
    model: SentenceTransformer,
) -> dict:
    """
    Process one user turn and return the assistant reply + any search results.
    Maintains multi-turn memory via _sessions dict (keyed by session_id).
    """
    if session_id not in _sessions:
        _sessions[session_id] = {
            "messages":     [],
            "session_id":   session_id,
            "last_results": [],
            "last_query":   None,
        }

    # Append new human message to existing history
    state = _sessions[session_id]
    state["messages"] = list(state["messages"]) + [HumanMessage(content=user_message)]

    graph = _build_graph(client, model)
    new_state = await graph.ainvoke(state)
    _sessions[session_id] = new_state

    # Pick the last non-tool-call AI message as the visible reply
    last_ai_text = next(
        (
            m.content
            for m in reversed(new_state["messages"])
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        ),
        "",
    )

    return {
        "session_id": session_id,
        "reply":      last_ai_text,
        "results":    new_state.get("last_results", []),
    }