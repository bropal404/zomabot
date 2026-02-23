import operator
import os
from typing import Annotated, Sequence, TypedDict, Union

import requests
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# --- 2. DEFINE TOOLS ---


@tool
def process_refund(item_name: str, reason: str, refund_amount_percentage: int = 100):
    """
    Issue a refund for a specific item.
    Args:
        item_name: The name of the item (e.g., "Chicken Pizza").
        reason: Why the refund is needed (e.g., "Burnt", "Missing", "Spilled").
        refund_amount_percentage: 100 for full refund, 50 for partial.
    """
    # Logic: In a real app, this hits the Stripe/Razorpay API
    return f"SUCCESS: Refund processed for '{item_name}'. Reason: {reason}. Amount: {refund_amount_percentage}% credited to wallet."


@tool
def contact_delivery_partner(message: str):
    """
    Send a message to the delivery partner/rider.
    Use this for location issues or 'Where are you?' queries when status is 'Out for Delivery'.
    """
    return f"RIDER_ALERT: Message sent to rider -> '{message}'. Rider will call customer shortly."


@tool
def escalate_to_support_admin(issue_summary: str, urgency: str):
    """
    Escalate to a human admin via Telegram.
    Use ONLY for: Severe safety issues, App crashes, or when the user is extremely abusive.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    alert_text = f"🚨 **ADMIN ALERT** 🚨\nUrgency: {urgency}\nIssue: {issue_summary}"

    if token and chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": alert_text},
            )
            return "ESCALATED: Admin has been notified via Telegram channel."
        except Exception as e:
            return f"ESCALATION FAILED: {str(e)}"

    return "ESCALATED: Admin notified (Simulation Mode - No Token Found)."


tools = [process_refund, contact_delivery_partner, escalate_to_support_admin]

# --- 3. MODEL SETUP ---
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- 4. NODES ---


def agent_node(state: AgentState):
    messages = state["messages"]
    # We invoke the LLM with the current conversation history
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# --- 5. CONDITIONAL LOGIC ---


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- 6. GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Compile the graph
agent_app = workflow.compile()
