from graph.state import SupportState
from langchain_core.messages import AIMessage, HumanMessage

# In a real app, these nodes would call LLMs (e.g. ChatOpenAI).
# For now, we stub the orchestrator and agent logic.

def intent_agent(state: SupportState) -> dict:
    """Analyzes the user's intent."""
    last_message = state["messages"][-1].content.lower()
    intent = "faq"
    if "refund" in last_message or "cancel" in last_message:
        intent = "billing"
    elif "broken" in last_message or "error" in last_message:
        intent = "technical"
    elif "complain" in last_message or "angry" in last_message:
        intent = "complaint"
    
    return {"intent": intent}

def sentiment_agent(state: SupportState) -> dict:
    """Analyzes the sentiment of the message."""
    last_message = state["messages"][-1].content.lower()
    sentiment = "neutral"
    if "angry" in last_message or "terrible" in last_message or "hate" in last_message:
        sentiment = "negative"
    return {"sentiment": sentiment}

def priority_agent(state: SupportState) -> dict:
    """Determines ticket priority based on intent and sentiment."""
    priority = "Normal"
    if state.get("sentiment") == "negative" or state.get("intent") == "complaint":
        priority = "High"
    if "urgent" in state["messages"][-1].content.lower():
        priority = "Urgent"
    return {"priority": priority}

# Domain Agents
def billing_agent(state: SupportState) -> dict:
    response = AIMessage(content="I understand you have a billing question. Let me check your account.")
    return {"messages": [response], "agent_outputs": {"billing_agent": "processed"}}

def technical_agent(state: SupportState) -> dict:
    response = AIMessage(content="I can help you with this technical issue. Could you share the error code?")
    return {"messages": [response], "agent_outputs": {"technical_agent": "processed"}}

def faq_agent(state: SupportState) -> dict:
    response = AIMessage(content="Here is the information from our FAQ section.")
    return {"messages": [response], "agent_outputs": {"faq_agent": "processed"}}

def escalation_agent(state: SupportState) -> dict:
    """Escalates to a human."""
    response = AIMessage(content="I am escalating this to a human agent who will be with you shortly.")
    return {"messages": [response], "escalation_required": True}
