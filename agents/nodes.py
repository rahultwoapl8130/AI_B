from graph.state import SupportState
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from core.config import settings

def get_llm():
    """Initialize NVIDIA Llama 3 via OpenAI-compatible API."""
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.NVIDIA_API_KEY,
        model="meta/llama-3.1-70b-instruct",
        temperature=0.3
    )

def intent_agent(state: SupportState) -> dict:
    """Analyzes the user's intent."""
    last_message = state["messages"][-1].content
    
    if not settings.NVIDIA_API_KEY:
        intent = "faq"
        if "refund" in last_message.lower() or "billing" in last_message.lower():
            intent = "billing"
        elif "error" in last_message.lower() or "not working" in last_message.lower():
            intent = "technical"
        return {"intent": intent, "agent_outputs": {}}

    try:
        llm = get_llm()
        prompt = f"Classify this customer message into EXACTLY one word: billing, technical, complaint, or faq. Message: '{last_message}'. Reply with only one word."
        response = llm.invoke(prompt)
        intent = response.content.strip().lower()
        if "billing" in intent: intent = "billing"
        elif "technical" in intent: intent = "technical"
        elif "complaint" in intent: intent = "complaint"
        else: intent = "faq"
    except Exception:
        intent = "faq"
    
    return {"intent": intent, "agent_outputs": {}}

def sentiment_agent(state: SupportState) -> dict:
    """Analyzes sentiment."""
    last_message = state["messages"][-1].content.lower()
    sentiment = "neutral"
    if any(w in last_message for w in ["angry", "terrible", "awful", "hate", "worst"]):
        sentiment = "negative"
    return {"sentiment": sentiment}

def priority_agent(state: SupportState) -> dict:
    """Determines priority."""
    priority = "Normal"
    if state.get("sentiment") == "negative" or state.get("intent") == "complaint":
        priority = "High"
    if "urgent" in state["messages"][-1].content.lower():
        priority = "Urgent"
    return {"priority": priority}

def billing_agent(state: SupportState) -> dict:
    """Handles billing questions."""
    last_message = state["messages"][-1].content
    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            response = llm.invoke(f"You are a friendly billing support agent for TechMart. Help this customer: {last_message}")
            return {"messages": [response], "agent_outputs": {"billing_agent": "processed"}}
        except Exception as e:
            pass
    return {"messages": [AIMessage(content="I understand you have a billing question. Our billing team will assist you shortly.")], "agent_outputs": {"billing_agent": "processed"}}

def technical_agent(state: SupportState) -> dict:
    """Handles technical issues."""
    last_message = state["messages"][-1].content
    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            response = llm.invoke(f"You are a technical support agent for TechMart. Help troubleshoot: {last_message}")
            return {"messages": [response], "agent_outputs": {"technical_agent": "processed"}}
        except Exception as e:
            pass
    return {"messages": [AIMessage(content="I can help you with this technical issue. Could you share more details?")], "agent_outputs": {"technical_agent": "processed"}}

def faq_agent(state: SupportState) -> dict:
    """Handles general FAQ questions."""
    last_message = state["messages"][-1].content
    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            response = llm.invoke(f"You are a helpful customer support agent for TechMart, an e-commerce platform. Answer this question helpfully: {last_message}")
            return {"messages": [response], "agent_outputs": {"faq_agent": "processed"}}
        except Exception as e:
            pass
    return {"messages": [AIMessage(content="Thank you for contacting TechMart support! How can I help you today?")], "agent_outputs": {"faq_agent": "processed"}}

def escalation_agent(state: SupportState) -> dict:
    """Escalates to human."""
    response = AIMessage(content="I understand your concern. I am escalating this to a human agent who will be with you within 24 hours.")
    return {"messages": [response], "escalation_required": True}
