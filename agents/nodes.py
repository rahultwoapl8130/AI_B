from graph.state import SupportState
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from core.config import settings

# Initialize NVIDIA Llama 3 Model via OpenAI-compatible API
def get_llm():
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.NVIDIA_API_KEY,
        model="meta/llama-3.1-70b-instruct",
        temperature=0.3
    )

def intent_agent(state: SupportState) -> dict:
    """Analyzes the user's intent."""
    last_message = state["messages"][-1].content
    
    # If no API key, fallback to simple logic
    if not settings.NVIDIA_API_KEY:
        intent = "faq"
        if "refund" in last_message.lower(): intent = "billing"
        elif "error" in last_message.lower(): intent = "technical"
        return {"intent": intent}
        
    llm = get_llm()
    prompt = f"Analyze this customer message and classify its intent as ONLY ONE of these words: [billing, technical, complaint, faq]. Message: '{last_message}'"
    
    response = llm.invoke(prompt)
    intent = response.content.strip().lower()
    
    # Clean up response
    if "billing" in intent: intent = "billing"
    elif "technical" in intent: intent = "technical"
    elif "complaint" in intent: intent = "complaint"
    else: intent = "faq"
    
    return {"intent": intent}

def sentiment_agent(state: SupportState) -> dict:
    """Analyzes the sentiment of the message."""
    last_message = state["messages"][-1].content.lower()
    sentiment = "neutral"
    if "angry" in last_message or "terrible" in last_message:
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

# Domain Agents (These now use the LLM to generate replies)
def billing_agent(state: SupportState) -> dict:
    if settings.NVIDIA_API_KEY:
        llm = get_llm()
        response = llm.invoke(f"You are a billing support agent. Reply politely to: {state['messages'][-1].content}")
        return {"messages": [response], "agent_outputs": {"billing_agent": "processed"}}
    
    return {"messages": [AIMessage(content="I understand you have a billing question. Let me check your account.")], "agent_outputs": {"billing_agent": "processed"}}

def technical_agent(state: SupportState) -> dict:
    if settings.NVIDIA_API_KEY:
        llm = get_llm()
        response = llm.invoke(f"You are a technical support agent. Help troubleshoot: {state['messages'][-1].content}")
        return {"messages": [response], "agent_outputs": {"technical_agent": "processed"}}
        
    return {"messages": [AIMessage(content="I can help you with this technical issue. Could you share the error code?")], "agent_outputs": {"technical_agent": "processed"}}

def faq_agent(state: SupportState) -> dict:
    if settings.NVIDIA_API_KEY:
        llm = get_llm()
        response = llm.invoke(f"You are a general FAQ agent. Answer this question: {state['messages'][-1].content}")
        return {"messages": [response], "agent_outputs": {"faq_agent": "processed"}}
        
    return {"messages": [AIMessage(content="Here is the information from our FAQ section.")], "agent_outputs": {"faq_agent": "processed"}}

def escalation_agent(state: SupportState) -> dict:
    """Escalates to a human."""
    response = AIMessage(content="I am escalating this to a human agent who will be with you shortly.")
    return {"messages": [response], "escalation_required": True}
