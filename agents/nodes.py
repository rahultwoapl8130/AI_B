from graph.state import SupportState
from langchain_core.messages import AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from core.config import settings

def get_llm():
    """NVIDIA Llama 3.1 via native ChatNVIDIA API."""
    return ChatNVIDIA(
        model="meta/llama-3.2-11b-vision-instruct",
        api_key=settings.NVIDIA_API_KEY,
        temperature=0.3,
        max_tokens=512
    )

def get_rag_context(query: str) -> str:
    """Retrieve relevant context from MongoDB knowledge base."""
    try:
        from rag.ingestion import search_knowledge_base
        docs = search_knowledge_base(query, top_k=3)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            return f"\n\n[Knowledge Base Context]:\n{context}"
    except Exception as e:
        print(f"RAG search failed: {e}")
    return ""

def intent_agent(state: SupportState) -> dict:
    """Classify user intent."""
    last_message = state["messages"][-1].content

    if not settings.NVIDIA_API_KEY:
        intent = "faq"
        if any(w in last_message.lower() for w in ["refund", "billing", "payment", "invoice"]):
            intent = "billing"
        elif any(w in last_message.lower() for w in ["error", "not working", "bug", "crash"]):
            intent = "technical"
        return {"intent": intent, "agent_outputs": {}}

    try:
        llm = get_llm()
        prompt = f"""Classify this customer message into exactly one category.
Categories: billing, technical, complaint, faq
Message: '{last_message}'
Reply with ONE word only."""
        response = llm.invoke(prompt)
        raw = response.content.strip().lower()
        if "billing" in raw: intent = "billing"
        elif "technical" in raw: intent = "technical"
        elif "complaint" in raw: intent = "complaint"
        else: intent = "faq"
    except Exception:
        intent = "faq"

    return {"intent": intent, "agent_outputs": {}}

def sentiment_agent(state: SupportState) -> dict:
    """Analyze message sentiment."""
    msg = state["messages"][-1].content.lower()
    sentiment = "negative" if any(w in msg for w in ["angry", "terrible", "awful", "hate", "worst", "frustrated"]) else "neutral"
    return {"sentiment": sentiment}

def priority_agent(state: SupportState) -> dict:
    """Determine ticket priority."""
    priority = "Normal"
    if state.get("sentiment") == "negative" or state.get("intent") == "complaint":
        priority = "High"
    if "urgent" in state["messages"][-1].content.lower():
        priority = "Urgent"
    return {"priority": priority}

def billing_agent(state: SupportState) -> dict:
    """Handle billing questions with RAG context."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            prompt = f"""You are a billing support agent for TechMart e-commerce.
Use the knowledge base context below to answer accurately.{rag_context}

Customer question: {query}
Provide a helpful, concise response."""
            response = llm.invoke(prompt)
            return {"messages": [response], "agent_outputs": {"billing_agent": "processed"}}
        except Exception as e:
            print(f"Billing agent LLM error: {e}")

    return {"messages": [AIMessage(content="I understand you have a billing question. Our billing team will assist you shortly.")], "agent_outputs": {"billing_agent": "processed"}}

def technical_agent(state: SupportState) -> dict:
    """Handle technical issues with RAG context."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            prompt = f"""You are a technical support agent for TechMart e-commerce.
Use the knowledge base context below to troubleshoot accurately.{rag_context}

Customer issue: {query}
Provide step-by-step troubleshooting help."""
            response = llm.invoke(prompt)
            return {"messages": [response], "agent_outputs": {"technical_agent": "processed"}}
        except Exception as e:
            print(f"Technical agent LLM error: {e}")

    return {"messages": [AIMessage(content="I can help with this technical issue. Please share more details.")], "agent_outputs": {"technical_agent": "processed"}}

def faq_agent(state: SupportState) -> dict:
    """Answer general questions using RAG context."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            prompt = f"""You are a helpful customer support agent for TechMart e-commerce.
Use the knowledge base context below to answer accurately.{rag_context}

Customer question: {query}
Give a clear, friendly answer."""
            response = llm.invoke(prompt)
            return {"messages": [response], "agent_outputs": {"faq_agent": "processed"}}
        except Exception as e:
            error_msg = f"LLM Error: {str(e)}"
            print(error_msg)
            return {"messages": [AIMessage(content=error_msg)], "agent_outputs": {"faq_agent": "processed"}}

    return {"messages": [AIMessage(content="API Key not found. Thank you for contacting TechMart Support!")], "agent_outputs": {"faq_agent": "processed"}}

def escalation_agent(state: SupportState) -> dict:
    """Escalate to human agent."""
    return {
        "messages": [AIMessage(content="I understand your concern and I'm escalating this to a senior agent. You will be contacted within 2 hours.")],
        "escalation_required": True
    }
