from graph.state import SupportState
from langchain_core.messages import AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from core.config import settings

def get_llm():
    """NVIDIA Llama 3.2 via native ChatNVIDIA API."""
    return ChatNVIDIA(
        model="meta/llama-3.2-11b-vision-instruct",
        api_key=settings.NVIDIA_API_KEY,
        temperature=0.3,
        max_tokens=512
    )

def get_rag_context(query: str) -> str:
    """Retrieve highly relevant context using Enterprise Hybrid RAG."""
    try:
        from rag.retrieval import get_enterprise_context
        context = get_enterprise_context(query, top_k=3)
        if context:
            print("Enterprise RAG: Successfully retrieved and reranked context.")
            return context
    except Exception as e:
        print(f"Enterprise RAG search failed: {e}")
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
Note: For general greetings (hi, hello) or questions about identity (who are you), classify as 'faq'.
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
    """Handle billing questions — strictly uses RAG context from uploaded documents."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            if rag_context:
                prompt = f"""You are a billing support agent for TechMart e-commerce.
Answer the customer's question STRICTLY and ONLY using the document context provided below.
Do NOT use any outside knowledge. If the answer is not in the context, say:
"I don't have information about that in our current knowledge base."

DOCUMENT CONTEXT:
{rag_context}

Customer question: {query}
Provide a helpful, concise response based ONLY on the above context."""
            else:
                prompt = f"""You are a billing support agent for TechMart e-commerce.
You do not have any specific document context for this question.
Politely inform the customer: "I don't have specific information about that in our knowledge base right now. 
Please contact our billing team directly for assistance."

Customer question: {query}"""
            response = llm.invoke(prompt)
            return {"messages": [response], "agent_outputs": {"billing_agent": "processed"}}
        except Exception as e:
            print(f"Billing agent LLM error: {e}")

    return {"messages": [AIMessage(content="I understand you have a billing question. Our billing team will assist you shortly.")], "agent_outputs": {"billing_agent": "processed"}}

def technical_agent(state: SupportState) -> dict:
    """Handle technical issues — strictly uses RAG context from uploaded documents."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            if rag_context:
                prompt = f"""You are a technical support agent for TechMart e-commerce.
Answer the customer's question STRICTLY and ONLY using the document context provided below.
Do NOT use any outside knowledge. If the answer is not in the context, say:
"I don't have technical documentation for that issue in our knowledge base."

DOCUMENT CONTEXT:
{rag_context}

Customer issue: {query}
Provide step-by-step help based ONLY on the above context."""
            else:
                prompt = f"""You are a technical support agent for TechMart e-commerce.
You do not have specific documentation for this issue in the knowledge base.
Politely inform the customer that you'll need to escalate or check the manuals.

Customer issue: {query}"""
            response = llm.invoke(prompt)
            return {"messages": [response], "agent_outputs": {"technical_agent": "processed"}}
        except Exception as e:
            print(f"Technical agent LLM error: {e}")

    return {"messages": [AIMessage(content="I can help with this technical issue. Please share more details.")], "agent_outputs": {"technical_agent": "processed"}}

def faq_agent(state: SupportState) -> dict:
    """Answer questions — strictly uses RAG context from uploaded documents."""
    query = state["messages"][-1].content
    rag_context = get_rag_context(query)

    if settings.NVIDIA_API_KEY:
        try:
            llm = get_llm()
            if rag_context:
                prompt = f"""You are a helpful and polite AI Support Agent for TechMart e-commerce.
If the user asks who you are, introduce yourself as the TechMart AI Support Agent, here to help with their uploaded documents.

CRITICAL RULES:
1. Answer the customer's question STRICTLY and ONLY using the document context provided below.
2. Do NOT use any general knowledge or outside information.
3. You MUST include inline citations for every fact using the exact citation block provided in the context. Example: (Citation: doc.pdf | Category: general | v1.0).
4. If the specific answer is not found in the context (e.g., questions about Narendra Modi, general world facts), say exactly:
"I am an AI assistant specifically trained on TechMart's uploaded documents. I'm sorry, but I don't have information about that in my knowledge base. Please ask me a query related to the uploaded PDFs so I can provide you with an accurate answer!"

DOCUMENT CONTEXT:
{rag_context}

Customer question: {query}
Give a clear, human-like, and accurate answer using ONLY the above document context."""
            else:
                prompt = f"""You are a helpful and polite AI Support Agent for TechMart e-commerce.
If the user asks who you are or says hello, introduce yourself in a friendly, human-like way (e.g., "Hello! I am the TechMart AI Support Agent. I'm here to answer questions based on the documents you've uploaded.").

If they ask a specific question, since there is no document context currently matching it, say:
"I am an AI assistant specifically trained on the uploaded documents. I'm sorry, but I don't have information about that in my current knowledge base. Please ask me a query related to the uploaded PDFs so I can provide you with an accurate answer!"

Do NOT answer from general knowledge.
Customer question: {query}"""
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
