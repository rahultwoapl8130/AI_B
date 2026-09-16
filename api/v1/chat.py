from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph.workflow import support_graph
import uuid

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    intent: str
    priority: str
    routed_to: str
    escalated: bool

@router.post("/chat", response_model=ChatResponse, summary="Test the LangGraph Orchestrator")
async def chat_endpoint(request: ChatRequest):
    """
    Sends a message to the LangGraph Multi-Agent system powered by NVIDIA Llama 3.
    Returns the AI's reply and the internal state (intent, routing, priority).
    """
    # 1. Initialize State
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "ticket_id": str(uuid.uuid4())[:8],
        "escalation_required": False
    }
    
    # 2. Run the graph
    final_state = support_graph.invoke(initial_state)
    
    # 3. Extract the final reply from the messages list
    # The last message is usually the AI's reply
    ai_reply = final_state["messages"][-1].content
    
    # 4. Extract routing information
    intent = final_state.get("intent", "unknown")
    priority = final_state.get("priority", "Normal")
    escalated = final_state.get("escalation_required", False)
    
    # Determine who handled it by looking at agent outputs
    routed_to = "faq_agent"
    if escalated:
        routed_to = "human_escalation"
    elif intent == "billing":
        routed_to = "billing_agent"
    elif intent == "technical":
        routed_to = "technical_agent"
        
    return ChatResponse(
        reply=ai_reply,
        intent=intent,
        priority=priority,
        routed_to=routed_to,
        escalated=escalated
    )
