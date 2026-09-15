from fastapi import APIRouter, HTTPException, Depends
from models.schemas import ChatMessageCreate, ChatMessageResponse
from database.mongo import sessions_collection, messages_collection
from datetime import datetime
from bson import ObjectId

router = APIRouter()

@router.post("/send", response_model=ChatMessageResponse)
async def send_message(chat_in: ChatMessageCreate):
    # Simulated orchestrator logic
    lower_input = chat_in.message.lower()
    
    agent_type = "router"
    response_text = ""
    
    if "refund" in lower_input or "billing" in lower_input:
        agent_type = "billing"
        response_text = "I can help you with your payment and refund inquiries. What is your order number?"
    elif "broken" in lower_input or "battery" in lower_input:
        agent_type = "technical"
        response_text = "I see you need technical assistance. Based on the User Manual, please try restarting."
    else:
        agent_type = "faq"
        response_text = "We offer standard shipping in 3-5 business days. Can I help you with anything else?"
    
    # In a real scenario, we'd save both user message and agent message to MongoDB here
    
    return ChatMessageResponse(
        message_id=str(ObjectId()),
        sender="agent",
        agent_type=agent_type,
        content=response_text,
        timestamp=datetime.utcnow()
    )

@router.post("/session")
async def create_session():
    # Simulated session creation
    session_id = str(ObjectId())
    return {"session_id": session_id, "created_at": datetime.utcnow()}
