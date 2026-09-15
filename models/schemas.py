from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class ChatMessageCreate(BaseModel):
    session_id: str
    message: str

class ChatMessageResponse(BaseModel):
    message_id: str
    sender: str
    agent_type: str
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = []
