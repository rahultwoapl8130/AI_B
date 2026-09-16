from typing import TypedDict, List, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator

class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    ticket_id: str
    intent: Optional[str]
    sentiment: Optional[str]
    priority: Optional[str]
    escalation_required: bool
    agent_outputs: dict
