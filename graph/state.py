from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class SupportState(TypedDict):
    """
    Represents the state of the LangGraph multi-agent workflow.
    """
    messages: List[BaseMessage]
    ticket_id: Optional[str]
    
    # Analyzed features
    intent: Optional[str]
    sentiment: Optional[str]
    priority: Optional[str]
    
    # Routing and execution
    routed_agent: Optional[str]
    escalation_required: bool
    
    # Outputs and Memory
    agent_outputs: Dict[str, Any]
    tool_results: List[Dict[str, Any]]
    
    # Telemetry
    latency_ms: Optional[int]
    errors: List[str]
