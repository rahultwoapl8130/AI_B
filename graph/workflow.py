from langgraph.graph import StateGraph, END
from graph.state import SupportState
from agents.nodes import (
    intent_agent, 
    sentiment_agent, 
    priority_agent,
    billing_agent,
    technical_agent,
    faq_agent,
    escalation_agent
)
from graph.edges import route_intent

def build_workflow() -> StateGraph:
    """
    Builds the LangGraph orchestrator workflow for the Support Multi-Agent system.
    """
    workflow = StateGraph(SupportState)

    # 1. Add Nodes (Agents)
    workflow.add_node("intent_agent", intent_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("priority_agent", priority_agent)
    
    workflow.add_node("billing_agent", billing_agent)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("faq_agent", faq_agent)
    workflow.add_node("escalation_agent", escalation_agent)

    # 2. Define Edges and Routing
    # Set the entrypoint to the analyzer agents
    workflow.set_entry_point("intent_agent")
    
    # In LangGraph, we can run nodes sequentially. 
    # For now, intent -> sentiment -> priority -> conditional router
    workflow.add_edge("intent_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "priority_agent")
    
    # Conditional routing after priority analysis
    workflow.add_conditional_edges(
        "priority_agent",
        route_intent,
        {
            "billing_agent": "billing_agent",
            "technical_agent": "technical_agent",
            "faq_agent": "faq_agent",
            "escalation_agent": "escalation_agent"
        }
    )

    # End the graph after a domain agent responds
    workflow.add_edge("billing_agent", END)
    workflow.add_edge("technical_agent", END)
    workflow.add_edge("faq_agent", END)
    workflow.add_edge("escalation_agent", END)

    return workflow.compile()

# Compile the graph
support_graph = build_workflow()
