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

def build_workflow():
    workflow = StateGraph(SupportState)

    workflow.add_node("intent_agent", intent_agent)
    workflow.add_node("sentiment_agent", sentiment_agent)
    workflow.add_node("priority_agent", priority_agent)
    workflow.add_node("billing_agent", billing_agent)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("faq_agent", faq_agent)
    workflow.add_node("escalation_agent", escalation_agent)

    workflow.set_entry_point("intent_agent")
    workflow.add_edge("intent_agent", "sentiment_agent")
    workflow.add_edge("sentiment_agent", "priority_agent")
    
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

    workflow.add_edge("billing_agent", END)
    workflow.add_edge("technical_agent", END)
    workflow.add_edge("faq_agent", END)
    workflow.add_edge("escalation_agent", END)

    return workflow.compile()

support_graph = build_workflow()
