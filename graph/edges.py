from graph.state import SupportState

def route_intent(state: SupportState) -> str:
    """
    Conditional routing logic based on the intent analyzed by the Intent Agent.
    """
    intent = state.get("intent")
    sentiment = state.get("sentiment")
    
    # Escalation Policy
    if sentiment == "negative" or state.get("escalation_required"):
        return "escalation_agent"
        
    if intent == "billing":
        return "billing_agent"
    elif intent == "technical":
        return "technical_agent"
    elif intent == "complaint":
        return "escalation_agent"
    else:
        return "faq_agent"
