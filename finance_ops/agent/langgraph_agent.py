import json
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from finance_ops.core.models import DecisionLabel, ReasonCode, AgentRecommendation

class InvestigationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    case_id: str
    error: Optional[str]

def create_agent_graph(llm: ChatGoogleGenerativeAI, tools: list):
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: InvestigationState):
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            return {"error": str(e)}
            
    def should_continue(state: InvestigationState):
        if state.get("error"):
            return END
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM makes a tool call, route to tools
        if last_message.tool_calls:
            return "tools"
            
        # Otherwise, we assume it's done and returned the final JSON
        return END

    graph_builder = StateGraph(InvestigationState)
    
    # Add nodes
    graph_builder.add_node("agent", agent_node)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    
    # Add edges
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    graph_builder.add_edge("tools", "agent")
    
    return graph_builder.compile()
