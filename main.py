import os
from config import ANTHROPIC_API_KEY, TAVILY_API_KEY
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from state import State, RequestAssistance
from nodes import chatbot, human_node, select_next_node
from visualize import visualize_graph

def build_graph():
    # Set up tools
    tool = TavilySearchResults(max_results=2, tavily_api_key=TAVILY_API_KEY)
    tools = [tool]
    
    # Set up LLM
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", api_key=ANTHROPIC_API_KEY)
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])

    # Build graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("chatbot", chatbot(llm_with_tools))
    graph_builder.add_node("tools", ToolNode(tools=[tool]))
    graph_builder.add_node("human", human_node)

    # Add edges
    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # Compile graph
    memory = MemorySaver()
    return graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["human"],
    )

def stream_graph_updates(graph, user_input: str, config: dict):
    events = graph.stream(
        {"messages": [("user", user_input)]}, 
        config,
        stream_mode="values"
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

def main():
    # Initialize graph
    graph = build_graph()
    visualize_graph(graph)
    config = {"configurable": {"thread_id": "1"}}

    # Chat loop
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(graph, user_input, config)
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main() 