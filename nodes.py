from langchain_core.messages import AIMessage, ToolMessage
from state import State

def chatbot(llm_with_tools):
    def _chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == "RequestAssistance"
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}
    return _chatbot

def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }

def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    from langgraph.prebuilt import tools_condition
    return tools_condition(state) 