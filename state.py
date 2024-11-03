from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly 
    or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """
    request: str 