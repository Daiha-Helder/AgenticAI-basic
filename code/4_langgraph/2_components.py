from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
    BaseMessage
)

from langchain_community.tools.tavily_search import TavilySearchResults

import os 
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# %% Defininido Tavily tool
tool = TavilySearchResults(
    max_results = 4
)

print(type(tool))
print(tool.name)

# %% Agente que acrescenta uma mensagem ao histórico
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:
    def __init__(self, model, tools, system=""):

        self.system = system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exist_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tool = {t.name:t for t in tools}
        self.model = model.bind_tools(tools)

    def exist_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_gemini(self, state: AgentState):
        messages = state['messages']

        if self.system:
            messages = [
                SystemMessage(content=self.system)
                ] + messages

        messages = self.model.invoke(messages)
        return {"messages": [messages]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        result = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ...bad tool name...")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name'].invoke(t['args'])]
            result.append(ToolMessage(
                tool_call_id = t['id'],
                name = t['name'],
                content=str(result)
            ))
        print("Back to the model")
        return {"messages": result}


            


