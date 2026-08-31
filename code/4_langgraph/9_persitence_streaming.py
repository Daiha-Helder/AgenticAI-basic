import os 
from dotenv import load_dotenv
load_dotenv()

import operator
from typing import Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage, AnyMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from langgraph.checkpoint import SqliteSaver
from typing_extensions import TypedDict
import sqlite3

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

class Agent: 

    def __init__(
            self,
            model,
            tools,
            checkpointer,
            system=""
            ):

        self.system = system

        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)

        graph.set_entry_point("llm")
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
            )
        graph.add_edge("action", "llm")

        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        def call_gemini(
                self, 
                state: AgentState):

            messages = state['messages']
            if self.system:
                messages = [SystemMessage(content=self.system)] + messages

            print("Mensagens enviadas ao modelo:", messages)
            messages = self.model.invoke(messages)
            return {'messages': [messages]}

        def exists_action(
                self,
                state: AgentState
                ):

            result = state['messages'][-1]
            return len(result.tool_calls) > 0

        def take_action(
                self,
                state: AgentState
                ):

            tool_calls = state['messages'][-1].tool_calls
            results = []
            for t in tool_calls:

                print(f"Calling Tool: {t['name']} with args: {t['args']}")

                result = self.tools[t['name']].invoke(t['args'])
                results.append(
                    ToolMessage(
                        tool_call_id=t['id'],
                        name=t['name'],
                        content=str(result)
                    )
                )

            print("Returning to LLM after action!")
            return {'messages': results}
            
                

      

