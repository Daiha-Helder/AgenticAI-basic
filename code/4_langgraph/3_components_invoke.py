from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display

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

import warnings
warnings.filterwarnings(action='ignore')


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# %% Defininido Tavily tool
tool_instance = TavilySearchResults(
    max_results = 4
)

# print(type(tool))
# print(tool.name)

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
        self.tools = {t.name:t for t in tools}
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
                tool_output = "bad tool name, retry"
            else:
                tool_output = self.tools[t['name']].invoke(t['args'])
            result.append(ToolMessage(
                tool_call_id = t['id'],
                name = t['name'],
                content=str(tool_output)
            ))
        print("Back to the model")
        return {"messages": result}

# %% Defininido o agente
prompt = """
Você é um assistente de pesquisa inteligente. Use o mecanismo de busca para procurar informações. \
Você tem permissãos para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza o que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!.  
"""

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature = 0
)

abot = Agent(model, [tool_instance], system=prompt)
messages = [HumanMessage(
    content="Como está o tempo em São Paulo e no Rio de Janeiro hoje?"
)]

print("Iniciando interação do agente: ")


result = abot.graph.invoke({"messages": messages})

print("\nResultado Final:")
print(result['messages'][-1].content)




