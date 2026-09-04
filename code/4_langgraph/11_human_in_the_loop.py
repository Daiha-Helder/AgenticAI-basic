from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage, BaseMessage, AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver

from IPython.display import Image, display
from typing_extensions import TypedDict
import sqlite3
from datetime import date
from typing import Annotated
from uuid import uuid4

import os 
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

def reduce_messages(
        left: list[AnyMessage],
        right: list[AnyMessage]
    ) -> list[AnyMessage]:

    for message in right:
        if not message.id:
            message.id = str(uuid4())

    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            merged.append(message)
    return merged


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


tool = TavilySearch(max_results=2)

class Agent: 

    def __init__(
            self,
            model,
            tools,
            system="",
            checkpointer = None
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

        self.graph = graph.compile(
            checkpointer = checkpointer,
            interrupt_before = ["action"] # Adiciona interrupção antes de chamar a ação
            )

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


current_date = date.today().strftime("%d/%m/%Y") 

prompt = f"""Você é um assistente de pesquisa inteligente e altamente atualizado. \
Sua principal prioridade é encontrar as informações mais RECENTES e em TEMPO REAL sempre que possível. \
A data atual é {current_date}. \
Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora", \
você DEVE **incluir a data atual '{current_date}' na sua consulta para a ferramenta de busca**. \
Por exemplo, se a pergunta é "tempo em cidade x hoje", a consulta para a ferramenta deve ser "tempo em cidade x {current_date}". \
Ignore ou descarte informações que claramente se refiram a datas passadas ou futuras ao responder perguntas sobre "hoje". \
Use o mecanismo de busca para procurar informações, sempre buscando o 'hoje' ou o 'agora' quando o contexto indicar. \
Você tem permissão para fazer múltiplas chamadas (seja em conjunto ou em sequência). \
Procure informações apenas quando tiver certeza do que você quer. \
Se precisar pesquisar alguma informação antes de fazer uma pergunta de acompanhamento, você tem permissão para fazer isso!
"""

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature = 0
)

abot = Agent(
    model, 
    [tool], 
    system=prompt, 
    checkpointer=memory
)

new_session_id = str(uuid4())
print(f"DEBUG: Iniciando nova conversa com ID: {new_session_id}\n")
new_user_message = "Qual é a distância entre Rio de Janeiro e Tóquio?"
new_messages = [HumanMessage(content=new_user_message)]
new_thread_config = {"configurable": {"thread_id": new_session_id}}

print("--- Iniciando NOVA Interação: Agente precessa a entrada e decide a ação ---")
print(f"Você: {new_user_message}")
print(f"DEBUG: Nova Thread ID: {new_session_id}")


print("\n --- Agente pensando e pausando ---")
try:
    for event in abot.graph.stream({"messages": new_messages}, new_thread_config):
        for k, v in event.items():
            if k=="llm":
                if v and ('messages' in v) and v['messages']:
                    llm_message_from_event = v['messages'][0]
                    if hasattr(llm_message_from_event, 'tool_calls') and llm_message_from_event.tool_calls:
                        print(f"\nAgente (decisão): {llm_message_from_event.tool_calls}")
                        print("\n --- AGENTE PAUSADO: Intervenção Humana Necessário ---")
                    elif llm_message_from_event.content:
                        print(f"\nAgente (resposta direta): {llm_message_from_event.content}")
                        print("\n --- AGENTE NÃO PAUSOU PARA FERRAMENTA (Resposta direta do LLM) ---")
except Exception as e:
    print(f"DEBUG: Stream interrompido como esperado: {e}")

current_state_snapshot = abot.graph.get_state(new_thread_config)

if current_state_snapshot:
    print(f"\nDEBUG: Estado atual obtido para NOVA thread ID:{new_session_id}")

    

