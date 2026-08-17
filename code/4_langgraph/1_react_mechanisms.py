import os 
import re 
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict

import os 
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

genai.configure(api_key=api_key)
client = genai.GenerativeModel('gemini-3.5-flash-lite')

class Agent:

    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({
                "role":"system",
                "content": self.system
            })

    def __call__(self, message):

        self.messages.append({
            "role":"user",
            "content": message
        })

        result = self.execute()

        self.messages.append({
            "role":"assistant",
            "content": result
        })

        return result

    def execute(self):
        # Constrói o prompt com todo o histório de mensagens
        prompt = ""
        for msg in self.messages:
            prompt += f"{msg['role']}: {msg['content']}\n"

        # Envia para o Gemini
        response = client.generate_content(prompt)
        return response.text

PROMPT_REACT = """"
Você funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
Ao final do ciclo, você fornece uma Resposta.
Use "Pensamento" para descrever seu raciocínio.
Use "Ação" para executar ferramentas - e então retorne "PAUSA".
A "Observação" será o resultado da ação executada.
Ações disponíveis:
    - consultar_estoque: retorna a quantidade disponível de um item no inventário(ex: "consultar_estoque: teclado")
    - consultar_preco_produto: retorna o preço unitário de um produto (ex: "consultar_preco_produto: mouse gamer")

Exemplo:
Pergunta: Quantos monitores temos em estoque?
Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
Ação: consultar_estoque: monitor
PAUSA

Observação: Temos 75 monitores em estoque.
Resposta: Há 75 monitores em estoque.
""".strip()


if __name__ == "__main__":
    agent = Agent(system="Você é um assistente útil e objetivo")
    print(agent("Qual é a capital da França?"))
