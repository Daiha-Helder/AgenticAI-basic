from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

modelo = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0.5,
    api_key = api_key
)

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Praia. Você é uma especialista em viagens com destinos para praia"),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr. Montana. Você é uma especialista em viagens com destinos para montanhas e atividades radicais"),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}")
    ]
)


roteador = prompt_roteador | modelo.with_structured_output(Rota)


def responda(pergunta: str):

    rota = roteador.invoke({"query": pergunta})['destino']
    print
    if rota == 'praia':
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})

print(responda("Quero surfar em um lugar quente"))