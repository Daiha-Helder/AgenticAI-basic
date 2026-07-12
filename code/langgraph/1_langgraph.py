from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

prompt_consultor = ChatPromptTemplate(
    [
        ("system", "Você é um consultor de viagens"),
        ("human", "{query}")
    ]
)

assistente = prompt_consultor | modelo | StrOutputParser()

print(assistente.invoke({
    "query": "Quero férias em praias do Brasil."
}))