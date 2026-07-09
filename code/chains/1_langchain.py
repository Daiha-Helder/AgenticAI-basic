from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"]
)

modelo = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.5,
    api_key = api_key
)

cadeia = prompt_cidade | modelo | StrOutputParser()

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)