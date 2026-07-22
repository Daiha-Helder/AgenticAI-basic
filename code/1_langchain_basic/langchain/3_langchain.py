from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")


numero_dias = 7
numero_criancas = 2
atividade = "praia"


# Observe que não existe "f" no começa da string
modelo_de_prompt = PromptTemplate(
    template = """
    Crie um roterio de viagem de {dias} dias,
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """
)

prompt = modelo_de_prompt.format(
    dias = numero_dias,
    numero_criancas = numero_criancas,
    atividade = atividade
)


print("Prompt : \n", prompt)

modelo = ChatOpenAI(
    model = "gpt-3.5-turbo",
    temperature = 0.5,
    api_key = api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)