from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_debug
from langsmith import traceable
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

# Carrega documentos
documento = TextLoader("documents/bank/GTB_gold_Nov23.txt", encoding="utf-8").load()

# Divide o texto em parte
partes = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
).split_documents(documento)

# print(f"\n --- Exemplo do primeiro embedding ---\n")
# print(partes[0].page_content[0:100])
# print(50*"---")

# Chama o modelo de embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# print(f"\n --- Embedding do primeiro chunk ---\n")
# print(embeddings_model.embed_query(partes[0].page_content)[0:10])
# print(50*"---")

# Crie o banco vetorial na memória
vectorstore = InMemoryVectorStore.from_documents(
    documents=partes,
    embedding=embeddings_model
)

# Chama o retriever
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

# Prompt para consultar documentos 
prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido. \n\nContexto: \n{contexto}"),
        ("human", "{query}")
    ]
)

# Modelo 
modelo = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.5,
)

# Define funções com e sem RAG

def no_rag(pergunta):
    return modelo.invoke(pergunta).content

@traceable
def rag(pergunta):

    # Cadeia
    cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

    # Retriever
    trechos = retriever.invoke(pergunta)

    # Contexto
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)

    return cadeia.invoke({"query": pergunta, "contexto": contexto})

print("Resposta sem RAG")
print(no_rag("O que devo fazer se tiver um item roubado?"))

print("\nResposta com RAG")
print(rag("O que devo fazer se tiver um item roubado?"))
