from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import glob
from dotenv import load_dotenv
import os



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

modelo = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.5, 
    api_key=api_key
    )

embeddings = OpenAIEmbeddings()

documentos_paths = glob.glob('documents/*.pdf')

documentos = sum(
    [
        PyPDFLoader(doc).load() for doc in documentos_paths
    ], []
)

partes = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
).split_documents(documents=documentos)

dados_recuperados = FAISS.from_documents(
    documents=partes,
    embedding=embeddings
).as_retriever(search_kwargs={'k':2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda usando exclusivamente o conteúdo fornecido"),
        ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(trecho.page_content for trecho in trechos)
    return cadeia.invoke(
        {
            "query": pergunta,
            "contexto": contexto
        }
    )

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha um cartão gold?"))