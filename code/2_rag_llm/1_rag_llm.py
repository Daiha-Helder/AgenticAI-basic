
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

path_pdf = 'documents/soccer/Regras_do_Jogo.pdf'


# Carrega o PDF
loader = PyPDFLoader(path_pdf)
documents = loader.load()
print('Número de páginas',len(documents))

# Divide os documentos em chunks menores
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents)

print('Número de chunks',len(chunks))

# Inicializa embedding
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key
)

# Cria o banco vetorial
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./code/rag_llm/chroma_regra_futebol"
)

# Cria o retriever
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
)

# Inicializa o modelo de liguagem
llm = ChatOpenAI(
    model="gpt-4o-mini",
     api_key=api_key
)

# Cria a cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents = True
)

# Pergunta de teste
pergunta = "Um jogador pode usar a mão para marca um gol?"

# Executa a pergunta no agente RAG
resposta  = qa_chain.invoke(pergunta)

print("Pergunta:")
print(pergunta)

print("\nResposta do Agente")
print(resposta["result"])

print("\nTrechos utilizados como contexto:\n")

for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")
    print(f"Fonte: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")
    print("Conteúdo:")
    print(doc.page_content)
    print("\n")