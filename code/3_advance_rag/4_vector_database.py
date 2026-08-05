from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
pc_api = os.getenv("PINECONE_API")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

#%%
# Carrega todos os documentos pdfs de um diretório
doc_folder = DirectoryLoader("documents/bank", glob="*.pdf").load()


# %%
# Quebra de chunk de forma semântica usando (modelo da Hugging Face)

hf_emb_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')
semantic_hf_splitter = SemanticChunker(hf_emb_model)
semantic_hf_partes = semantic_hf_splitter.split_documents(doc_folder)


# print('Número de chunks (semântica Hugging Face)', len(semantic_hf_partes))

# %% Construindo Vector Store com InMemoryVectorStore

vector_store_IMVS = InMemoryVectorStore.from_documents(
    documents=semantic_hf_partes,
    embedding=hf_emb_model
)

retriever_IMVS = vector_store_IMVS.as_retriever(search_kwargs={'k':3})

# print(retriever_IMVS.invoke("Sala VIP"))

# %% Construindo Vector Store com FAISS

vector_store_FAISS = FAISS.from_documents(
    documents=semantic_hf_partes,
    embedding=hf_emb_model
)

retriever_FAISS = vector_store_FAISS.as_retriever(search_kwargs={'k':3})

# print(retriever_FAISS.invoke("Sala VIP"))

# %% Construindo Vector Store com Pinecone

pc_vector_store = PineconeVectorStore(
    host="https://rag-langchain-1f462bc.svc.aped-4627-b74a.pinecone.io",
    pinecone_api_key=pc_api,
    embedding=hf_emb_model
)


pc_vector_store.add_documents(semantic_hf_partes)
retriever_pc = pc_vector_store.similarity_search_with_score("Seguro viagem")

print("Pinecone")
print(15*"==")
print(retriever_pc)

