
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
import glob
import os
import streamlit as st
from dotenv import load_dotenv



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

# %% CONFIGURAÇÕES GERAIS

# Diretório do banco vetorial
PERSIST_DIRECTORY = "./code/rag_llm/chroma_rh"

# Modelo de embedding
EMBEDDING_MODEL = "text-embedding-3-small"

# Modelo de linguagem
LLM_MODEL = "gpt-4o-mini"

# %% LEITURA DOS DOCUMENTOS 

@st.cache_data
def carregar_documentos():
    documents_path = glob.glob('documents/rh/*.pdf')

    documentos = []
    for _document in documents_path:
        
        loader = PyPDFLoader(_document)
        docs = loader.load()
        for doc in docs:
            print(f'Documentos: {doc}\n')
            print(f'Parte dos documentos: {doc}\n')
            print(5*'----')

            doc.metadata['documento'] = _document.split('/')[2].split('.pdf')[0]
        
        documentos.extend(docs)

        return documentos

# %% CHUNKING

def gerar_chunks(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150
    )

    return splitter.split_documents(documentos)

# %% ENRIQUECIMENTO COM METADADOS

def enriquecer_chunks(chunks):

    for chunk in chunks:
        texto = chunk.page_content.lower()

        if "férias" in texto:
            chunk.metadata["categoria"] = "férias"
        elif "home office" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif ("conduta" in texto) or ("ética" in texto):
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    
    return chunks

# %% VECTOR STORE

@st.cache_resource
def criar_vectorstore(_chunks):

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )

    return vectorstore


# %% RERANKING 

def rerank_documentos(pergunta, documentos, llm):

    PROMPT = """
    Você é um especialista em políticas internas de RH.

    Pergunta do usuário:
    {pergunta}

    Trecho do documento:
    {texto}

    Avalie a relevência desse trecho para responder a pergunta.
    Responda apenas com um número de 0 a 10.
    """

    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template=PROMPT
    )

    documentos_com_score = []

    for doc in documentos:
        score = llm.invoke(
            prompt_rerank.format(
                pergunta=pergunta,
                texto=doc.page_content
            )
        ).content

        try:
            score = float(score)
        except:
            score = 0
        
        documentos_com_score.append((score, doc))
    
    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x:x[0],
        reverse=True
    )

    return [doc for _, doc in documentos_ordenados]

# %% PIPELINE RAG COMPLETO

def responder_pergunta(pergunta, vectorstore):

    """"
    Pipeline completo:
        - Recuperação
        - Reranking
        - Geração de resposta
    """

    # LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # Recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8,
    )

    # Reranking 
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )

    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    PROMPT_FINAL = f"""
    Você é um agente de RH corporativo.
    Responda APENAS com base nas políticas internas abaixo.

    Contexto:
    {contexto_texto}

    Pergunta:
    {pergunta}
    """

    resposta = llm.invoke(PROMPT_FINAL)

    return resposta.content, contexto_final

# %% INTERFACE STREAMLIT

st.set_page_config(page_title="CHAT de RH com RAG", layout="wide")
st.title("CHAT de RH Política Internas")

pergunta = st.text_area("Digite sua pergunta sobre políticas internas de RH: ")

if pergunta:
    with st.spinner("Consultando políticas internas..."):
        documentos = carregar_documentos()
        chunks = gerar_chunks(documentos)
        chunks = enriquecer_chunks(chunks)
        vectorstore = criar_vectorstore(chunks)

        resposta, fontes = responder_pergunta(pergunta, vectorstore)

    st.subheader("Resposta")
    st.write(resposta)

    st.subheader("Fonte utilizadas")
    for i, doc in enumerate(fontes, start=1):
        st.markdown(f"**Trecho {i}**")
        st.write(f"Documento: {doc.metadata.get('documento')}")
        st.write(f"Categoria: {doc.metadata.get('categoria')}")
        st.write(doc.page_content)
        st.divider()