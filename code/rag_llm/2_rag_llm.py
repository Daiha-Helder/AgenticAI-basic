
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
import random
import glob
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

# %% Carrega documentos 

# Lista os caminhos dos pdfc com as bulas
# documents_path = glob.glob('documents/medicine/*.pdf')

# Lista que armazenará todos os documentos carregados
documentos = []

# Percorre cada bula 
for doc_path in documents_path:

    # Cria o loades do PDF
    loader = PyPDFLoader(doc_path)
    
    # Carrega o conteúdo do PDF
    docs = loader.load()
   
    # Adiciona o nome do medicamento como metadado
    for doc in docs:
        doc.metadata['medicamento'] = doc_path.split('/')[2].split('.pdf')[0]
    print(doc.metadata['medicamento'])
   
    # Adiciona os documentos à lista principal
    documentos.extend(docs)

# Quantidade total de páginas carregadas
# print(f'Quantidade total de páginas carregadas: {len(documentos)}')

# %% Cria os chunks

# Cria o objeto responsávle pelo chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, # Tamanho máximo de cada chunk
    chunk_overlap=150 # Sobreposição entre chunks
)

# Divide os documentos em chunks
chunks = text_splitter.split_documents(documents=documentos)

# Quantidade total de chunks gerados
print(f'Quantidade total de chunks gerados: {len(chunks)}')

# Percorre cada chunk para classificar semanticamente seu conteúdo
for chunk in chunks:

    # Normaliza o texto para facilitar as verificações 
    texto = chunk.page_content.lower()

    # Indentificação do medicamento 
    if ("identificação do medicamento" in texto) or ("composição" in texto):
        chunk.metadata["categoria"] = "identificacao"

    # Indicações terapêuticas 
    elif ("indicação" in texto) or ("para que esse medicamento é indicado" in texto):
        chunk.metadata["categoria"] = "indicacao"

    # Funcionamento do medicamento
    elif ("como este medicamento funciona" in texto) or ("ação" in texto):
        chunk.metadata["categoria"] = "como_funciona"

    # Contraindicações
    elif ("contraindicações" in texto) or ("quando não devo usar" in texto):
        chunk.metadata["categoria"] = "contraindicacao"

    # Advertências e precauções
    elif ("advertência" in texto) or ("precaução" in texto) or ("o que devo saber antes de usar" in texto):
        chunk.metadata["categoria"] = "advertencias_precaucoes"
    
    # Interações medicamentosas
    elif ("interação" in texto) or ("interações medicamentosas" in texto):
        chunk.metadata["categoria"] = "interacoes"

    # Posologia e modo de uso
    elif ("dose" in texto) or ("posologia" in texto) or ("como devo usar" in texto):
        chunk.metadata["categoria"] = "posologia_modo_uso"

    # Reações adversas
    elif ("reações adversas" in texto) or ("quais os males" in texto):
        chunk.metadata["categoria"] = "reacoes_adversas"
    
    # Armazanamento
    elif ("onde, como e por quanto tempo posso guardar" in texto) or ("armazenar" in texto):
        chunk.metadata["categoria"] = "armazenamento"
    
    # Superdosagem
    elif ("quantidade maior do que a indicada" in texto) or ("superdosagem" in texto):
        chunk.metadata["categoria"] = "superdosagem"
    
    # Conteúdo geral/ administrativo
    else:
        chunk.metadata["categoria"] = "geral"

# # Seleciona dois chunks aleatório
# chunks_aleatorios = random.sample(chunks, 2)

# # Imprime os metadados e um texto do conteúdo
# for i, chunk in enumerate(chunks_aleatorios, start=1):
#     print(f"\n--- Chunk Aleatório {i} ---")
#     print("Metadados:")
#     print(chunk.metadata)
#     print("\nConteúdo (início):")
#     print(chunk.page_content[:300])


# %% Cria banco vetorial

# Inicializa o modelo de embedding (forma atual)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Cria banco vetorial
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./code/rag_llm/chroma_bulas"
)

# Cria o retriever para busca semântica
retriever = vectorstore.as_retriever(
    search_kwargs = {"k": 4}
)

# %% Integração da pipeline de RAG

# Inicializa o modelo de linguagem 
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Cria a cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents = True
)

# %% Teste de validação das respostas

# Pergunta de teste
pergunta = "Quais são as contradições da dipirona?"

# Executa a pergunta no agente RAG (forma atual)
resposta = qa_chain.invoke(pergunta)

print("Pergunta:")
print(pergunta)

print("\nResposta do Agente:")
print(resposta["result"])

print("\nTrechos utilizados como contexto:\n")

# Percorre os documentos recuperados 
for i, doc in enumerate(resposta["source_documents"], start=1):
    print(f"--- Trecho {i} ---")

    # Metadados principais
    print(f"Medicamento: {doc.metadata.get('medicamento', 'N/A')}")
    print(f"Categoria: {doc.metadata.get('categoria', 'N/A')}")
    print(f"Documento: {doc.metadata.get('source', 'Documento desconhecido')}")
    print(f"Página: {doc.metadata.get('page', 'N/A')}")

    # Conteúdo recuperado
    print("\nConteúdo do chunk:")
    print(doc.page_content)
    print('\n')