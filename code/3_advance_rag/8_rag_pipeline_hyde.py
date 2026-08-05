from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



#%%
# Carrega todos os documentos pdfs de um diretório
doc_folder = DirectoryLoader("documents/bank", glob="*.pdf").load()

#%%
# Definindo o tokenizador
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

#%% 
# Dividindo o documento
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1250,
    chunk_overlap=125
)

partes = splitter.split_documents(doc_folder)

#%%
# Criando banco vetorial (vector store)
embeddings = OllamaEmbeddings(model="bge-m3:567m")

vector_store = FAISS.from_documents(
    documents=partes,
    embedding=embeddings
)

#%%
# Definindo o prompt e o modelo
modelo = OllamaLLM(model="gemma4:e4b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente os conteúdo fornecidos. Seja breve na resposta. \n\nContexto:\n{contexto}"),
    ("human", "{query}")
])

#%%
# Usando o retriver para fazer a recuperação
pergunta = "Como fazer um seguro viagem?"

retriever = vector_store.as_retriever()
trechos = retriever.invoke(pergunta)
contexto = "\n\n".join(trecho.page_content for trecho in trechos)

#%%
# Definindo o prompt da hyde

hyde_prompt_template = """"
Escreva um paragrafo que possa responder a pergunta apresentada. Não adicione informações.
Pergunta: {user_quetion}
Parágrafo:
"""

hyde_prompt = PromptTemplate.from_template(hyde_prompt_template)
hyde_chain = hyde_prompt | modelo | StrOutputParser()

#%%
# Definindo a chain hyde 

hyde_rag_chain = ({
    "contexto": RunnablePassthrough() | hyde_chain | retriever,
    "query": RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)

print(hyde_chain.invoke(pergunta))







