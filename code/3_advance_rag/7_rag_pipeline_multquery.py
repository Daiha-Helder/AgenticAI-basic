from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers import MultiQueryRetriever




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
# Definindo o prompt e a chain do mult-retriever

multi_query_prompt_template = """"
Você é um assistente de modelo de linguagem de IA.
Sua tarefa é gerar cinco versões diferentes da pergunta do usuário para recuperar documentos relevantes de um banco de dados vetorial.
Após gerar as múltiplas perspectivas sobre a pergunta do usuário, seu objetivo é ajudar o usuário a superar algumas das limitações da busca por similaridade baseada em distância.
Forneça estas perguntas alternativas separadas por quebras de linha.
Pergunta original: {question}
"""

multi_query_prompt = PromptTemplate.from_template(multi_query_prompt_template) 
multi_query_chain = multi_query_prompt | modelo | CommaSeparatedListOutputParser()
#%%
# Construindo a chain do mult-retriever integrado

multi_retriever = MultiQueryRetriever(
    retriever=retriever,
    llm_chain=multi_query_chain
)

multi_rag_chain = (
    {
        "contexto": RunnablePassthrough() | multi_retriever,
        "query": RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)

resposta = multi_rag_chain.invoke(pergunta)
print(resposta)

