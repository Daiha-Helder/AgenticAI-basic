from langchain_classic.evaluation.qa import QAEvalChain, QAGenerateChain
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")


modelo = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
)


#%%
# Carrega todos os documentos pdfs de um diretório
doc_folder = DirectoryLoader("documents/bank", glob="*.pdf").load()


#%% 
# Dividindo o documento
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1250,
    chunk_overlap=125
)

partes = splitter.split_documents(doc_folder)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(
    documents=partes,
    embedding=embeddings
)

retriever = vector_store.as_retriever()

#%%
# Criando função que avalia o resultado 
eval_chain = QAEvalChain.from_llm(modelo)


def evaluate_question_answer(question_answer, generations):
    # Variável question_answer: query, answer
    # generations: result

    evaluates = eval_chain.evaluate(question_answer, generations)
    correct_answer = 0
    for i, e in enumerate(question_answer):
        correct_answer = correct_answer + (1 if evaluates[i]["results"].split("\n")[-1].split(":")[-1].strip()=="CORRECT" else 0)

    return correct_answer/len(question_answer)

# %%
# Função que preparado os dados para serem avaliados

qa_chain = QAGenerateChain.from_llm(modelo)
question_answer = qa_chain.apply_and_parse(
    [{"doc":p.page_content} for p in partes]
)

# Salvando os daddos
with open('./data/qa_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(question_answer, f, indent=4, ensure_ascii=False)

print(f"Quantidade de perguntas e respsotas: {len(question_answer)}")

# %%
# Ler os arquivos

with open('./data/qa_pairs.json', 'r') as f:
    pairs = json.load(f)

question_answer = [p['qa_pairs'] for p in pairs]
question_answer = question_answer[:10]

# %%
# Avaliando respostas sem o RAG
generation_without_rag = []
for qa in question_answer:
    generation_without_rag.append({"result": modelo.invoke(qa["query"]).content})


# Avalia perguntas em respostas sem o RAG
qa_without_rag = evaluate_question_answer(
    question_answer, 
    generation_without_rag
)

print(f"Resultado da avaliação sem RAG: {qa_without_rag}")

# %%
# Criando o RAG

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente os conteúdo fornecidos. \n\nContexto:\n{contexto}"),
    ("human", "{query}")
])

rag_chain = ({
    "contexto": RunnablePassthrough() | retriever,
    "query": RunnablePassthrough()
    }
    | prompt | modelo | StrOutputParser()
)


generation_with_rag = []
for qa in question_answer:
    generation_with_rag.append({"result": rag_chain.invoke(qa["query"])})

qa_with_rag = evaluate_question_answer(
    question_answer, 
    generation_with_rag
)

print(f"Resultado da avaliação com RAG: {qa_with_rag}")

# %%
# Utilizando multi RAG

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


generation_with_rag_and_multiquery = []
for qa in question_answer:
    generation_with_rag_and_multiquery.append({"result": multi_rag_chain.invoke(qa["query"])})

qa_with_rag_and_multiquery = evaluate_question_answer(
    question_answer, 
    generation_with_rag_and_multiquery
)


print(f"Resultado da avaliação com RAG e Multiquery: {qa_with_rag_and_multiquery}")