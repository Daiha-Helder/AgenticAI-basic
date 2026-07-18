
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
import glob
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

documents_path = glob.glob('documents/medicine/*.pdf')
print(documents_path)

for doc_path in documents_path:

    
    loader = PyPDFLoader(doc_path)
    docs = loader.load()
   
    for doc in docs:
        doc.metadata['medicamento'] = doc_path.split('/')[2].split('.pdf')[0]
    print(doc.metadata['medicamento'])
   

