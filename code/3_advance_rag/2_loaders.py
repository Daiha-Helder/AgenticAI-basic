from langchain_community.document_loaders import (
        TextLoader, 
        PyPDFLoader, 
        WebBaseLoader, 
        DirectoryLoader, 
        MergedDataLoader,
        RecursiveUrlLoader
)
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

# %%
# Carrega documento PDF
doc_txt = TextLoader("documents/bank/GTB_gold_Nov23.txt").load()

# Extrai o texto de cada página e junta tudo em uma única string
texto_txt = "\n".join([pagina.page_content for pagina in doc_txt])

print(texto_txt)

# %%
# Carrega documento PDF
doc_pdf = PyPDFLoader("documents/bank/GTB_gold_Nov23.pdf").load()

# Extrai o texto de cada página e junta tudo em uma única string
texto_pdf = "\n".join([pagina.page_content for pagina in doc_pdf])

# print(texto_pdf)

# %%
# Carrega página WEB
doc_web = WebBaseLoader(web_path="https://forbes.com.br/forbes-tech/2025/08/o-que-explica-o-fracasso-do-chatgpt-5-e-como-a-openai-vai-reagir/").load()

# Extrai o texto de cada página e junta tudo em uma única string
texto_web = "\n".join([pagina.page_content.strip() for pagina in doc_web])

# print(texto_web)

# %%
# Carrega todos os documentos pdfs de um diretório
doc_folder = DirectoryLoader("documents/bank", glob="*.pdf").load()

folder_pdf = "\n".join(doc.page_content for doc in doc_folder)
print(folder_pdf)

# %%
# Carrega todos os arquivos de um diretório independente da extensão

all_loaders = MergedDataLoader(loaders=[
    WebBaseLoader(web_path="https://forbes.com.br/forbes-tech/2025/08/o-que-explica-o-fracasso-do-chatgpt-5-e-como-a-openai-vai-reagir/"),
    TextLoader("documents/bank/GTB_gold_Nov23.txt", encoding='utf-8')
])

all_docs = all_loaders.load()

print("Dimensão do arquivo", len(all_docs))

texto_all = "\n".join([pagina.page_content for pagina in all_docs])

print(texto_all)

# %%
# Baixa uma página inteira com suas raízes

loader = RecursiveUrlLoader("https://python.langchain.com/api_refence/").load()
print("Dimensão do arquivo", len(loader))
