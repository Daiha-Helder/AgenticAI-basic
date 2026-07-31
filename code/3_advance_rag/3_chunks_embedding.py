from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModel
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

#%%
# Carrega todos os documentos pdfs de um diretório
doc_folder = DirectoryLoader("documents/bank", glob="*.pdf").load()


# %%
# Quebra por caracteres
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

partes = splitter.split_documents(doc_folder)
print('Número de chunks (caracteres)', len(partes))


# %%
# Quebra por tokens
token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50
) 

token_partes = token_splitter.split_documents(doc_folder)
print('Número de chunks (OpenAI)', len(token_partes))


# %% 
# Importando modelo de embedding

emb_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small') 
emb_model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')

hf_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=emb_tokenizer,
    chunk_size=512,
    chunk_overlap=50
)

hf_partes = hf_splitter.split_documents(doc_folder)

print('Número de chunks (hugging face)', len(hf_partes))

# %%
# Quebra de chunk de forma semântica (usando modelo da OpenAI)

sematic_openai_splitter = SemanticChunker(OpenAIEmbeddings())
sematic_openai_partes = sematic_openai_splitter.split_documents(doc_folder)


print('Número de chunks (semântica OpenAI)', len(sematic_openai_partes))

# %%
# Quebra de chunk de forma semântica usando (modelo da Hugging Face)

hf_emb_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')
semantic_hf_splitter = SemanticChunker(hf_emb_model)
semantic_hf_partes = semantic_hf_splitter.split_documents(doc_folder)


print('Número de chunks (semântica Hugging Face)', len(semantic_hf_partes))