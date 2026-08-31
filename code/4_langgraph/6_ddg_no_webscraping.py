import os 
import google.generativeai as genai
from tavily import TavilyClient
import requests
import re
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


from dotenv import load_dotenv
load_dotenv()


GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

client = TavilyClient(
    api_key=TAVILY_API_KEY
)

%% Tavily (busca agêntica)
result = client.search(
    query="O que são multiagentes de Inteligência Artificial?",
    include_answer=True
)
print(result['answer'])

# %% Duck Duck Go (busca convencional): Traz o link mas não faz a rapagem

cidade = "Belém"

query = f"""
Liste os 5 principais restaurantes em {cidade}, segundo avaliações recentes no TripAdvisor ou sites similares de turismo.
Para cada restaurante, informe:
- Tipo de culinária (ex: regional, italiana, japonesa)
- Uma breve descrição (máx. 2 linhas)
- Avaliação média (se disponível)
- Faixa de preço
Responda apenas com dados atualizados e relevantes para turistas.
"""

ddg = DDGS()

def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i['href'] for i in results]
    except Exception as e:
        raise e


for link in search(query):
    print(link)

