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

cliente_tavily = TavilyClient(
    api_key=TAVILY_API_KEY
)

cidade = "Belém do Pará"
tavily_query = f"""
    Restaurantes em {cidade} tripadvisor com maior quantidade de reviews e faixa de preço
"""

print("Iniciando busca agêntica por URLs do Tripadvisor com Tavily...")
tripadvisor_url = None

try:
    tavily_results = cliente_tavily.search(
        query=tavily_query,
        max_results=5
    )

    if tavily_results and tavily_results["results"]:
        print(f"Tavily encontrou {len(tavily_results['results'])} resultados. Analisando...")

        for result in tavily_results["results"]:
            url = result["url"]

            if ("tripadvisor.com" in url) or ("tripadvisor.com.br" in url):
                tripadvisor_url = url
                break

            if not tripadvisor_url:
                print("Nenhum URL relevante do Tripadvisor foi encontrado nos primeiros resultados.")
        
except Exception as e:
    print(f"Erro na busca agêntica com Tavily: {e}."
           "Verique sua chave de API ou conexão")

if tripadvisor_url:
    clean_url = re.sub(r'-oa\d+-', '-', tripadvisor_url)
    tripadvisor_url = clean_url
    print(" URL encontrada limpa de paginação.")

print("-"*50)
print(f"URL Final do Tripadvisor para raspagem: {tripadvisor_url if tripadvisor_url else 'NÃO ENCONTRADO'}")