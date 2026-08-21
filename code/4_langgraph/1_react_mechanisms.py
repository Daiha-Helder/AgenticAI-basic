import os 
import re 
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os 
from dotenv import load_dotenv



load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    raise ValueError("A chave da API não foi definida no .env")

print("Chave carregada com sucesso!")

genai.configure(api_key=api_key)
client = genai.GenerativeModel('gemini-3.5-flash-lite')

class Agent:

    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({
                "role":"system",
                "content": self.system
            })

    def __call__(self, message):

        self.messages.append({
            "role":"user",
            "content": message
        })

        result = self.execute()

        self.messages.append({
            "role":"assistant",
            "content": result
        })

        return result

    def execute(self):
        # Constrói o prompt com todo o histório de mensagens
        prompt = ""
        for msg in self.messages:
            prompt += f"{msg['role']}: {msg['content']}\n"

        # Envia para o Gemini
        response = client.generate_content(prompt)
        return response.text

PROMPT_REACT = """"
Você funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
Ao final do ciclo, você fornece uma Resposta.
Use "Pensamento" para descrever seu raciocínio.
Use "Ação" para executar ferramentas - e então retorne "PAUSA".
A "Observação" será o resultado da ação executada.
Ações disponíveis:
    - consultar_estoque: retorna a quantidade disponível de um item no inventário(ex: "consultar_estoque: teclado")
    - consultar_preco_produto: retorna o preço unitário de um produto (ex: "consultar_preco_produto: mouse gamer")
    - encontrar produto mais caro: retorna o nome e o preço do produto mais caro no inventário (não requer argumentos)
    - calcular_valor_total_lista: calcula o valor total de uma lista de itens de compra. Recebe uma string com itens separados por vírgula (ex: "teclado, mouse gamer, monitor")

Exemplo:
Pergunta: Quantos monitores temos em estoque?
Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
Ação: consultar_estoque: monitor
PAUSA

Observação: Temos 75 monitores em estoque.
Resposta: Há 75 monitores em estoque.

Exemplo:
Pergunta: Qual é o produto mais caro?
Pensamento: Preciso usar a ação encontrar_produto_mais_caro para descobrir qual produto tem o maior preço.
Ação: encontrar_produto_mais_caro
PAUSA

Observação: O produto mais caro é o(a) monitor com preço de R$999.90.
Resposta: O produto mais caro é o(a) monitor com preço de R$999.90.

Exemplo:
Pergunta: Quanto custa um teclado e um mouse gamer?
Pensamento: O usuário quer saber o valor total de vário itens. Devo usar a ação calcular_valor_total_lista com os itens "teclado, mouse gamer".
Ação: calcular_valor_total_lista: teclado, mouser gamer
PAUSA

Observação: O valor total dos itens encontrados é R$ 249.50.
Resposta: O valor total do teclado e do mouse gamer é R$ 249.50.
""".strip()


class AgenteState(TypedDict):
    pergunta: str
    historico: list[str]
    acao_pendente: str
    resposta_final: str

def consultar_estoque(item:str) -> str:
    """
    Simula a consulta de estoque de um item no inventário
    """
    item = item.lower()
    estoque = {
        "monitor": 75,
        "teclado": 120,
        "mouse gamer": 80,
        "webcam": 40,
        "headset": 60,
        "impressora": 15
    }

    if item in estoque:
        return f"Temos {estoque[item]} {item}s em estoque."
    else:
        return f"Item '{item}' não encontrado no inventário."


def consultar_preco_produto(produto: str) -> str:
    """
    Simula a consulta do preço unitário de um produto
    """
    produto = produto.lower()
    precos = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    if produto in precos:
        return f"O preço de um(a) {produto} é R$ {precos[produto]:.2f}."
    else:
        return f"Produto '{produto}' não encontrado na lista de preços."

def ferramenta_encontrar_produto_mais_caro():
    """
    Retorna o nome e o preço do produto mais caro no inventário.
    Esta função não precisa de argumentos.
    """

    precos_do_inventario  = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    if not precos_do_inventario:
        return "Nenhum produto encontrado na lista de preço para a comparação"

    nome_produto_mais_caro = max(
        precos_do_inventario,
        key=precos_do_inventario.get)

    valor_produto_mais_caro = precos_do_inventario[nome_produto_mais_caro]

    return f"O produto mais caro é o(a) {nome_produto_mais_caro} com preço de R$ {valor_produto_mais_caro:.2f}"

def ferramenta_calcular_valor_total_list(lista_itens: str) -> str:
    """
    Calcula o valor total de uma lista de itens de compra.
    Recebe uma string com itens separados por virgula (ex: "teclado, mouse gamer, monitor").
    """

    precos_do_inventario  = {
        "monitor": 999.90,
        "teclado": 150.00,
        "mouse gamer": 99.50,
        "webcam": 120.00,
        "headset": 180.00,
        "impressora": 750.00
    }

    itens_processados = [item.strip().lower() for item in lista_itens.split(',')]
    valor_total = 0.0
    itens_nao_encontrados = []

    for item in itens_processados:
        if item in precos_do_inventario:
            valor_total += precos_do_inventario[item]
        else:
            itens_nao_encontrados.append(item)

    resposta = f"O valor total dos itens encontrados é R$ {valor_total:.2f}"
    if itens_nao_encontrados:
        resposta += f"Os seguintes itens não foram encontrados e não foram incluídos no cálculo: {', '.join(itens_nao_encontrados)}"
    return resposta

    
def run_react_agent(pergunta: str, max_iterations: int = 5) -> str:
    """
    Executa o ciclo ReAct para uma dada pergunta usando o modelo Gemini
    """
    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    chat = model.start_chat(history=[])

    chat.send_message(PROMPT_REACT)
    current_prompt = pergunta

    for i in range(max_iterations):
        response = chat.send_message(current_prompt)
        response_text = response.text.strip()

        print(f"--- Iteração {i+1} ---")
        print(f"Modelo pensou/respondeu:\n{response_text}\n")

        response_match_final = re.search(r"Resposta:\s*(.*)", response_text, re.DOTALL)
        if response_match_final:
            return response_match_final.group(1).strip()

        match = re.search(r"Ação:\s*(\w+)(?::\s*([^\n]*))?", response_text)
       

        if match:
            action_name = match.group(1).strip()
            action_arg = match.group(2).strip() if match.group(2) is not None else ""

            observacao_da_acao = ""
            if action_name == "consultar_estoque":
                observacao_da_acao = consultar_estoque(action_arg)
            elif action_name == "consultar_preco_produto":
                observacao_da_acao = consultar_preco_produto(action_arg)
            elif action_name == "encontrar_produto_mais_caro":
                observacao_da_acao = ferramenta_encontrar_produto_mais_caro()
            elif action_name == "calcular_valor_total_lista":
                observacao_da_acao = ferramenta_calcular_valor_total_list(action_arg)
            else:
                observacao_da_acao = f"Erro: Ação '{action_name}' desconhecida. Verifique o prompt ou a implementação da ferramenta."

            current_prompt = f"Observação: {observacao_da_acao}"

            print(f"Executou ação: {action_name} com argumento '{action_arg}'")
            print(f"Observação: {observacao_da_acao}\n")
        else:
            return f"Erro: O agente não conseguiu extrair uma Ação ou Resposta final após {i+1} iterações. Última resposta do modelo: {response_text}"

    return "Erro: Limite máximo de iterações atingido sem uma resposta final do agente."

def iniciar_conversacao_com_agente():
    print("--- Agente de Inventário Interativo ---")
    print("Digite sua pergunta sobre o inventário, ou digite 'sair' para encerrar")
    print("-"*50)

    while True:
        pergunta_usuario = input("\nVocê: ")

        if pergunta_usuario.lower().strip() == 'sair':
            print("Encerrando a conversa. Até logo")
            break

        print("\nAgente: Processando...")
        try:

            resposta_agente = run_react_agent(pergunta=pergunta_usuario)
            print(f"\nAgente: {resposta_agente}")

        except Exception as e:

            print(f"\nAgente: Ocorreu um erro ao processar sua pergunta: {e}")
            print("Por favor, tente novamente ou digite 'sair'.")


if __name__ == "__main__":
    
    # Interação 1: Consulta Estoque
    # pergunta_1 = "Quantos mouses gamers estão no inventário?"
    # print(f"**Interação 1:{pergunta_1}**")
    # resposta_1 = run_react_agent(pergunta_1)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 1:** {resposta_1}\n")
    # print("\n" + "=="*50 + "\n")

    # Interação 2: Consultar Preço
    # pergunta_2 = "Qual o preço de um headset?"
    # print(f"\n**Interação 2: {pergunta_2}**")
    # resposta_2 = run_react_agent(pergunta_2)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 2:** {resposta_2}\n")
    # print("\n" + "="*80 + "\n")

    # Interação 3: Item não encontrado no estoque
    # pergunta_3 = "Temos cadeiras em estoque?"
    # print(f"\n**Interação 3: {pergunta_3}**")
    # resposta_3 = run_react_agent(pergunta_3)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 3:** {resposta_3}\n")
    # print("\n" + "="*80 + "\n")

    # Interação 4: Consultar Preço
    # pergunta_4 = "Qual o valor da impressora?"
    # print(f"\n**Interação 2: {pergunta_4}**")
    # resposta_4 = run_react_agent(pergunta_4)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 4:** {resposta_4}\n")
    # print("\n" + "="*80 + "\n")


    # Interação 5: Encontrar o Produto Mais Caro (A nova funcionalidade!)
    # pergunta_5 = "Qual é o produto mais caro?"
    # print(f"\n**Interação 5: {pergunta_5}**")
    # resposta_5 = run_react_agent(pergunta_5)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 5:** {resposta_5}\n")
    # print("\n--- Fim das Interações ---")

    # Interação 6: Calcular Valor Total da Lista
    # pergunta_6 = "Qual o valor de um teclado, mouse gamer, monitor?"
    # print(f"\n**Interação 5: {pergunta_6}**")
    # resposta_6 = run_react_agent(pergunta_6)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 6:** {resposta_6}\n")
    # print("\n--- Fim das Interações ---")

    # Interação 7: Calcular Valor Total da Lista
    # pergunta_7 = "Qual o valor de um headset e uma cadeira?"
    # print(f"\n**Interação 5: {pergunta_7}**")
    # resposta_7 = run_react_agent(pergunta_7)
    # print(f"\n**RESPOSTA FINAL DO AGENTE 7:** {resposta_7}\n")
    # print("\n--- Fim das Interações ---")

    iniciar_conversacao_com_agente()