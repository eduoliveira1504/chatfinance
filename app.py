import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import date, timedelta
import re
import random
import locale

# NOVO: Configura o locale para português para formatar o nome do mês
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_TIME, 'Portuguese_Brazil.1252')


# --- FUNÇÕES DE ANÁLISE ---

@st.cache_data
def carregar_dados_csv(arquivo_enviado):
    """Carrega e prepara os dados do CSV."""
    try:
        df = pd.read_csv(arquivo_enviado)
        df.columns = df.columns.str.lower()
        if 'date' in df.columns:
            pass
        elif 'unnamed: 0' in df.columns:
            df.rename(columns={'unnamed: 0': 'date'}, inplace=True)
        else:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True) # Garante que os dados estão ordenados
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def obter_previsao(dados, dias_previsao):
    """Treina o modelo e retorna os dados para a resposta."""
    df_treino = dados[['date', 'close']].copy()
    df_treino['date_ordinal'] = df_treino['date'].map(date.toordinal)
    X = df_treino[['date_ordinal']]
    y = df_treino['close']
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    data_final_historico = df_treino['date'].max()
    ultimos_dias = pd.date_range(start=data_final_historico + timedelta(days=1), periods=dias_previsao)
    df_previsao = pd.DataFrame(ultimos_dias, columns=['date'])
    df_previsao['date_ordinal'] = df_previsao['date'].map(date.toordinal)
    previsoes = modelo.predict(df_previsao[['date_ordinal']])
    df_previsao['previsao_close'] = previsoes
    
    ultimo_preco_real = dados['close'].iloc[-1]
    preco_previsto_final = df_previsao['previsao_close'].iloc[-1]
    
    return ultimo_preco_real, preco_previsto_final, modelo, df_previsao

def mostrar_grafico_previsao(dados, modelo, df_previsao):
    """Cria e exibe o gráfico de previsão com Plotly."""
    st.plotly_chart(go.Figure(
        data=[
            go.Scatter(x=dados['date'], y=dados['close'], mode='lines', name='Preço Histórico', line=dict(color='royalblue')),
            go.Scatter(x=dados['date'], y=modelo.predict(dados[['date']].apply(lambda x: x.map(date.toordinal))), mode='lines', name='Tendência (Regressão)', line=dict(color='orange', dash='dash')),
            go.Scatter(x=df_previsao['date'], y=df_previsao['previsao_close'], mode='lines', name='Previsão Futura', line=dict(color='red', dash='dot'))
        ],
        layout=go.Layout(title=f'Histórico e Previsão para {dados["ticker"].iloc[0]}', xaxis_title='Data', yaxis_title='Preço de Fechamento')
    ), use_container_width=True)

# --- FUNÇÃO DE RECOMENDAÇÃO (ATUALIZADA) ---
def mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao):
    """Exibe a recomendação de forma mais direta."""
    st.subheader("Com base na previsão, minha recomendação é:")
    variacao_percentual = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100
    
    frase_mudanca = f"A mudança prevista para os próximos {dias_previsao} dias é de **{variacao_percentual:.2f}%**"
    
    if variacao_percentual > 5:
        frase_recomendacao = "logo, **recomendo a compra**."
    elif variacao_percentual < -5:
        frase_recomendacao = "logo, **recomendo a venda**."
    else:
        frase_recomendacao = "logo, **não recomendo uma nova operação** no momento (manter posição)."
        
    st.markdown(f"{frase_mudanca}, {frase_recomendacao}")
    st.warning("Lembre-se, esta é uma análise simplificada e não uma recomendação financeira profissional.")

def mostrar_estatisticas(dados):
    """Calcula e exibe estatísticas básicas sobre a ação."""
    st.subheader("Aqui estão algumas estatísticas sobre os dados históricos:")
    preco_max = dados['close'].max()
    data_max = dados.loc[dados['close'].idxmax()]['date'].strftime('%d/%m/%Y')
    preco_min = dados['close'].min()
    data_min = dados.loc[dados['close'].idxmin()]['date'].strftime('%d/%m/%Y')
    ultimo_preco = dados['close'].iloc[-1]
    retorno_periodo = ((ultimo_preco / dados['close'].iloc[0]) - 1) * 100
    st.markdown(f"""
    - **Último Preço Registrado:** ${ultimo_preco:,.2f}
    - **Preço Máximo no Período:** ${preco_max:,.2f} (em {data_max})
    - **Preço Mínimo no Período:** ${preco_min:,.2f} (em {data_min})
    - **Retorno no Período Total:** `{retorno_periodo:.2f}%`
    """)

# --- CÉREBRO DO CHATBOT (ATUALIZADO) ---
def interpretar_pergunta(pergunta):
    """Interpreta a pergunta do usuário usando palavras-chave."""
    pergunta = pergunta.lower()
    if re.search(r'gráfico|grafico|previsão|previsao|plotar|preço|preco', pergunta):
        return "mostrar_grafico"
    elif re.search(r'recomendação|recomendacao|compro|vendo|vender|comprar|mantenho|manter', pergunta):
        return "dar_recomendacao"
    elif re.search(r'tabela|dados|histórico|historico', pergunta):
        return "mostrar_tabela"
    elif re.search(r'estatísticas|estatisticas|resumo|números|numeros|máximo|maximo|mínimo|minimo', pergunta):
        return "mostrar_estatisticas"
    # NOVA HABILIDADE:
    elif re.search(r'última|ultima|recente|final|atualização|atualizacao', pergunta):
        return "mostrar_ultima_data"
    elif re.search(r'olá|ola|oi|bom dia|boa tarde|boa noite|ajuda', pergunta):
        return "saudacao"
    return "desconhecido"

# --- INTERFACE PRINCIPAL DO STREAMLIT ---

st.set_page_config(page_title="Chatbot de Análise de Ações", layout="wide")
st.title("🤖 Chatbot Analisador de Ações")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ... (Barra lateral e exibição de histórico permanecem iguais)
with st.sidebar:
    st.header("Configurações da Análise")
    arquivo_enviado = st.file_uploader("1. Envie seu arquivo CSV:", type=['csv'])
    
    dados_completos = None
    if arquivo_enviado:
        dados_completos = carregar_dados_csv(arquivo_enviado)
        if dados_completos is not None:
            lista_tickers = sorted(dados_completos['ticker'].unique())
            ticker_selecionado = st.selectbox("2. Escolha uma Ação:", lista_tickers)
            dias_previsao = st.slider("3. Dias para Previsão Futura:", 1, 30, 7)
        else:
            st.error("O arquivo enviado não pôde ser processado.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message: st.markdown(message["content"])
        if "chart_data" in message: mostrar_grafico_previsao(message["chart_data"]["dados"], message["chart_data"]["modelo"], message["chart_data"]["df_previsao"])
        if "recommendation_data" in message: mostrar_recomendacao(message["recommendation_data"]["ultimo_preco"], message["recommendation_data"]["preco_previsto"], message["recommendation_data"]["dias_previsao"])
        if "stats_data" in message: mostrar_estatisticas(message["stats_data"])
        if "table_data" in message: st.dataframe(message["table_data"])

# --- LÓGICA DO CHATBOT (ATUALIZADA) ---
if dados_completos is not None:
    if prompt := st.chat_input(f"Pergunte sobre {ticker_selecionado}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            acao = interpretar_pergunta(prompt)
            dados_ticker = dados_completos[dados_completos['ticker'] == ticker_selecionado].copy()
            
            frases_grafico = ["Claro, preparando o gráfico para você...", "Ok, aqui está a análise visual da previsão:", "Com certeza! Veja o gráfico de preços e a tendência:"]
            frases_recomendacao = ["Analisando os números para te dar uma recomendação...", "Com base na projeção, minha sugestão é a seguinte:", "Ok, aqui vai minha recomendação sobre o que fazer:"]
            
            if acao == "desconhecido":
                # ... (mesma lógica)
                resposta = "Desculpe, não entendi. Você pode pedir pelo 'gráfico', 'recomendação', 'estatísticas' ou 'tabela de dados'."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            elif acao == "saudacao":
                # ... (mesma lógica)
                resposta = f"Olá! Sou seu assistente de análise para a ação **{ticker_selecionado}**. Como posso ajudar? Você pode pedir pelo 'gráfico', 'recomendação' ou por 'estatísticas'."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            elif dados_ticker.empty:
                # ... (mesma lógica)
                resposta = f"Não encontrei dados para '{ticker_selecionado}'."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            else:
                # --- LÓGICA DE RESPOSTA ATUALIZADA ---
                if acao in ["mostrar_grafico", "dar_recomendacao"]:
                    ultimo_preco, preco_previsto, modelo, df_previsao = obter_previsao(dados_ticker, dias_previsao)
                
                if acao == "mostrar_grafico":
                    st.markdown(random.choice(frases_grafico))
                    mostrar_grafico_previsao(dados_ticker, modelo, df_previsao)
                    st.session_state.messages.append({"role": "assistant", "chart_data": {"dados": dados_ticker, "modelo": modelo, "df_previsao": df_previsao}})

                elif acao == "dar_recomendacao":
                    st.markdown(random.choice(frases_recomendacao))
                    mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao)
                    st.session_state.messages.append({"role": "assistant", "recommendation_data": {"ultimo_preco": ultimo_preco, "preco_previsto": preco_previsto, "dias_previsao": dias_previsao}})
                
                # NOVO BLOCO DE RESPOSTA
                elif acao == "mostrar_ultima_data":
                    ultima_data = dados_ticker['date'].iloc[-1]
                    # Formata a data para um formato mais amigável, ex: "05 de junho de 2025"
                    ultima_data_formatada = ultima_data.strftime('%d de %B de %Y')
                    resposta = f"A última entrada de dados que tenho para **{ticker_selecionado}** é do dia **{ultima_data_formatada}**."
                    st.markdown(resposta)
                    st.session_state.messages.append({"role": "assistant", "content": resposta})
                
                elif acao == "mostrar_estatisticas":
                    # ... (mesma lógica)
                    mostrar_estatisticas(dados_ticker)
                    st.session_state.messages.append({"role": "assistant", "stats_data": dados_ticker})
                    
                elif acao == "mostrar_tabela":
                    # ... (mesma lógica)
                    st.subheader("Aqui estão os dados históricos:")
                    st.dataframe(dados_ticker)
                    st.session_state.messages.append({"role": "assistant", "table_data": dados_ticker})
else:
    st.info("Para começar, por favor, envie um arquivo CSV na barra lateral esquerda.")
