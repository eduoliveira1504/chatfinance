import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import date, timedelta
import re

# --- FUNÇÕES DE ANÁLISE (as mesmas de antes, sem alterações) ---

@st.cache_data
def carregar_dados_csv(arquivo_enviado):
    """Carrega e prepara os dados do CSV, tratando nomes de colunas."""
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
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def obter_previsao(dados, dias_previsao):
    """Treina o modelo e retorna os dados para a resposta."""
    df_treino = dados[['date', 'close']].copy()
    df_treino.sort_values(by='date', inplace=True)
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
    st.subheader("Aqui está o gráfico com a previsão:")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados['date'], y=dados['close'], mode='lines', name='Preço Histórico', line=dict(color='royalblue')))
    X_hist = dados[['date']].copy()
    X_hist['date_ordinal'] = X_hist['date'].map(date.toordinal)
    tendencia_historica = modelo.predict(X_hist[['date_ordinal']])
    fig.add_trace(go.Scatter(x=dados['date'], y=tendencia_historica, mode='lines', name='Tendência (Regressão)', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=df_previsao['date'], y=df_previsao['previsao_close'], mode='lines', name='Previsão Futura', line=dict(color='red', dash='dot')))
    fig.update_layout(title=f'Histórico e Previsão para {dados["ticker"].iloc[0]}', xaxis_title='Data', yaxis_title='Preço de Fechamento')
    st.plotly_chart(fig, use_container_width=True)

def mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao):
    """Exibe a recomendação de comprar, vender ou manter."""
    st.subheader("Com base na previsão, minha recomendação é:")
    variacao_percentual = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100
    if variacao_percentual > 5:
        recomendacao = f"**Comprar.** O modelo prevê uma **alta de {variacao_percentual:.2f}%** nos próximos {dias_previsao} dias."
        st.success(recomendacao)
    elif variacao_percentual < -5:
        recomendacao = f"**Vender.** O modelo prevê uma **baixa de {variacao_percentual:.2f}%** nos próximos {dias_previsao} dias."
        st.error(recomendacao)
    else:
        recomendacao = f"**Manter.** O modelo não prevê uma variação significativa ({variacao_percentual:.2f}%) para os próximos {dias_previsao} dias."
        st.info(recomendacao)
    st.warning("Lembre-se, esta é uma análise simplificada e não uma recomendação financeira profissional.")

# --- NOVO: CÉREBRO DO CHATBOT (Análise de palavras-chave) ---
def interpretar_pergunta(pergunta):
    """Interpreta a pergunta do usuário usando palavras-chave."""
    pergunta = pergunta.lower()
    if re.search(r'gráfico|grafico|previsão|previsao|plotar|preço|preco', pergunta):
        return "mostrar_grafico"
    elif re.search(r'recomendação|recomendacao|compro|vendo|vender|comprar|mantenho|manter', pergunta):
        return "dar_recomendacao"
    elif re.search(r'tabela|dados|histórico|historico', pergunta):
        return "mostrar_tabela"
    return "desconhecido"

# --- INTERFACE PRINCIPAL DO STREAMLIT ---

st.set_page_config(page_title="Chatbot de Análise de Ações", layout="wide")
st.title("🤖 Chatbot Analisador de Ações")

# Inicializa o histórico de mensagens na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRA LATERAL PARA CONFIGURAÇÕES ---
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

# Exibe mensagens antigas no início
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # A resposta pode conter diferentes tipos de conteúdo, então tratamos caso a caso
        if "content" in message:
            st.markdown(message["content"])
        if "chart_data" in message:
            mostrar_grafico_previsao(message["chart_data"]["dados"], message["chart_data"]["modelo"], message["chart_data"]["df_previsao"])
        if "recommendation_data" in message:
            mostrar_recomendacao(message["recommendation_data"]["ultimo_preco"], message["recommendation_data"]["preco_previsto"], message["recommendation_data"]["dias_previsao"])
        if "table_data" in message:
            st.dataframe(message["table_data"])

# --- LÓGICA DO CHATBOT ---
if dados_completos is not None:
    # Caixa de texto para o usuário digitar a pergunta
    if prompt := st.chat_input(f"Pergunte sobre {ticker_selecionado}..."):
        # Adiciona a pergunta do usuário ao histórico e exibe na tela
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepara a resposta do assistente
        with st.chat_message("assistant"):
            acao = interpretar_pergunta(prompt)
            dados_ticker = dados_completos[dados_completos['ticker'] == ticker_selecionado].copy()
            
            if acao == "desconhecido":
                resposta = "Desculpe, não entendi. Tente perguntar sobre 'gráfico', 'recomendação' ou 'tabela de dados'."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            elif dados_ticker.empty:
                resposta = f"Não encontrei dados para '{ticker_selecionado}'."
                st.markdown(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})

            else:
                # Se a ação precisa da previsão, calcula antes
                if acao in ["mostrar_grafico", "dar_recomendacao"]:
                    ultimo_preco, preco_previsto, modelo, df_previsao = obter_previsao(dados_ticker, dias_previsao)
                
                # Responde de acordo com a ação interpretada
                if acao == "mostrar_grafico":
                    mostrar_grafico_previsao(dados_ticker, modelo, df_previsao)
                    st.session_state.messages.append({"role": "assistant", "chart_data": {"dados": dados_ticker, "modelo": modelo, "df_previsao": df_previsao}})

                elif acao == "dar_recomendacao":
                    mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao)
                    st.session_state.messages.append({"role": "assistant", "recommendation_data": {"ultimo_preco": ultimo_preco, "preco_previsto": preco_previsto, "dias_previsao": dias_previsao}})

                elif acao == "mostrar_tabela":
                    st.subheader("Aqui estão os dados históricos:")
                    st.dataframe(dados_ticker)
                    st.session_state.messages.append({"role": "assistant", "table_data": dados_ticker})
else:
    st.info("Para começar, por favor, envie um arquivo CSV na barra lateral esquerda.")