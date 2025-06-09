import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import date, timedelta
import re
import numpy as np

# --- FUNÇÕES DE ANÁLISE ---

@st.cache_data
def carregar_dados_csv(arquivo_enviado):
    """Carrega e prepara os dados do CSV, tratando nomes de colunas."""
    try:
        df = pd.read_csv(arquivo_enviado)
        df.columns = df.columns.str.lower()
        if 'date' in df.columns: pass
        elif 'unnamed: 0' in df.columns: df.rename(columns={'unnamed: 0': 'date'}, inplace=True)
        else: return None
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(by='date', inplace=True)
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
    X_hist = dados[['date']].copy()
    X_hist['date_ordinal'] = X_hist['date'].map(date.toordinal)
    tendencia_historica = modelo.predict(X_hist[['date_ordinal']])
    fig = go.Figure(data=[
        go.Scatter(x=dados['date'], y=dados['close'], mode='lines', name='Preço Histórico'),
        go.Scatter(x=dados['date'], y=tendencia_historica, mode='lines', name='Tendência (Regressão)'),
        go.Scatter(x=df_previsao['date'], y=df_previsao['previsao_close'], mode='lines', name='Previsão Futura')
    ], layout=go.Layout(title=f'Histórico e Previsão para {dados["ticker"].iloc[0]}'))
    st.plotly_chart(fig, use_container_width=True)

def mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao):
    """Exibe a recomendação de forma mais direta."""
    st.subheader("Com base na previsão, minha recomendação é:")
    variacao_percentual = ((preco_previsto - ultimo_preco) / ultimo_preco) * 100
    frase_mudanca = f"A mudança prevista para os próximos {dias_previsao} dias é de **{variacao_percentual:.2f}%**"
    if variacao_percentual > 5: frase_recomendacao = "logo, **recomendo a compra**."
    elif variacao_percentual < -5: frase_recomendacao = "logo, **recomendo a venda**."
    else: frase_recomendacao = "logo, **não recomendo uma nova operação** no momento (manter posição)."
    st.markdown(f"{frase_mudanca}, {frase_recomendacao}")
    st.warning("Lembre-se, esta é uma análise simplificada.")

def mostrar_estatisticas(dados):
    """Calcula e exibe estatísticas básicas sobre a ação."""
    st.subheader(f"Estatísticas para {dados['ticker'].iloc[0]}:")
    preco_max = dados['close'].max()
    data_max = dados.loc[dados['close'].idxmax()]['date'].strftime('%d/%m/%Y')
    preco_min = dados['close'].min()
    data_min = dados.loc[dados['close'].idxmin()]['date'].strftime('%d/%m/%Y')
    ultimo_preco = dados['close'].iloc[-1]
    retorno_periodo = ((ultimo_preco / dados['close'].iloc[0]) - 1) * 100
    st.markdown(f"""
    - **Último Preço:** ${ultimo_preco:,.2f}
    - **Preço Máximo:** ${preco_max:,.2f} (em {data_max})
    - **Preço Mínimo:** ${preco_min:,.2f} (em {data_min})
    - **Retorno no Período:** `{retorno_periodo:.2f}%`
    """)

# --- NOVA FUNÇÃO DE EXPLICAÇÃO ---
def mostrar_explicacoes_metricas():
    """Cria um expansor com a explicação das principais métricas financeiras."""
    with st.expander("☝🏻🤓 O que esses valores significam?"):
        st.markdown("""
        - **Retorno Esperado (Anual):** É o quanto, em média, se espera que a carteira ou o ativo renda ao longo de um ano, com base nos dados históricos.

        - **Risco (Volatilidade Anual):** Mede o "sobe e desce" da carteira. Um valor de risco mais alto significa que o valor da carteira/ativo tende a oscilar mais, tornando o investimento mais imprevisível.

        - **Índice de Sharpe:** É a principal métrica para avaliar a qualidade de um investimento. Ele mede o retorno que você obteve para cada unidade de risco que correu. Um índice de Sharpe mais alto é sempre melhor.
            - **< 1.0:** O retorno pode não estar compensando o risco assumido.
            - **> 1.0:** Geralmente considerado um bom desempenho, onde o retorno compensa o risco.
        """)

def mostrar_sharpe_ratio(dados, taxa_livre_risco=0):
    """Calcula e exibe o Índice de Sharpe anualizado para um único ativo."""
    st.subheader(f"Análise de Risco x Retorno (Índice de Sharpe) para {dados['ticker'].iloc[0]}")
    retornos = dados['close'].pct_change().dropna()
    sharpe_ratio = (retornos.mean() - taxa_livre_risco) / retornos.std() * np.sqrt(252)
    st.metric(label="Índice de Sharpe Anualizado", value=f"{sharpe_ratio:.2f}")
    if sharpe_ratio < 1: st.warning("O retorno pode não estar compensando o risco corrido.")
    else: st.success("O ativo apresentou um bom retorno para o nível de risco.")
    mostrar_explicacoes_metricas() # Adiciona as explicações

def mostrar_grafico_comparativo(dados_completos, tickers):
    """Cria e exibe um gráfico comparando a performance normalizada de vários tickers."""
    st.subheader(f"Gráfico Comparativo de Performance para {', '.join(tickers)}")
    carteira_df = dados_completos[dados_completos['ticker'].isin(tickers)]
    tabela_precos = carteira_df.pivot(index='date', columns='ticker', values='close').dropna()
    df_normalizado = (tabela_precos / tabela_precos.iloc[0]) * 100
    fig = go.Figure()
    for ticker in df_normalizado.columns:
        fig.add_trace(go.Scatter(x=df_normalizado.index, y=df_normalizado[ticker], mode='lines', name=ticker))
    fig.update_layout(title="Performance Normalizada (Base 100)", xaxis_title='Data', yaxis_title='Performance')
    st.plotly_chart(fig, use_container_width=True)

def calcular_e_mostrar_portfolio_otimo(dados_completos, tickers_selecionados, num_simulacoes=10000):
    """Realiza a otimização de portfólio e exibe visualizações avançadas."""
    st.subheader(f"Análise de Carteira Ótima para {', '.join(tickers_selecionados)}")
    carteira_df = dados_completos[dados_completos['ticker'].isin(tickers_selecionados)]
    tabela_precos = carteira_df.pivot(index='date', columns='ticker', values='close').dropna()
    if tabela_precos.shape[0] < 2:
        st.error("Não há dados históricos suficientes para os tickers selecionados para realizar a análise.")
        return
    retornos_diarios = tabela_precos.pct_change().dropna()
    resultados_simulacao, pesos_aleatorios = [], []
    with st.spinner(f"Realizando {num_simulacoes} simulações de Monte Carlo..."):
        for _ in range(num_simulacoes):
            pesos = np.random.random(len(tickers_selecionados))
            pesos /= np.sum(pesos)
            pesos_aleatorios.append(pesos)
            retorno = np.sum(retornos_diarios.mean() * pesos) * 252
            volatilidade = np.sqrt(np.dot(pesos.T, np.dot(retornos_diarios.cov() * 252, pesos)))
            resultados_simulacao.append([retorno, volatilidade, retorno / volatilidade])
    simulacao_df = pd.DataFrame(resultados_simulacao, columns=['retorno', 'volatilidade', 'sharpe'])
    max_sharpe_port = simulacao_df.loc[simulacao_df['sharpe'].idxmax()]
    min_vol_port = simulacao_df.loc[simulacao_df['volatilidade'].idxmin()]
    
    st.markdown("#### Fronteira Eficiente: O Universo de Possibilidades")
    fig_fronteira = go.Figure()
    fig_fronteira.add_trace(go.Scatter(x=simulacao_df['volatilidade'], y=simulacao_df['retorno'], mode='markers', marker=dict(color=simulacao_df['sharpe'], showscale=True, colorscale='Viridis', colorbar=dict(title='Índice de Sharpe'))))
    fig_fronteira.add_trace(go.Scatter(x=[max_sharpe_port['volatilidade'], min_vol_port['volatilidade']], y=[max_sharpe_port['retorno'], min_vol_port['retorno']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Carteiras Otimizadas'))
    fig_fronteira.update_layout(title='Fronteira Eficiente de Markowitz', xaxis_title='Risco (Volatilidade Anual)', yaxis_title='Retorno Anual', showlegend=False)
    st.plotly_chart(fig_fronteira, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Detalhes das Carteiras Otimizadas")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Carteira de Risco Mínimo")
        pesos_min_vol = pesos_aleatorios[simulacao_df['volatilidade'].idxmin()]
        fig_min_vol = go.Figure(data=[go.Pie(labels=tickers_selecionados, values=pesos_min_vol, hole=.4, textinfo='label+percent')])
        fig_min_vol.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_min_vol, use_container_width=True)
        st.info(f"**Retorno:** `{(min_vol_port['retorno']*100):.2f}%` | **Risco:** `{(min_vol_port['volatilidade']*100):.2f}%`")
    with col2:
        st.markdown("##### Carteira Ótima (Max Sharpe)")
        pesos_max_sharpe = pesos_aleatorios[simulacao_df['sharpe'].idxmax()]
        fig_max_sharpe = go.Figure(data=[go.Pie(labels=tickers_selecionados, values=pesos_max_sharpe, hole=.4, textinfo='label+percent')])
        fig_max_sharpe.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_max_sharpe, use_container_width=True)
        st.success(f"**Retorno:** `{(max_sharpe_port['retorno']*100):.2f}%` | **Risco:** `{(max_sharpe_port['volatilidade']*100):.2f}%`")
    
    mostrar_explicacoes_metricas() # Adiciona as explicações


# --- CÉREBRO DO CHATBOT ---
def extrair_tickers(prompt):
    """Usa RegEx para encontrar TODOS os tickers na pergunta."""
    return re.findall(r'\b([A-Z]{1,5}\d{0,2})\b', prompt.upper())

def interpretar_pergunta_acao(pergunta):
    """Interpreta a ação que o usuário quer fazer."""
    pergunta = pergunta.lower()
    if re.search(r'markowitz|carteira|portfólio|portfolio|otima|ótima', pergunta): return "analisar_carteira"
    if re.search(r'comparar|comparativo| contra | vs ', pergunta): return "comparar_grafico"
    if re.search(r'sharpe|risco', pergunta): return "mostrar_sharpe"
    if re.search(r'gráfico|grafico|previsão|previsao', pergunta): return "mostrar_grafico"
    elif re.search(r'recomendação|recomendacao', pergunta): return "dar_recomendacao"
    elif re.search(r'tabela|dados|histórico|historico', pergunta): return "mostrar_tabela"
    elif re.search(r'estatísticas|estatisticas|resumo', pergunta): return "mostrar_estatisticas"
    elif re.search(r'última|ultima|recente|final', pergunta): return "mostrar_ultima_data"
    elif re.search(r'olá|ola|oi|ajuda', pergunta): return "saudacao"
    return "desconhecido"


# --- INTERFACE PRINCIPAL DO STREAMLIT (com lógica do chat modificada) ---
st.set_page_config(page_title="Chatbot de Análise de Ações", layout="wide")
st.title("🤖 Chatbot Analisador de Ações Inteligente")

if "messages" not in st.session_state: st.session_state.messages = []
if "ticker_atual" not in st.session_state: st.session_state.ticker_atual = None

with st.sidebar:
    st.header("Configurações")
    arquivo_enviado = st.file_uploader("1. Envie seu arquivo CSV:", type=['csv'])
    dias_previsao = st.slider("2. Dias para Previsão Futura:", 1, 30, 7)

dados_completos = None
if arquivo_enviado:
    dados_completos = carregar_dados_csv(arquivo_enviado)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pergunte sobre uma ou mais ações..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if dados_completos is None:
            resposta = "Por favor, primeiro envie um arquivo CSV com os dados na barra lateral."
            st.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})
        else:
            tickers_encontrados = extrair_tickers(prompt)
            acao = interpretar_pergunta_acao(prompt)
            lista_tickers_validos_no_csv = dados_completos['ticker'].unique()
            
            contexto_mudou = False
            if tickers_encontrados:
                primeiro_ticker_valido = next((t for t in tickers_encontrados if t in lista_tickers_validos_no_csv), None)
                if primeiro_ticker_valido and primeiro_ticker_valido != st.session_state.ticker_atual:
                    st.session_state.ticker_atual = primeiro_ticker_valido
                    contexto_mudou = True

            # --- INÍCIO DA LÓGICA MODIFICADA ---

            # CASO 1: Análise de múltiplos tickers (carteira ou comparação)
            if acao in ["analisar_carteira", "comparar_grafico"]:
                tickers_validos_na_pergunta = [t for t in tickers_encontrados if t in lista_tickers_validos_no_csv]
                if len(tickers_validos_na_pergunta) < 2:
                    st.markdown(f"Para esta análise, por favor, mencione pelo menos 2 tickers válidos na sua pergunta.")
                else:
                    if acao == "analisar_carteira":
                        calcular_e_mostrar_portfolio_otimo(dados_completos, tickers_validos_na_pergunta)
                    elif acao == "comparar_grafico":
                        mostrar_grafico_comparativo(dados_completos, tickers_validos_na_pergunta)
            
            # CASO 2: O usuário muda o ticker mas NÃO especifica uma ação clara (ex: digita só "TTWO")
            elif contexto_mudou and acao in ["desconhecido", "saudacao"]:
                st.markdown(f"Ok, mudei o foco da análise para **{st.session_state.ticker_atual}**. O que gostaria de saber?")

            # CASO 3: Há uma ação clara e um ticker em foco (seja ele novo ou antigo)
            elif acao not in ["saudacao", "desconhecido"] and st.session_state.ticker_atual:
                dados_ticker = dados_completos[dados_completos['ticker'] == st.session_state.ticker_atual].copy()
                
                if contexto_mudou: # Opcional: Adiciona uma pequena confirmação antes do resultado
                    st.markdown(f"Analisando **{st.session_state.ticker_atual}**...")

                if acao == "mostrar_sharpe": mostrar_sharpe_ratio(dados_ticker)
                elif acao == "mostrar_grafico":
                    ultimo_preco, preco_previsto, modelo, df_previsao = obter_previsao(dados_ticker, dias_previsao)
                    mostrar_grafico_previsao(dados_ticker, modelo, df_previsao)
                elif acao == "dar_recomendacao":
                    ultimo_preco, preco_previsto, modelo, df_previsao = obter_previsao(dados_ticker, dias_previsao)
                    mostrar_recomendacao(ultimo_preco, preco_previsto, dias_previsao)
                elif acao == "mostrar_estatisticas": mostrar_estatisticas(dados_ticker)
                elif acao == "mostrar_tabela": st.dataframe(dados_ticker)
            
            # CASO 4: Saudação geral
            elif acao == "saudacao":
                st.markdown(f"Olá! Sou seu assistente de análises financeiras!\n Você pode me perguntar sobre: Previsão/Gráfico, Recomendação, Risco/Sharpe, Estatísticas, Comparação, Carteira Ótima, Dados Brutos\n\n Ainda estou em desenvolvimento, então é possível que alguns pontos ainda não estejam 100%!\nObrigado pela coloboração!") 
            
            # CASO 5: Fallback - Não entendeu ou não há ticker em foco
            else: 
                st.markdown("Não sei sobre qual ação você quer conversar. Por favor, inclua o ticker na sua pergunta (ex: 'Qual a previsão para a TTWO?').")

            
