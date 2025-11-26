import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simplex import SimplexSolver
from prettytable import PrettyTable
from io import StringIO
from contextlib import redirect_stdout

# ======================================================
# Funções auxiliares para mostrar o Simplex em formato tableau
# ======================================================

def simplex_tableau_verbose(c, A, b):
    """Monta o tableau inicial do método Simplex (forma padrão Max)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape  # m = nº restrições, n = nº variáveis de decisão

    # Tamanho do tableau: m linhas de restrições + 1 linha de Z
    # Colunas: n variáveis + m folgas + 1 termo independente (b)
    tableau = np.zeros((m + 1, n + m + 1), dtype=float)

    # Parte das variáveis de decisão
    tableau[:m, :n] = A
    # Identidade para variáveis de folga
    tableau[:m, n:n + m] = np.eye(m)
    # Coluna b (lado direito)
    tableau[:m, -1] = b
    # Linha da função objetivo (coeficientes negativos para Max)
    tableau[-1, :n] = -np.array(c, dtype=float)
    return tableau


def mostrar_tableau(tableau):
    """Mostra o tableau com nomes de linhas e colunas (estilo PrettyTable)."""
    m = tableau.shape[0] - 1          # nº restrições
    total_cols = tableau.shape[1]
    n = total_cols - m - 1            # nº variáveis de decisão

    # Nomes das colunas: x1..xn, s1..sm, b
    colunas = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)] + ["b"]

    t = PrettyTable()
    t.field_names = ["Linha"] + colunas

    for idx, row in enumerate(tableau):
        if idx < m:
            nome_linha = f"R{idx+1}"
        else:
            nome_linha = "Z"
        t.add_row([nome_linha] + [f"{val:.2f}" for val in row])
    print(t)


def simplex_verbose(c, A, b):
    """
    Executa o método Simplex (tableau) e imprime todas as iterações
    em formato de tabela ASCII (PrettyTable).
    """
    tableau = simplex_tableau_verbose(c, A, b)
    m, n = len(A), len(A[0])
    iteracao = 0

    print("Tableau inicial:")
    mostrar_tableau(tableau)

    # Enquanto houver coeficiente negativo na linha de Z (colunas das variáveis)
    while any(tableau[-1, :-1] < 0):
        iteracao += 1
        print("\n" + "=" * 70)
        print(f"Iteração {iteracao}:")

        # Escolha da coluna pivô (variável que entra na base)
        col_pivo = int(np.argmin(tableau[-1, :-1]))

        # Verifica se o problema é ilimitado
        if np.all(tableau[:-1, col_pivo] <= 0):
            print("Problema ilimitado (sem solução ótima finita).")
            return None, None

        # Razão mínima (evita divisão por zero)
        razoes = np.full(m, np.inf)
        for i in range(m):
            if tableau[i, col_pivo] > 0:
                razoes[i] = tableau[i, -1] / tableau[i, col_pivo]
        lin_pivo = int(np.argmin(razoes))

        print(f"Coluna pivô: {col_pivo}  |  Linha pivô: {lin_pivo}")

        # Normaliza a linha pivô
        pivo = tableau[lin_pivo, col_pivo]
        tableau[lin_pivo, :] /= pivo

        # Zera as demais posições da coluna pivô
        for i in range(tableau.shape[0]):
            if i != lin_pivo:
                fator = tableau[i, col_pivo]
                tableau[i, :] -= fator * tableau[lin_pivo, :]

        print("Tableau após o pivoteamento:")
        mostrar_tableau(tableau)

    print("\n" + "=" * 70)
    print("Solução ótima encontrada:")
    mostrar_tableau(tableau)

    # Recupera os valores ótimos das variáveis de decisão
    n_vars = len(c)
    x = np.zeros(n_vars)
    for j in range(n_vars):
        col = tableau[:-1, j]
        if np.isclose(col, 0).sum() == (len(col) - 1) and np.isclose(col, 1).sum() == 1:
            lin = int(np.where(np.isclose(col, 1))[0][0])
            x[j] = tableau[lin, -1]

    z_otimo = tableau[-1, -1]

    print("\nValores ótimos das variáveis:")
    for i, val in enumerate(x, start=1):
        print(f"x{i} = {val:.4f}")
    print(f"\nValor ótimo da função objetivo Z* = {z_otimo:.4f}")

    return x, z_otimo


# =========================
# 🎨 CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Método Simplex - M210",
    page_icon="📊",
    layout="wide",
)

# CSS personalizado
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #0e1117 0%, #1b1f27 100%);
            color: #fafafa;
        }

        h1, h2, h3, h4 {
            color: #f1f1f1 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #11141a;
            color: #fafafa;
        }

        .stButton>button {
            background: linear-gradient(90deg, #00b4d8, #0077b6);
            color: white;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #0077b6, #00b4d8);
            transform: scale(1.03);
        }

        [data-testid="stMetricDelta"] {
            color: #90e0ef !important;
        }

        div[data-testid="stTabs"] button {
            background-color: #202633;
            color: #fafafa;
            border-radius: 10px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background-color: #0077b6;
            color: white;
            font-weight: bold;
        }

        .stDataFrame {border-radius: 12px; overflow: hidden;}
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧠 TÍTULO E INTRODUÇÃO
# =========================
st.title("📊 Método Simplex - Trabalho Prático M210")
st.markdown("""
### 💡 Objetivo
Resolver **Problemas de Programação Linear (PPL)** utilizando o **método Simplex Tableau**, implementado totalmente em Python, **sem bibliotecas de otimização**.

O modelo resolvido é:
\\[
\\text{Max Z = c₁x₁ + c₂x₂ + ... + cₙxₙ} \\\\
\\text{sujeito a: } A·x \\leq b, \\quad x \\geq 0
\\]
""")

st.divider()

# =========================
# ⚙️ BARRA LATERAL
# =========================
with st.sidebar:
    st.header("⚙️ Configurações do Problema")
    n = st.number_input("Número de variáveis (n)", 2, 4, 3)
    m = st.number_input("Número de restrições (m)", 1, 6, 3)
    st.markdown("---")
    st.info("Defina **n** e **m**, insira os coeficientes e clique em **Resolver**.")

# =========================
# 📈 ENTRADAS DO PROBLEMA
# =========================
st.markdown(f"### 🔢 Configuração: {int(n)} variáveis e {int(m)} restrições")

# Função objetivo
st.subheader("1️⃣ Função Objetivo – Max Z")
cols_c = st.columns(int(n))
c = np.zeros(int(n))
for i in range(int(n)):
    c[i] = cols_c[i].number_input(
        f"Coeficiente de x{i+1}",
        value=1.0,
        step=0.1,
        key=f"c_{i}"
    )
st.caption("Exemplo: se Z = 3x₁ + 5x₂ → c₁ = 3, c₂ = 5")

# Restrições
st.subheader("2️⃣ Restrições (A·x ≤ b)")
A = np.zeros((int(m), int(n)))
b = np.zeros(int(m))
for i in range(int(m)):
    st.markdown(f"**Restrição {i+1}:**")
    linha = st.columns(int(n) + 1)
    for j in range(int(n)):
        A[i, j] = linha[j].number_input(
            f"a{i+1}{j+1}",
            value=1.0,
            step=0.1,
            key=f"A_{i}_{j}"
        )
    b[i] = linha[-1].number_input(
        f"b{i+1}",
        value=10.0,
        step=0.5,
        key=f"b_{i}"
    )

st.caption("Obs.: O método assume todas as variáveis **x ≥ 0** e restrições no formato “≤”.")
st.divider()

# =========================
# 🚀 BOTÃO DE EXECUÇÃO
# =========================
if st.button("🚀 Resolver com Simplex"):
    try:
        solver = SimplexSolver(A, b, c)
        resultado = solver.resolver()
        st.success("✅ Solução ótima encontrada!")

        # Métricas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lucro ótimo (Z*)", f"{resultado['z_otimo']:.4f}")
        with col2:
            st.metric("Número de Variáveis", f"{int(n)}")

        st.markdown("---")

        # Abas de resultado
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📘 Resumo", "📈 Variáveis", "💰 Preços-sombra", "📋 Tableau"]
        )

        with tab1:
            st.markdown("#### 📘 Resumo da Solução")
            st.latex(
                "Z^* = " + " + ".join([f"{c[i]}x_{i+1}" for i in range(int(n))]) +
                f" = {resultado['z_otimo']:.4f}"
            )
            st.write("**Valores ótimos:**")
            for i, val in enumerate(resultado["solucao"], 1):
                st.write(f"➡️ x{i} = `{val:.4f}`")

        with tab2:
            st.markdown("#### 📈 Valores ótimos das variáveis")
            df_vars = pd.DataFrame({
                "Variável": [f"x{i+1}" for i in range(int(n))],
                "Valor ótimo": [round(val, 4) for val in resultado["solucao"]]
            })
            st.dataframe(df_vars, use_container_width=True)

        with tab3:
            st.markdown("#### 💰 Preços-sombra das restrições")
            st.write(
                "Cada valor indica quanto o **lucro ótimo Z** aumentaria se o lado direito "
                "b daquela restrição fosse aumentado em 1 unidade."
            )
            df_shadow = pd.DataFrame({
                "Restrição": [f"R{i+1}" for i in range(len(resultado['preco_sombra']))],
                "Preço-sombra": [round(val, 4) for val in resultado["preco_sombra"]]
            })
            st.dataframe(df_shadow, use_container_width=True)

        with tab4:
            st.markdown("#### 📋 Tableau Final do Simplex (resumo)")
            num_vars = int(n)
            num_rest = int(m)
            colunas = (
                [f"x{i+1}" for i in range(num_vars)] +
                [f"s{i+1}" for i in range(num_rest)] +
                ["b"]
            )
            linhas = [f"R{i+1}" for i in range(num_rest)] + ["Z"]
            df_tableau = pd.DataFrame(
                np.round(resultado["tableau_final"], 4),
                index=linhas,
                columns=colunas
            )
            st.dataframe(df_tableau, use_container_width=True)

            st.markdown("#### 🧮 Passo a passo do método Simplex (tableaus ASCII)")
            st.write(
                "Abaixo estão todas as iterações do método Simplex, no mesmo formato "
                "utilizado no console, com bordas e nomes de linhas/colunas."
            )

            # Captura toda a saída do simplex_verbose (tableaux, iterações, solução)
            buffer = StringIO()
            with redirect_stdout(buffer):
                simplex_verbose(c, A, b)
            texto_saida = buffer.getvalue()

            st.text_area(
                "Tableaus gerados em cada iteração:",
                value=texto_saida,
                height=500
            )

        # Gráfico para 2 variáveis
        if int(n) == 2:
            st.markdown("#### 📉 Representação Gráfica (n = 2)")
            x_opt, y_opt = resultado["solucao"][0], resultado["solucao"][1]
            max_x = max(x_opt * 1.5, 5)
            max_y = max(y_opt * 1.5, 5)

            x_vals = np.linspace(0, max_x, 100)
            y_vals = np.linspace(0, max_y, 100)
            X, Y = np.meshgrid(x_vals, y_vals)

            mascara_viavel = np.ones_like(X, dtype=bool)
            for i in range(int(m)):
                mascara_viavel &= (A[i, 0] * X + A[i, 1] * Y <= b[i] + 1e-9)

            fig, ax = plt.subplots(facecolor="#0e1117")
            ax.scatter(X[mascara_viavel], Y[mascara_viavel],
                       s=5, alpha=0.5, color="#00b4d8")
            ax.scatter([x_opt], [y_opt], s=120,
                       color="#ff9f1c", marker="o", label="Ótimo")
            ax.set_xlabel("x₁", color="white")
            ax.set_ylabel("x₂", color="white")
            ax.set_title("Região Viável e Ponto Ótimo", color="white")
            ax.grid(True, color="#222", linestyle="--", alpha=0.5)
            ax.tick_params(colors="white")
            ax.legend(facecolor="#1b1f27", edgecolor="#333", labelcolor="white")
            st.pyplot(fig)
        else:
            st.info("Gráfico disponível apenas para 2 variáveis (x₁ e x₂).")

    except Exception as e:
        st.error(f"❌ Ocorreu um erro: {e}")
        st.info("Verifique se os coeficientes estão corretos e se o problema não é ilimitado.")
else:
    st.info("Defina os coeficientes e clique em **🚀 Resolver com Simplex**.")
