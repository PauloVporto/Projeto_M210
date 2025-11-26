import streamlit as st
import numpy as np
from simplex import SimplexSolver

st.set_page_config(page_title="Simplex Interativo - M210", layout="centered")
st.title("📊 Método Simplex - Trabalho Prático M210")
st.write("Simulador educacional do método Simplex Tableau (sem bibliotecas externas).")

st.sidebar.header("Entrada de dados")
n = st.sidebar.number_input("Número de variáveis", 2, 4, 3)
m = st.sidebar.number_input("Número de restrições", 1, 5, 3)

st.subheader("Coeficientes da Função Objetivo (Max Z = ...)")
c = np.zeros(n)
for i in range(n):
    c[i] = st.number_input(f"Coeficiente c{i+1}", value=1.0, step=0.1)

st.subheader("Coeficientes das Restrições (A x ≤ b)")
A = np.zeros((m, n))
b = np.zeros(m)
for i in range(m):
    st.text(f"Restrição {i+1}:")
    cols = st.columns(n + 1)
    for j in range(n):
        A[i, j] = cols[j].number_input(f"A{i+1}{j+1}", value=1.0, step=0.1, key=f"A{i}{j}")
    b[i] = cols[-1].number_input(f"b{i+1}", value=10.0, step=0.1, key=f"b{i}")

if st.button("Resolver"):
    solver = SimplexSolver(A, b, c)
    resultado = solver.resolver()
    st.success("✅ Solução ótima encontrada!")
    st.write(f"**Lucro ótimo (Z):** {resultado['z_otimo']:.2f}")
    st.write("**Valores das variáveis:**")
    for i, val in enumerate(resultado["solucao"], start=1):
        st.write(f"x{i} = {val:.2f}")
    st.write("**Preços-sombra:**")
    for i, val in enumerate(resultado["preco_sombra"], start=1):
        st.write(f"Restrição {i}: {val:.2f}")
    st.subheader("Tableau Final")
    st.dataframe(np.round(resultado["tableau_final"], 3))
