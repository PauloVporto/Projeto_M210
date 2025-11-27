import numpy as np

class SimplexSolver:
    def __init__(self, A, b, c):
        """
        Construtor da classe.
        A: matriz das restrições (coeficientes das variáveis de decisão)
        b: vetor do lado direito das restrições
        c: vetor dos coeficientes da função objetivo (Max Z)
        """
        # Converte tudo para numpy com tipo float
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)

        # m = número de restrições, n = número de variáveis de decisão
        self.m, self.n = self.A.shape

        # Vai guardar o tableau do Simplex
        self.tableau = None

        # Lista com os índices das variáveis básicas (quais estão na base)
        self.basis = []

    def montar_tableau(self):
        """
        Monta o tableau inicial do método Simplex.

        Estrutura:
        [ A | I | b ]
        [ -c | 0 | 0 ]
        """
        # Matriz identidade para as variáveis de folga (s1, s2, ..., sm)
        identidade = np.eye(self.m)

        # Monta a parte de cima do tableau: A, identidade e coluna b
        # A: coeficientes das variáveis de decisão
        # identidade: variáveis de folga
        # b: lado direito das restrições
        tableau = np.hstack([self.A, identidade, self.b.reshape(-1, 1)])

        # Linha da função objetivo:
        # -c (negativa porque é problema de máximo)
        # zeros para as variáveis de folga e para o termo independente
        z_row = np.hstack([-self.c, np.zeros(self.m + 1)])

        # Junta as linhas das restrições com a linha da função objetivo
        self.tableau = np.vstack([tableau, z_row])

        # No início, as variáveis básicas são as de folga (s1...sm)
        # Elas ficam logo depois das variáveis de decisão (x1...xn)
        self.basis = list(range(self.n, self.n + self.m))

    def passo(self):
        """
        Executa UM passo (uma iteração) do método Simplex.

        Retorna:
        - True  -> se ainda há iterações para fazer
        - False -> se chegou na solução ótima (não há coeficientes negativos em Z)
        """
        # Escolhe a coluna pivô:
        # é a coluna com o coeficiente mais negativo na linha da função objetivo (última linha)
        col_pivo = np.argmin(self.tableau[-1, :-1])

        # Se esse coeficiente já for >= 0, não há mais como melhorar Z -> chegou no ótimo
        if self.tableau[-1, col_pivo] >= 0:
            return False

        # Coluna da variável que vai entrar na base (coluna pivô, sem a linha de Z)
        col = self.tableau[:-1, col_pivo]

        # Se todos os valores forem <= 0, o problema é ilimitado (Z cresce indefinidamente)
        if np.all(col <= 0):
            raise ValueError("Problema ilimitado.")

        # Calcula a razão mínima (teste da razão):
        # para cada linha: b_i / a_i(coluna_pivo), apenas se a_i > 0
        # onde a_i é o coeficiente da coluna pivô na linha i
        razoes = np.where(col > 0, self.tableau[:-1, -1] / col, np.inf)

        # Linha pivô é aquela com a menor razão positiva
        lin_pivo = np.argmin(razoes)

        # Elemento pivô (interseção da linha pivô com a coluna pivô)
        pivo = self.tableau[lin_pivo, col_pivo]

        # Normaliza a linha pivô: transforma o pivô em 1
        self.tableau[lin_pivo, :] /= pivo

        # Para cada outra linha, zera a coluna pivô usando combinação linear
        for i in range(self.tableau.shape[0]):
            if i != lin_pivo:
                fator = self.tableau[i, col_pivo]
                self.tableau[i, :] -= fator * self.tableau[lin_pivo, :]

        # Atualiza a base: agora, na linha 'lin_pivo', a variável básica
        # passa a ser aquela correspondente à coluna pivô
        self.basis[lin_pivo] = col_pivo

        # Ainda pode haver coeficientes negativos em Z -> continua
        return True

    def resolver(self):
        """
        Executa o método Simplex completo até encontrar a solução ótima.

        Retorna um dicionário com:
        - z_otimo: valor ótimo da função objetivo
        - solucao: valores ótimos das variáveis de decisão (x1, x2, ..., xn)
        - preco_sombra: multiplicadores das restrições (valores na linha de Z nas colunas das folgas)
        - tableau_final: tableau completo após a última iteração
        """
        # Monta o tableau inicial
        self.montar_tableau()

        # Executa iterações enquanto a função passo() retornar True
        while self.passo():
            pass  # não precisa fazer nada aqui, o trabalho é feito em passo()

        # No final, o valor ótimo de Z está na última linha, última coluna
        z_otimo = self.tableau[-1, -1]

        # Cria um vetor para guardar o valor de TODAS as variáveis (decisão + folga)
        solucao = np.zeros(self.n + self.m)

        # Para cada linha das restrições, a variável básica daquela linha
        # (armazenada em self.basis) recebe o valor do lado direito (coluna b)
        for i, var in enumerate(self.basis):
            solucao[var] = self.tableau[i, -1]

        # Retorna apenas as variáveis de decisão na solução (primeiros n elementos)
        # preço_sombra: valores da linha de Z nas colunas das variáveis de folga
        return {
            "z_otimo": z_otimo,
            "solucao": solucao[:self.n],
            "preco_sombra": self.tableau[-1, self.n:self.n + self.m],
            "tableau_final": self.tableau,
        }
