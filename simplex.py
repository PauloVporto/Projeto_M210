import numpy as np

class SimplexSolver:
    def __init__(self, A, b, c):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.m, self.n = self.A.shape
        self.tableau = None
        self.basis = []

    def montar_tableau(self):
        identidade = np.eye(self.m)
        tableau = np.hstack([self.A, identidade, self.b.reshape(-1, 1)])
        z_row = np.hstack([-self.c, np.zeros(self.m + 1)])
        self.tableau = np.vstack([tableau, z_row])
        self.basis = list(range(self.n, self.n + self.m))

    def passo(self):
        col_pivo = np.argmin(self.tableau[-1, :-1])
        if self.tableau[-1, col_pivo] >= 0:
            return False
        col = self.tableau[:-1, col_pivo]
        if np.all(col <= 0):
            raise ValueError("Problema ilimitado.")
        razoes = np.where(col > 0, self.tableau[:-1, -1] / col, np.inf)
        lin_pivo = np.argmin(razoes)
        pivo = self.tableau[lin_pivo, col_pivo]
        self.tableau[lin_pivo, :] /= pivo
        for i in range(self.tableau.shape[0]):
            if i != lin_pivo:
                self.tableau[i, :] -= self.tableau[i, col_pivo] * self.tableau[lin_pivo, :]
        self.basis[lin_pivo] = col_pivo
        return True

    def resolver(self):
        self.montar_tableau()
        while self.passo():
            pass
        z_otimo = self.tableau[-1, -1]
        solucao = np.zeros(self.n + self.m)
        for i, var in enumerate(self.basis):
            solucao[var] = self.tableau[i, -1]
        return {
            "z_otimo": z_otimo,
            "solucao": solucao[:self.n],
            "preco_sombra": self.tableau[-1, self.n:self.n + self.m],
            "tableau_final": self.tableau,
        }
