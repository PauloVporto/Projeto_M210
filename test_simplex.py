import numpy as np
import pytest
from simplex import SimplexSolver


def test_exemplo_basico():
    """Problema clássico de 2 variáveis"""
    # Max Z = 3x1 + 5x2
    # 1x1 + 2x2 <= 8
    # 3x1 + 2x2 <= 12
    # x1, x2 >= 0

    A = np.array([
        [1, 2],
        [3, 2]
    ])
    b = np.array([8, 12])
    c = np.array([3, 5])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    # Ótimo: x1=2, x2=3, Z=21
    np.testing.assert_almost_equal(result["solucao"][0], 2, decimal=2)
    np.testing.assert_almost_equal(result["solucao"][1], 3, decimal=2)
    np.testing.assert_almost_equal(result["z_otimo"], 21, decimal=2)


def test_tres_variaveis():
    """Problema 3 variáveis simples"""
    # Max Z = 2x1 + 3x2 + 4x3
    # x1 + 2x2 + x3 <= 20
    # 2x1 + x2 + 3x3 <= 30
    # x1, x2, x3 >= 0

    A = np.array([
        [1, 2, 1],
        [2, 1, 3]
    ])
    b = np.array([20, 30])
    c = np.array([2, 3, 4])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    assert result["z_otimo"] > 0
    assert np.all(result["solucao"] >= 0)


def test_restricao_redundante():
    """Teste com restrição redundante"""
    # Max Z = 5x1 + 4x2
    # 6x1 + 4x2 <= 24
    # x1 + 2x2 <= 6
    # -x1 + x2 <= 1
    A = np.array([
        [6, 4],
        [1, 2],
        [-1, 1]
    ])
    b = np.array([24, 6, 1])
    c = np.array([5, 4])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    # Esperado: x1≈3, x2≈1.5, Z≈21
    np.testing.assert_almost_equal(result["solucao"][0], 3, decimal=1)
    np.testing.assert_almost_equal(result["solucao"][1], 1.5, decimal=1)
    np.testing.assert_almost_equal(result["z_otimo"], 21, decimal=1)


def test_zero_nas_restricoes():
    """Teste com zeros nos coeficientes"""
    # Max Z = 2x1 + 3x2
    # 2x1 <= 10
    # 3x2 <= 15
    A = np.array([
        [2, 0],
        [0, 3]
    ])
    b = np.array([10, 15])
    c = np.array([2, 3])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    # Esperado: x1=5, x2=5, Z=25
    np.testing.assert_almost_equal(result["solucao"][0], 5, decimal=1)
    np.testing.assert_almost_equal(result["solucao"][1], 5, decimal=1)
    np.testing.assert_almost_equal(result["z_otimo"], 25, decimal=1)


def test_degenerado():
    """Caso degenerado (ótimo ocorre em vértice com variável zero)"""
    # Max Z = 4x1 + 3x2
    # 2x1 + x2 <= 8
    # x1 + 2x2 <= 8
    A = np.array([
        [2, 1],
        [1, 2]
    ])
    b = np.array([8, 8])
    c = np.array([4, 3])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    # Ótimo: x1=3.2, x2=1.6, Z≈17.6
    np.testing.assert_almost_equal(result["z_otimo"], 17.6, decimal=1)


def test_variavel_inativa():
    """Variável que não contribui para o resultado ótimo"""
    # Max Z = 2x1 + 3x2 + 0x3
    # x1 + 2x2 + x3 <= 10
    # 3x1 + x2 + 2x3 <= 15
    A = np.array([
        [1, 2, 1],
        [3, 1, 2]
    ])
    b = np.array([10, 15])
    c = np.array([2, 3, 0])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    # x3 deve ser zero no ótimo
    assert abs(result["solucao"][2]) < 1e-6


def test_matriz_identidade():
    """Teste rápido com restrições simples e independentes"""
    # Max Z = x1 + x2 + x3
    # x1 <= 3
    # x2 <= 4
    # x3 <= 5
    A = np.eye(3)
    b = np.array([3, 4, 5])
    c = np.array([1, 1, 1])

    solver = SimplexSolver(A, b, c)
    result = solver.resolver()

    np.testing.assert_almost_equal(result["solucao"], np.array([3, 4, 5]), decimal=1)
    np.testing.assert_almost_equal(result["z_otimo"], 12, decimal=1)


def test_ilimitado():
    """Problema sem solução limitada deve gerar erro"""
    # Max Z = x1 + x2
    # x1 - x2 <= 10
    # x1, x2 >= 0
    A = np.array([[1, -1]])
    b = np.array([10])
    c = np.array([1, 1])

    solver = SimplexSolver(A, b, c)
    with pytest.raises(ValueError):
        solver.resolver()
