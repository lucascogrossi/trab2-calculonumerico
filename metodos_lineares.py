import numpy as np
import time

def carregar_matriz_A(caminho):
    return np.loadtxt(caminho)

def carregar_vetor_b(caminho):
    return np.loadtxt(caminho)

def carregar_sistema_estendido(caminho):
    M = np.loadtxt(caminho)
    A = M[:, :-1]
    b = M[:, -1]
    return A, b

def resolver_triangular_superior(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = 0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    return x

def resolver_triangular_inferior(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        soma = 0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = (b[i] - soma) / L[i, i]
    return y

def verificar_diagonal_dominante(A):
    """
    Verifica se a matriz A é estritamente diagonalmente dominante.
    Retorna True se for diagonalmente dominante, False caso contrário.
    """
    n = A.shape[0]
    for i in range(n):
        diagonal = np.abs(A[i, i])
        soma_linha = np.sum(np.abs(A[i, :])) - diagonal
        if diagonal <= soma_linha:
            return False
    return True

def criterio_sassenfeld(A):
    """
    Calcula os coeficientes de Sassenfeld (beta) para o método de Gauss-Seidel.
    Se max(beta) < 1, o método converge.
    Retorna: (converge: bool, betas: array, max_beta: float)
    """
    n = A.shape[0]
    beta = np.zeros(n)

    for i in range(n):
        if np.abs(A[i, i]) < 1e-15:
            print(f"Aviso: elemento diagonal A[{i},{i}] é zero ou muito pequeno.")
            return False, beta, float('inf')

        soma = 0
        for j in range(n):
            if j != i:
                if j < i:
                    soma += np.abs(A[i, j]) * beta[j]
                else:
                    soma += np.abs(A[i, j])

        beta[i] = soma / np.abs(A[i, i])

    max_beta = np.max(beta)
    converge = max_beta < 1.0

    return converge, beta, max_beta

def tentar_reorganizar_linhas(A, b):
    """
    Tenta reorganizar as linhas da matriz para torná-la diagonalmente dominante.
    Retorna: (A_reorg, b_reorg, sucesso)
    """
    n = A.shape[0]
    A_temp = A.copy()
    b_temp = b.copy()

    # Cria lista de linhas não utilizadas
    linhas_disponiveis = list(range(n))
    nova_ordem = []

    # Para cada coluna (posição diagonal), encontra a melhor linha
    for col in range(n):
        melhor_linha = -1
        melhor_razao = 0

        for linha in linhas_disponiveis:
            diagonal = np.abs(A_temp[linha, col])
            soma = np.sum(np.abs(A_temp[linha, :])) - diagonal

            # Verifica se esta linha pode ser dominante nesta posição
            if diagonal > soma and diagonal > melhor_razao:
                melhor_razao = diagonal
                melhor_linha = linha

        if melhor_linha != -1:
            nova_ordem.append(melhor_linha)
            linhas_disponiveis.remove(melhor_linha)
        else:
            # Não conseguiu tornar diagonalmente dominante
            return A, b, False

    # Se conseguiu encontrar uma ordem válida, reorganiza
    if len(nova_ordem) == n:
        A_reorg = A_temp[nova_ordem, :]
        b_reorg = b_temp[nova_ordem]

        # Verifica se realmente ficou diagonalmente dominante
        if verificar_diagonal_dominante(A_reorg):
            print("\n✓ Sucesso! Linhas reorganizadas para tornar matriz diagonal dominante.")
            print(f"  Nova ordem das linhas: {nova_ordem}")
            return A_reorg, b_reorg, True

    return A, b, False

def gauss_sem_pivoteamento(A, b):
    inicio = time.time()
    n = A.shape[0]
    M = A.astype(float).copy()
    c = b.astype(float).copy()

    for k in range(n - 1):
        if np.abs(M[k, k]) < 1e-15:
            print(f"Erro: pivo zero ou muito pequeno na posicao ({k},{k}).")
            print("SOLUCAO: Use um metodo com pivoteamento (parcial ou completo).")
            return None, time.time() - inicio

        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            for j in range(k, n):
                M[i, j] = M[i, j] - m * M[k, j]
            c[i] = c[i] - m * c[k]

    if np.abs(M[n-1, n-1]) < 1e-15:
        print("Erro: matriz singular ou mal condicionada.")
        print("O sistema pode nao ter solucao unica.")
        return None, time.time() - inicio

    x = resolver_triangular_superior(M, c)
    tempo = time.time() - inicio
    return x, tempo

def gauss_pivoteamento_parcial(A, b):
    inicio = time.time()
    n = A.shape[0]
    M = A.astype(float).copy()
    c = b.astype(float).copy()
    
    for k in range(n - 1):
        max_idx = k
        for i in range(k + 1, n):
            if np.abs(M[i, k]) > np.abs(M[max_idx, k]):
                max_idx = i
        
        if np.abs(M[max_idx, k]) < 1e-15:
            print("Erro: matriz singular.")
            return None, time.time() - inicio
        
        if max_idx != k:
            M[[k, max_idx]] = M[[max_idx, k]]
            c[[k, max_idx]] = c[[max_idx, k]]
        
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            for j in range(k, n):
                M[i, j] = M[i, j] - m * M[k, j]
            c[i] = c[i] - m * c[k]
    
    if np.abs(M[n-1, n-1]) < 1e-15:
        print("Erro: matriz singular.")
        return None, time.time() - inicio
    
    x = resolver_triangular_superior(M, c)
    tempo = time.time() - inicio
    return x, tempo

def gauss_pivoteamento_completo(A, b):
    inicio = time.time()
    n = A.shape[0]
    M = A.astype(float).copy()
    c = b.astype(float).copy()
    ordem = np.arange(n)
    
    for k in range(n - 1):
        max_val = 0
        max_i, max_j = k, k
        for i in range(k, n):
            for j in range(k, n):
                if np.abs(M[i, j]) > max_val:
                    max_val = np.abs(M[i, j])
                    max_i, max_j = i, j
        
        if max_val < 1e-15:
            print("Erro: matriz singular.")
            return None, time.time() - inicio
        
        if max_i != k:
            M[[k, max_i]] = M[[max_i, k]]
            c[[k, max_i]] = c[[max_i, k]]
        
        if max_j != k:
            M[:, [k, max_j]] = M[:, [max_j, k]]
            ordem[[k, max_j]] = ordem[[max_j, k]]
        
        for i in range(k + 1, n):
            m = M[i, k] / M[k, k]
            for j in range(k, n):
                M[i, j] = M[i, j] - m * M[k, j]
            c[i] = c[i] - m * c[k]
    
    if np.abs(M[n-1, n-1]) < 1e-15:
        print("Erro: matriz singular.")
        return None, time.time() - inicio
    
    x_perm = resolver_triangular_superior(M, c)
    x = np.zeros(n)
    for i in range(n):
        x[ordem[i]] = x_perm[i]
    
    tempo = time.time() - inicio
    return x, tempo

def fatoracao_lu(A, b):
    inicio = time.time()
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()

    for k in range(n - 1):
        if np.abs(U[k, k]) < 1e-15:
            print(f"Erro: pivo zero ou muito pequeno em ({k},{k}).")
            print("SOLUCAO: Use pivoteamento ou outro metodo.")
            return None, time.time() - inicio

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] = U[i, j] - L[i, k] * U[k, j]

    if np.abs(U[n-1, n-1]) < 1e-15:
        print("Erro: matriz singular ou mal condicionada.")
        return None, time.time() - inicio

    y = resolver_triangular_inferior(L, b.astype(float))
    x = resolver_triangular_superior(U, y)
    tempo = time.time() - inicio
    return x, tempo

def fatoracao_cholesky(A, b):
    inicio = time.time()
    n = A.shape[0]
    
    for i in range(n):
        for j in range(n):
            if abs(A[i, j] - A[j, i]) > 1e-10:
                print("Erro: matriz nao e simetrica.")
                return None, time.time() - inicio
    
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1):
            soma = 0
            if i == j:
                for k in range(j):
                    soma += L[i, k] ** 2
                val = A[i, i] - soma
                if val <= 0:
                    print("Erro: matriz nao e positiva definida.")
                    return None, time.time() - inicio
                L[i, j] = np.sqrt(val)
            else:
                for k in range(j):
                    soma += L[i, k] * L[j, k]
                L[i, j] = (A[i, j] - soma) / L[j, j]
    
    y = resolver_triangular_inferior(L, b.astype(float))
    x = resolver_triangular_superior(L.T, y)
    tempo = time.time() - inicio
    return x, tempo

def gauss_jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    inicio = time.time()
    n = A.shape[0]

    # Faz cópia para não modificar originais
    A_trabalho = A.astype(float).copy()
    b_trabalho = b.astype(float).copy()

    # Verifica se os elementos diagonais são não-nulos
    for i in range(n):
        if np.abs(A_trabalho[i, i]) < 1e-15:
            print(f"Erro: elemento diagonal A[{i},{i}] e zero.")
            return None, 0, None, time.time() - inicio

    # Verifica critério de convergência
    diag_dom = verificar_diagonal_dominante(A_trabalho)
    if diag_dom:
        print("Matriz e diagonalmente dominante - convergencia garantida.")
    else:
        print("AVISO: Matriz NAO e diagonalmente dominante.")
        print("Tentando reorganizar linhas para tornar diagonal dominante...")
        A_trabalho, b_trabalho, sucesso = tentar_reorganizar_linhas(A_trabalho, b_trabalho)

        if not sucesso:
            print("AVISO: Nao foi possivel reorganizar para diagonal dominante.")
            print("Convergencia nao esta garantida para Gauss-Jacobi.")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).copy()

    for k in range(max_iter):
        x_novo = np.zeros(n)

        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A_trabalho[i, j] * x[j]
            x_novo[i] = (b_trabalho[i] - soma) / A_trabalho[i, i]

        erro = np.linalg.norm(x_novo - x)

        if erro < tol:
            tempo = time.time() - inicio
            return x_novo, k + 1, erro, tempo

        if erro > 1e10 or np.any(np.isnan(x_novo)):
            print("Erro: metodo divergiu.")
            return None, k + 1, erro, time.time() - inicio

        x = x_novo

    tempo = time.time() - inicio
    print(f"Aviso: maximo de iteracoes ({max_iter}) atingido.")
    return x, max_iter, erro, tempo

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=1000):
    inicio = time.time()
    n = A.shape[0]

    # Faz cópia para não modificar originais
    A_trabalho = A.astype(float).copy()
    b_trabalho = b.astype(float).copy()

    # Verifica se os elementos diagonais são não-nulos
    for i in range(n):
        if np.abs(A_trabalho[i, i]) < 1e-15:
            print(f"Erro: elemento diagonal A[{i},{i}] e zero.")
            return None, 0, None, time.time() - inicio

    # Verifica critério de Sassenfeld (mais preciso para Gauss-Seidel)
    converge_sass, betas, max_beta = criterio_sassenfeld(A_trabalho)
    if converge_sass:
        print(f"Criterio de Sassenfeld satisfeito (max beta = {max_beta:.6f} < 1).")
        print("Convergencia garantida para Gauss-Seidel.")
    else:
        print(f"AVISO: Criterio de Sassenfeld NAO satisfeito (max beta = {max_beta:.6f}).")
        # Verifica diagonal dominante como alternativa
        diag_dom = verificar_diagonal_dominante(A_trabalho)
        if diag_dom:
            print("Matriz e diagonalmente dominante - convergencia garantida.")
        else:
            print("Tentando reorganizar linhas para tornar diagonal dominante...")
            A_trabalho, b_trabalho, sucesso = tentar_reorganizar_linhas(A_trabalho, b_trabalho)

            if sucesso:
                # Recalcula Sassenfeld com a matriz reorganizada
                converge_sass, betas, max_beta = criterio_sassenfeld(A_trabalho)
                if converge_sass:
                    print(f"Apos reorganizacao: Sassenfeld satisfeito (max beta = {max_beta:.6f}).")
            else:
                print("AVISO: Nao foi possivel reorganizar para diagonal dominante.")
                print("Convergencia nao esta garantida.")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.astype(float).copy()

    for k in range(max_iter):
        x_ant = x.copy()

        for i in range(n):
            soma = 0
            for j in range(n):
                if j != i:
                    soma += A_trabalho[i, j] * x[j]
            x[i] = (b_trabalho[i] - soma) / A_trabalho[i, i]

        erro = np.linalg.norm(x - x_ant)

        if erro < tol:
            tempo = time.time() - inicio
            return x, k + 1, erro, tempo

        if erro > 1e10 or np.any(np.isnan(x)):
            print("Erro: metodo divergiu.")
            return None, k + 1, erro, time.time() - inicio

    tempo = time.time() - inicio
    print(f"Aviso: maximo de iteracoes ({max_iter}) atingido.")
    return x, max_iter, erro, tempo

def menu():
    print("\n" + "="*50)
    print("RESOLUCAO DE SISTEMAS LINEARES")
    print("="*50)
    print("\nCarregar dados:")
    print("1 - Carregar A e b (matriz_A.txt e vetor_b.txt)")
    print("2 - Carregar matriz estendida (sistema_estendido.txt)")
    print("3 - Digitar matriz manualmente")
    print("\nMetodos diretos:")
    print("4 - Gauss sem pivoteamento")
    print("5 - Gauss com pivoteamento parcial")
    print("6 - Gauss com pivoteamento completo")
    print("7 - Fatoracao LU")
    print("8 - Fatoracao de Cholesky")
    print("\nMetodos iterativos:")
    print("9 - Gauss-Jacobi")
    print("10 - Gauss-Seidel")
    print("\n11 - Mostrar sistema atual")
    print("0 - Sair")
    print("="*50)

def main():
    A = None
    b = None
    
    while True:
        menu()
        op = input("\nEscolha uma opcao: ")
        
        if op == "0":
            print("Encerrando...")
            break
        
        elif op == "1":
            try:
                A = carregar_matriz_A("matriz_A.txt")
                b = carregar_vetor_b("vetor_b.txt")
                print(f"Sistema {A.shape[0]}x{A.shape[1]} carregado.")
            except Exception as e:
                print(f"Erro ao carregar: {e}")
        
        elif op == "2":
            try:
                A, b = carregar_sistema_estendido("sistema_estendido.txt")
                print(f"Sistema {A.shape[0]}x{A.shape[1]} carregado.")
            except Exception as e:
                print(f"Erro ao carregar: {e}")
        
        elif op == "3":
            try:
                n = int(input("Ordem do sistema: "))
                print("Digite a matriz A (linha por linha, valores separados por espaco):")
                A = np.zeros((n, n))
                for i in range(n):
                    linha = input(f"Linha {i+1}: ").split()
                    for j in range(n):
                        A[i, j] = float(linha[j])
                print("Digite o vetor b (valores separados por espaco):")
                b = np.array([float(x) for x in input().split()])
                print(f"Sistema {n}x{n} carregado.")
            except Exception as e:
                print(f"Erro: {e}")
        
        elif op == "11":
            if A is None:
                print("Nenhum sistema carregado.")
            else:
                print("\nMatriz A:")
                print(A)
                print("\nVetor b:")
                print(b)
        
        elif op in ["4", "5", "6", "7", "8", "9", "10"]:
            if A is None:
                print("Carregue um sistema primeiro.")
                continue
            
            if op == "4":
                x, tempo = gauss_sem_pivoteamento(A, b)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "5":
                x, tempo = gauss_pivoteamento_parcial(A, b)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "6":
                x, tempo = gauss_pivoteamento_completo(A, b)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "7":
                x, tempo = fatoracao_lu(A, b)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "8":
                x, tempo = fatoracao_cholesky(A, b)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "9":
                tol = float(input("Tolerancia (ex: 1e-6): "))
                max_it = int(input("Maximo de iteracoes: "))
                x, iters, erro, tempo = gauss_jacobi(A, b, tol=tol, max_iter=max_it)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Iteracoes: {iters}")
                    print(f"Erro final: {erro:.2e}")
                    print(f"Tempo: {tempo:.6f} s")
            
            elif op == "10":
                tol = float(input("Tolerancia (ex: 1e-6): "))
                max_it = int(input("Maximo de iteracoes: "))
                x, iters, erro, tempo = gauss_seidel(A, b, tol=tol, max_iter=max_it)
                if x is not None:
                    print(f"\nSolucao: {x}")
                    print(f"Iteracoes: {iters}")
                    print(f"Erro final: {erro:.2e}")
                    print(f"Tempo: {tempo:.6f} s")
        
        else:
            print("Opcao invalida.")

if __name__ == "__main__":
    main()
