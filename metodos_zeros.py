import math
import configparser

def evaluate_function(func_str, x):
    try:
        allowed_names = {
            "math": math,
            "x": x,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e
        }
        return eval(func_str, {"__builtins__": {}}, allowed_names)
    except (NameError, SyntaxError, ZeroDivisionError) as e:
        raise ValueError(f"Erro ao avaliar a função '{func_str}' em x={x}: {e}")


def bisection_method(f, a, b, tolerance, max_iter):
    try:
        f_a = evaluate_function(f, a)
        f_b = evaluate_function(f, b)
    except ValueError as e:
        return [("Erro", str(e), "", "")], None

    if f_a * f_b >= 0:
        return [("Erro", "f(a) e f(b) devem ter sinais opostos.", "", "")], None

    iterations_data = []
    header = ["Iteração (k)", "a", "b", "x_k", "f(x_k)", "|b - a|"]
    iterations_data.append(header)

    k = 0
    while (b - a) / 2.0 > tolerance and k < max_iter:
        x_k = (a + b) / 2.0
        try:
            f_xk = evaluate_function(f, x_k)
        except ValueError as e:
            return [("Erro", f"Erro na iteração {k+1}: {str(e)}", "", "")], None

        iterations_data.append([k + 1, a, b, x_k, f_xk, b - a])

        if abs(f_xk) < tolerance:
            final_root = x_k
            return iterations_data, final_root

        f_a = evaluate_function(f, a)
        if f_a * f_xk < 0:
            b = x_k
        else:
            a = x_k
        k += 1

    final_root = (a + b) / 2.0
    return iterations_data, final_root

def linear_iterative_method(f, phi, x0, tolerance, max_iter):
    iterations_data = []
    header = ["Iteração (k)", "x_k", "x_{k+1}", "f(x_{k+1})", "|x_{k+1} - x_k|"]
    iterations_data.append(header)

    k = 0
    x_current = x0

    while k < max_iter:
        try:
            x_next = evaluate_function(phi, x_current)
            f_x_next = evaluate_function(f, x_next)
        except ValueError as e:
            return [("Erro", f"Erro na iteração {k+1}: {str(e)}", "", "")], None

        error = abs(x_next - x_current)

        iterations_data.append([k + 1, x_current, x_next, f_x_next, error])

        if error < tolerance or abs(f_x_next) < tolerance:
            return iterations_data, x_next

        # Verificar divergência
        if abs(x_next) > 1e10 or math.isnan(x_next) or math.isinf(x_next):
            return [("Erro", "Método divergiu", "", "")], None

        x_current = x_next
        k += 1

    return iterations_data, x_current

def newton_method(f, f_prime, x0, tolerance, max_iter):
    iterations_data = []
    header = ["Iteração (k)", "x_k", "f(x_k)", "f'(x_k)", "x_{k+1}", "|x_{k+1} - x_k|"]
    iterations_data.append(header)

    k = 0
    x_current = x0
    while k < max_iter:
        try:
            f_x = evaluate_function(f, x_current)
            f_prime_x = evaluate_function(f_prime, x_current)
        except ValueError as e:
            return [("Erro", f"Erro na iteração {k+1}: {str(e)}", "", "")], None

        if abs(f_prime_x) < 1e-12:
            return [("Erro", "Derivada próxima de zero.", "", "")], None

        x_next = x_current - f_x / f_prime_x
        error = abs(x_next - x_current)

        iterations_data.append([k + 1, x_current, f_x, f_prime_x, x_next, error])

        if error < tolerance or abs(f_x) < tolerance:
            return iterations_data, x_next

        # Verificar divergência
        if abs(x_next) > 1e10 or math.isnan(x_next) or math.isinf(x_next):
            return [("Erro", "Método divergiu", "", "")], None

        x_current = x_next
        k += 1

    return iterations_data, x_current

def secant_method(f, x0, x1, tolerance, max_iter):
    iterations_data = []
    header = ["Iteração (k)", "x_{k-1}", "x_k", "f(x_k)", "x_{k+1}", "|x_{k+1} - x_k|"]
    iterations_data.append(header)

    k = 0
    x_prev, x_curr = x0, x1

    while k < max_iter:
        try:
            f_prev = evaluate_function(f, x_prev)
            f_curr = evaluate_function(f, x_curr)
        except ValueError as e:
            return [("Erro", f"Erro na iteração {k+1}: {str(e)}", "", "")], None

        if abs(f_curr - f_prev) < 1e-12:
            return [("Erro", "Diferença f(xk) - f(xk-1) próxima de zero.", "", "")], None

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        error = abs(x_next - x_curr)

        iterations_data.append([k + 1, x_prev, x_curr, f_curr, x_next, error])

        if error < tolerance or abs(f_curr) < tolerance:
            return iterations_data, x_next

        # Verificar divergência
        if abs(x_next) > 1e10 or math.isnan(x_next) or math.isinf(x_next):
            return [("Erro", "Método divergiu", "", "")], None

        x_prev, x_curr = x_curr, x_next
        k += 1

    return iterations_data, x_curr

def false_position_method(f, a, b, tolerance, max_iter):
    try:
        f_a = evaluate_function(f, a)
        f_b = evaluate_function(f, b)
    except ValueError as e:
        return [("Erro", str(e), "", "")], None

    if f_a * f_b >= 0:
        return [("Erro", "f(a) e f(b) devem ter sinais opostos.", "", "")], None

    iterations_data = []
    header = ["Iteração (k)", "a", "b", "x_k", "f(x_k)", "|b - a|"]
    iterations_data.append(header)

    k = 0
    x_k = a

    while abs(b - a) > tolerance and k < max_iter:
        try:
            f_a = evaluate_function(f, a)
            f_b = evaluate_function(f, b)
        except ValueError as e:
            return [("Erro", f"Erro na iteração {k+1}: {str(e)}", "", "")], None

        if abs(f_b - f_a) < 1e-12:
            return [("Erro", "Diferença f(b) - f(a) próxima de zero.", "", "")], None

        x_prev = x_k
        x_k = (a * f_b - b * f_a) / (f_b - f_a)

        try:
            f_xk = evaluate_function(f, x_k)
        except ValueError as e:
            return [("Erro", f"Erro ao avaliar f(x_k): {str(e)}", "", "")], None

        iterations_data.append([k + 1, a, b, x_k, f_xk, abs(b - a)])

        if abs(f_xk) < tolerance:
            break

        f_a = evaluate_function(f, a)
        if f_a * f_xk < 0:
            b = x_k
        else:
            a = x_k
        k += 1

    return iterations_data, x_k

def main():
    config = configparser.ConfigParser()
    try:
        config.read('input.txt')
        
        f_str = config['Function']['f(x)']
        f_prime_str = config['Function']['f_prime(x)']
        phi_str = config['Function']['phi(x)']
        
        a = float(config['Parameters']['a'])
        b = float(config['Parameters']['b'])
        x0 = float(config['Parameters']['x0'])
        x1 = float(config['Parameters']['x1'])
        tolerance = float(config['Parameters']['tolerance'])
        max_iter = int(config['Parameters']['max_iterations'])
        
    except (KeyError, FileNotFoundError) as e:
        print(f"Erro ao ler o arquivo 'input.txt': {e}")
        return

    bisection_results, bisection_root = bisection_method(f_str, a, b, tolerance, max_iter)
    linear_iter_results, linear_iter_root = linear_iterative_method(f_str, phi_str, x0, tolerance, max_iter)
    newton_results, newton_root = newton_method(f_str, f_prime_str, x0, tolerance, max_iter)
    secant_results, secant_root = secant_method(f_str, x0, x1, tolerance, max_iter)
    false_pos_results, false_pos_root = false_position_method(f_str, a, b, tolerance, max_iter)

    with open('results.txt', 'w', encoding='utf-8') as file:
        file.write("="*80 + "\n")
        file.write("RELATÓRIO COMPARATIVO DOS MÉTODOS DE ZEROS DE FUNÇÕES\n")
        file.write("="*80 + "\n\n")
        file.write(f"Função f(x): {f_str}\n")
        file.write(f"Tolerância: {tolerance}\n")
        file.write(f"Máximo de Iterações: {max_iter}\n\n")

        all_methods = [
            ("Método da Bissecção", bisection_results, bisection_root),
            ("Método Iterativo Linear (Ponto Fixo)", linear_iter_results, linear_iter_root),
            ("Método de Newton-Raphson", newton_results, newton_root),
            ("Método da Secante", secant_results, secant_root),
            ("Método da Falsa Posição (Regula Falsi)", false_pos_results, false_pos_root)
        ]

        for name, data, root in all_methods:
            file.write("-" * 80 + "\n")
            file.write(f"{name.upper()}\n")
            file.write("-" * 80 + "\n")
            
            if not data:
                file.write("Nenhum dado de iteração gerado.\n\n")
                continue

            header = data[0]
            rows = data[1:]
            
            col_widths = [len(str(h)) for h in header]
            for row in rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(f"{cell:.10f}" if isinstance(cell, float) else str(cell)))

            header_line = " | ".join(header[i].ljust(col_widths[i]) for i in range(len(header)))
            file.write(header_line + "\n")
            file.write("-|-".join("-" * w for w in col_widths) + "\n")

            for row in rows:
                row_str = " | ".join(
                    (f"{cell:.10f}" if isinstance(cell, float) else str(cell)).ljust(col_widths[i]) 
                    for i, cell in enumerate(row)
                )
                file.write(row_str + "\n")

            if root is not None:
                file.write(f"\nRaiz Aproximada Encontrada: {root:.10f}\n\n")
            else:
                file.write("\nNão foi possível encontrar a raiz com este método.\n\n")

    print("Processo concluído. Resultados salvos em 'results.txt'.")

if __name__ == "__main__":
    main()