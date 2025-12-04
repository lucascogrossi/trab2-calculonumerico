import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import metodos_lineares as ml
    import metodos_zeros as mz
except ImportError as e:
    print("ERRO CRÍTICO: Não foi possível importar os módulos.")
    print("Certifique-se de que 'metodos_lineares.py' e 'metodos_zeros.py' estão na mesma pasta.")
    raise e

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora Numérica")
        self.root.geometry("950x750") 

        style = ttk.Style()
        style.theme_use('clam')

        self.tab_control = ttk.Notebook(root)
        
        self.tab_sistemas = ttk.Frame(self.tab_control)
        self.tab_zeros = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.tab_sistemas, text='Sistemas Lineares')
        self.tab_control.add(self.tab_zeros, text='Zeros de Funções (Extra)')
        self.tab_control.pack(expand=1, fill="both")

        self.setup_sistemas_tab()
        self.setup_zeros_tab()

        self.A = None
        self.b = None

    def escrever_log(self, widget, texto):
        """Escreve no widget de texto mesmo se ele estiver travado (ReadOnly)."""
        widget.config(state=tk.NORMAL)  
        widget.insert(tk.END, texto)    
        widget.config(state=tk.DISABLED)
        widget.see(tk.END)              

    def limpar_log(self, widget):
        """Limpa o widget de texto mesmo se ele estiver travado."""
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.config(state=tk.DISABLED)

    # ABA 1: SISTEMAS LINEARES
    def setup_sistemas_tab(self):
        # Frame de Controles
        frame_top = tk.Frame(self.tab_sistemas, pady=10)
        frame_top.pack()

        # Botões de Carregamento
        btn_frame = tk.Frame(frame_top)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        tk.Button(btn_frame, text="Carregar Matriz A", command=self.load_A, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Carregar Vetor b", command=self.load_b, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Carregar Estendido", command=self.load_extended, width=15).pack(side=tk.LEFT, padx=5)

        # Status do Arquivo
        self.label_status = tk.Label(frame_top, text="Nenhum sistema carregado", fg="red", font=("Arial", 10, "bold"))
        self.label_status.grid(row=1, column=0, columnspan=2, pady=5)

        # Seleção de Método
        tk.Label(frame_top, text="Método:").grid(row=2, column=0, sticky="e", padx=5)
        self.combo_metodos_linear = ttk.Combobox(frame_top, values=[
            "Gauss sem Pivoteamento", 
            "Gauss com Pivoteamento Parcial", 
            "Gauss com Pivoteamento Completo", 
            "Fatoração LU", 
            "Fatoração de Cholesky",
            "Gauss-Jacobi", 
            "Gauss-Seidel"
        ], width=35, state="readonly")
        self.combo_metodos_linear.grid(row=2, column=1, sticky="w", padx=5)
        self.combo_metodos_linear.current(0)

        # Parâmetros Iterativos
        frame_params = tk.LabelFrame(frame_top, text="Parâmetros (Apenas Iterativos)")
        frame_params.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        
        tk.Label(frame_params, text="Tolerância:").pack(side=tk.LEFT, padx=5)
        self.entry_tol_lin = tk.Entry(frame_params, width=10)
        self.entry_tol_lin.insert(0, "1e-6")
        self.entry_tol_lin.pack(side=tk.LEFT)
        
        tk.Label(frame_params, text="Max Iter:").pack(side=tk.LEFT, padx=5)
        self.entry_iter_lin = tk.Entry(frame_params, width=10)
        self.entry_iter_lin.insert(0, "1000")
        self.entry_iter_lin.pack(side=tk.LEFT)

        # Botão Calcular
        tk.Button(frame_top, text="CALCULAR SISTEMA", command=self.calcular_linear, bg="#dddddd", font=("Arial", 10, "bold")).grid(row=4, column=0, columnspan=2, pady=15)

        # Área de resultado
        frame_result = tk.Frame(self.tab_sistemas)
        frame_result.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar_lin = tk.Scrollbar(frame_result)
        scrollbar_lin.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_result_linear = tk.Text(frame_result, height=15, font=("Consolas", 10), 
                                         state=tk.DISABLED,
                                         yscrollcommand=scrollbar_lin.set)
        self.txt_result_linear.pack(side=tk.LEFT, fill="both", expand=True)
        
        scrollbar_lin.config(command=self.txt_result_linear.yview)

    # ABA 2: ZEROS DE FUNÇÕES
    def setup_zeros_tab(self):
        frame_top = tk.Frame(self.tab_zeros, pady=10)
        frame_top.pack()

        # Entradas de Funções
        grid_opts = {'padx': 5, 'pady': 2, 'sticky': 'w'}
        
        tk.Label(frame_top, text="Função f(x):").grid(row=0, column=0, **grid_opts)
        self.entry_fx = tk.Entry(frame_top, width=40)
        self.entry_fx.insert(0, "x * math.log(x) - 1")
        self.entry_fx.grid(row=0, column=1, **grid_opts)

        tk.Label(frame_top, text="Derivada f'(x) [Newton]:").grid(row=1, column=0, **grid_opts)
        self.entry_dfx = tk.Entry(frame_top, width=40)
        self.entry_dfx.insert(0, "math.log(x) + 1")
        self.entry_dfx.grid(row=1, column=1, **grid_opts)

        tk.Label(frame_top, text="Phi(x) [MIL]:").grid(row=2, column=0, **grid_opts)
        self.entry_phi = tk.Entry(frame_top, width=40)
        self.entry_phi.insert(0, "math.exp(1/x)")
        self.entry_phi.grid(row=2, column=1, **grid_opts)

        # Intervalos e Chutes
        frame_vals = tk.Frame(frame_top)
        frame_vals.grid(row=3, column=0, columnspan=2, pady=10)
        
        tk.Label(frame_vals, text="a:").pack(side=tk.LEFT)
        self.entry_a = tk.Entry(frame_vals, width=6)
        self.entry_a.insert(0, "1.0")
        self.entry_a.pack(side=tk.LEFT, padx=5)

        tk.Label(frame_vals, text="b:").pack(side=tk.LEFT)
        self.entry_b = tk.Entry(frame_vals, width=6)
        self.entry_b.insert(0, "2.0")
        self.entry_b.pack(side=tk.LEFT, padx=5)

        tk.Label(frame_vals, text="x0:").pack(side=tk.LEFT)
        self.entry_x0 = tk.Entry(frame_vals, width=6)
        self.entry_x0.insert(0, "1.5")
        self.entry_x0.pack(side=tk.LEFT, padx=5)

        # Seleção de Método
        tk.Label(frame_top, text="Método:").grid(row=4, column=0, sticky="e", pady=5)
        self.combo_metodos_zeros = ttk.Combobox(frame_top, values=[
            "Bissecção", 
            "Ponto Fixo (MIL)", 
            "Newton-Raphson", 
            "Secante", 
            "Falsa Posição"
        ], state="readonly")
        self.combo_metodos_zeros.grid(row=4, column=1, sticky="w", pady=5)
        self.combo_metodos_zeros.current(0)

        # Botão Calcular
        tk.Button(frame_top, text="CALCULAR RAIZ", command=self.calcular_raiz, bg="#dddddd", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=2, pady=10)

        # Área de resultado
        frame_result = tk.Frame(self.tab_zeros)
        frame_result.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar_zeros = tk.Scrollbar(frame_result)
        scrollbar_zeros.pack(side=tk.RIGHT, fill=tk.Y)

        self.txt_result_zeros = tk.Text(frame_result, height=20, font=("Consolas", 10),
                                        state=tk.DISABLED, 
                                        yscrollcommand=scrollbar_zeros.set)
        self.txt_result_zeros.pack(side=tk.LEFT, fill="both", expand=True)
        
        scrollbar_zeros.config(command=self.txt_result_zeros.yview)

    # LÓGICA - SISTEMAS LINEARES
    def load_A(self):
        filename = filedialog.askopenfilename(title="Selecione o arquivo da Matriz A")
        if filename:
            try:
                self.A = ml.carregar_matriz_A(filename)
                self.label_status.config(text=f"Matriz A {self.A.shape} carregada", fg="orange")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar: {e}")

    def load_b(self):
        filename = filedialog.askopenfilename(title="Selecione o arquivo do Vetor b")
        if filename:
            try:
                self.b = ml.carregar_vetor_b(filename)
                if self.A is not None:
                    self.label_status.config(text=f"Sistema Completo: A{self.A.shape}, b({len(self.b)})", fg="green")
                else:
                    self.label_status.config(text=f"Vetor b ({len(self.b)}) carregado", fg="orange")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar: {e}")

    def load_extended(self):
        filename = filedialog.askopenfilename(title="Selecione o Sistema Estendido")
        if filename:
            try:
                self.A, self.b = ml.carregar_sistema_estendido(filename)
                self.label_status.config(text=f"Sistema Estendido Carregado: {self.A.shape}", fg="green")
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar: {e}")

    def calcular_linear(self):
        if self.A is None or self.b is None:
            messagebox.showwarning("Aviso", "Você precisa carregar a Matriz A e o Vetor b primeiro!")
            return

        metodo = self.combo_metodos_linear.get()
        
        self.limpar_log(self.txt_result_linear)
        self.escrever_log(self.txt_result_linear, f"Método: {metodo}\n\n")

        try:
            res = None
            tempo = 0
            
            # --- MÉTODOS DIRETOS ---
            if metodo == "Gauss sem Pivoteamento":
                res, tempo = ml.gauss_sem_pivoteamento(self.A, self.b)
            
            elif metodo == "Gauss com Pivoteamento Parcial":
                res, tempo = ml.gauss_pivoteamento_parcial(self.A, self.b)
            
            elif metodo == "Gauss com Pivoteamento Completo":
                res, tempo = ml.gauss_pivoteamento_completo(self.A, self.b)
            
            elif metodo == "Fatoração LU":
                res, tempo = ml.fatoracao_lu(self.A, self.b)
            
            elif metodo == "Fatoração de Cholesky":
                res, tempo = ml.fatoracao_cholesky(self.A, self.b)
            
            # --- MÉTODOS ITERATIVOS ---
            elif metodo in ["Gauss-Jacobi", "Gauss-Seidel"]:
                try:
                    tol = float(self.entry_tol_lin.get())
                    max_it = int(self.entry_iter_lin.get())
                except ValueError:
                    messagebox.showerror("Erro", "Tolerância ou Máx Iterações inválidos.")
                    return

                if metodo == "Gauss-Jacobi":
                    res, k, erro, tempo = ml.gauss_jacobi(self.A, self.b, tol=tol, max_iter=max_it)
                else:
                    res, k, erro, tempo = ml.gauss_seidel(self.A, self.b, tol=tol, max_iter=max_it)
                
                # Exibe detalhes iterativos
                if res is not None:
                    self.escrever_log(self.txt_result_linear, f"Iterações: {k}\n")
                    self.escrever_log(self.txt_result_linear, f"Erro Final: {erro:.2e}\n")
            
            # EXIBIÇÃO DO RESULTADO FINAL
            if res is not None:
                self.escrever_log(self.txt_result_linear, f"Solução: {res}\n")
                self.escrever_log(self.txt_result_linear, f"Tempo: {tempo:.6f} s\n")
            else:
                self.escrever_log(self.txt_result_linear, "O método falhou ou divergiu.\nConsulte o console para mais detalhes de erro.")

        except Exception as e:
            self.escrever_log(self.txt_result_linear, f"\nERRO DE EXECUÇÃO:\n{str(e)}")

    # LÓGICA - ZEROS DE FUNÇÕES
    def calcular_raiz(self):
        metodo = self.combo_metodos_zeros.get()
        f_str = self.entry_fx.get()
        df_str = self.entry_dfx.get()
        phi_str = self.entry_phi.get()

        try:
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            x0 = float(self.entry_x0.get())
        except ValueError:
            messagebox.showerror("Erro", "Verifique se a, b e x0 são números válidos.")
            return

        tol = 1e-6
        max_iter = 100

        self.limpar_log(self.txt_result_zeros)
        self.escrever_log(self.txt_result_zeros, f"Método: {metodo}\nFunção: {f_str}\n")
        self.escrever_log(self.txt_result_zeros, "="*85 + "\n")

        log = []
        raiz = None

        try:
            if metodo == "Bissecção":
                log, raiz = mz.bisection_method(f_str, a, b, tol, max_iter)
            elif metodo == "Ponto Fixo (MIL)":
                log, raiz = mz.linear_iterative_method(f_str, phi_str, x0, tol, max_iter)
            elif metodo == "Newton-Raphson":
                log, raiz = mz.newton_method(f_str, df_str, x0, tol, max_iter)
            elif metodo == "Secante":
                log, raiz = mz.secant_method(f_str, x0, b, tol, max_iter)
            elif metodo == "Falsa Posição":
                log, raiz = mz.false_position_method(f_str, a, b, tol, max_iter)

            # FORMATAÇÃO DA TABELA
            if log:
                if isinstance(log[0], tuple) and log[0][0] == "Erro":
                    self.escrever_log(self.txt_result_zeros, f"\nERRO: {log[0][1]}\n")
                    return

                header = log[0]
                col_width = 15
                header_str = " | ".join([f"{str(h):<{col_width}}" for h in header])
                
                self.escrever_log(self.txt_result_zeros, header_str + "\n")
                self.escrever_log(self.txt_result_zeros, "-" * len(header_str) + "\n")

                for row in log[1:]:
                    row_str = ""
                    for cell in row:
                        if isinstance(cell, float):
                            cell_str = f"{cell:<{col_width}.6f}"
                        else:
                            cell_str = f"{str(cell):<{col_width}}"
                        row_str += cell_str + " | "
                    
                    self.escrever_log(self.txt_result_zeros, row_str[:-3] + "\n")

                if raiz is not None:
                    self.escrever_log(self.txt_result_zeros, "\n" + "="*85 + "\n")
                    self.escrever_log(self.txt_result_zeros, f"RAIZ ENCONTRADA: {raiz:.8f}")
                else:
                    self.escrever_log(self.txt_result_zeros, "\n\nNão foi possível convergir para uma raiz.")
            else:
                self.escrever_log(self.txt_result_zeros, "Nenhum dado gerado.")

        except Exception as e:
            self.escrever_log(self.txt_result_zeros, f"\nERRO CRÍTICO NO CÁLCULO:\n{e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()