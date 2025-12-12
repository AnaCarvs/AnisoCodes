import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from tqdm import tqdm
import os

"""
SCRIPT 05: FINAL STACK COMPARISON (ISO vs ANISO)
================================================
Objetivo:
1. Gerar seções de Stack final para dados Isotrópicos (Vrms) e Anisotrópicos (Vrms + Eta).
2. Comparar visualmente as duas seções empilhadas.
----------------------------------
Técnica:
- Usa "Smart Stack" (Soma / Contagem de traços não-nulos) para evitar distorções por mute.
----------------------------------
Parametros:
- Nt: Número de amostras por traço.
- dt: Taxa de amostragem (s).
----------------------------------
Outputs:
- Seções sísmicas empilhadas em arquivos binários.
----------------------------------
Referências:
- Yilmaz, O. (2001). Seismic Data Analysis. SEG.
---------------------------------
Autor: Ana Paula Carvalho
Data: Dezembro de 2025
Versão: 1.0
---------------------------------

"""

# ================= CONFIGURAÇÃO =================
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'

# Inputs
F_HEAD      = f'{BASE_DIR}/outputs/Trace_Headers.csv'
F_IN_ISO    = f'{BASE_DIR}/outputs/Line_NMO_Corrected.bin' 
F_IN_ANI    = f'{BASE_DIR}/outputs/Line_Aniso_Corrected.bin'

# Outputs (Seções Finais)
F_STACK_ISO = f'{BASE_DIR}/outputs/Final_Stack_Isotropic.bin'
F_STACK_ANI = f'{BASE_DIR}/outputs/Final_Stack_Anisotropic.bin'

Nt = 2501           
dt = 0.001          

# ================= FUNÇÕES DE STACK =================

def run_smart_stack(f_bin_input, headers, label="Data"):
    print(f"\n--- Processando Stack: {label} ---")
    
    if not os.path.exists(f_bin_input):
        print(f"ERRO: Arquivo {f_bin_input} não encontrado.")
        return None, None

    # Mapeia input
    n_traces = len(headers)
    data_map = np.memmap(f_bin_input, dtype='float32', mode='r', shape=(Nt, n_traces), order='F')
    
    # Identifica CMPs
    col_cmp = 'cmp' if 'cmp' in headers.columns else 'cmp_x'
    idx_col = 'global_trace_index' if 'global_trace_index' in headers.columns else 'g_idx'
    
    unique_cmps = np.sort(headers[col_cmp].unique())
    n_cmps = len(unique_cmps)
    
    # Array de Saída (Time x CMP)
    stack_section = np.zeros((Nt, n_cmps), dtype=np.float32)
    
    # Agrupa índices para performance
    grouped = headers.groupby(col_cmp)[idx_col].apply(np.array).to_dict()
    
    for i, cmp_val in enumerate(tqdm(unique_cmps)):
        indices = grouped.get(cmp_val, [])
        if len(indices) == 0: continue
        
        # Carrega gather (apenas índices inteiros)
        gather = np.array(data_map[:, indices.astype(int)])
        
        # --- SMART STACK LOGIC ---
        # Soma ao longo do offset (axis=1)
        sum_trace = np.sum(gather, axis=1)
        
        # Conta quantos traços não são zero (para normalizar corretamente o mute)
        # Usamos uma tolerância pequena para float
        live_count = np.sum(np.abs(gather) > 1e-9, axis=1)
        
        # Evita divisão por zero
        live_count[live_count < 1] = 1.0
        
        # Média normalizada pelos traços vivos
        stack_section[:, i] = sum_trace / live_count

    return unique_cmps, stack_section

# ================= VISUALIZAÇÃO =================

class StackViewer:
    def __init__(self, cmps, stk_iso, stk_ani):
        self.cmps = cmps
        self.stk_iso = stk_iso
        self.stk_ani = stk_ani
        self.guides = []
        self.show_guides = False
        
        self.init_plot()
        
    def init_plot(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True, sharex=True)
        self.fig.canvas.manager.set_window_title("5. Comparação Final de Stack")
        
        # Calcula clim comum para comparação justa
        # Usamos percentil 99 do dado anisotrópico (geralmente melhor focado)
        vm = np.percentile(np.abs(self.stk_ani), 99.0)
        if vm == 0: vm = 1.0
        
        # Plot ISO
        self.im1 = self.ax1.imshow(self.stk_iso, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
                                   extent=[self.cmps[0], self.cmps[-1], Nt*dt, 0])
        self.ax1.set_title("Stack Isotrópico (Vrms apenas)")
        self.ax1.set_ylabel("Tempo (s)")
        self.ax1.set_xlabel("CMP")
        
        # Plot ANISO
        self.im2 = self.ax2.imshow(self.stk_ani, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
                                   extent=[self.cmps[0], self.cmps[-1], Nt*dt, 0])
        self.ax2.set_title("Stack Anisotrópico (Vrms + Eta)")
        self.ax2.set_xlabel("CMP")
        
        # Checkbox
        ax_chk = self.fig.add_axes([0.92, 0.5, 0.06, 0.1])
        self.chk = CheckButtons(ax_chk, ["Guias"], [False])
        self.chk.on_clicked(self.toggle_guides)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.show()
        
    def toggle_guides(self, label):
        self.show_guides = not self.show_guides
        
        # Limpa linhas antigas
        for l in self.guides: l.remove()
        self.guides = []
        
        if self.show_guides:
            # Cria linhas a cada 200ms
            times = np.arange(0, Nt*dt, 0.2)
            for t in times:
                l1 = self.ax1.axhline(t, color='red', linestyle='--', alpha=0.4, linewidth=1)
                l2 = self.ax2.axhline(t, color='red', linestyle='--', alpha=0.4, linewidth=1)
                self.guides.extend([l1, l2])
        
        self.fig.canvas.draw_idle()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    # Carrega Headers
    if not os.path.exists(F_HEAD):
        print("Headers não encontrados.")
        exit()
        
    df_head = pd.read_csv(F_HEAD)
    df_head.columns = df_head.columns.str.strip()
    
    # 1. Gera Stack Isotrópico
    cmps, stack_iso = run_smart_stack(F_IN_ISO, df_head, "Isotrópico")
    
    # 2. Gera Stack Anisotrópico
    _, stack_ani = run_smart_stack(F_IN_ANI, df_head, "Anisotrópico (Eta)")
    
    if stack_iso is not None and stack_ani is not None:
        # Salva em disco
        print("\nSalvando arquivos de Stack...")
        stack_iso.tofile(F_STACK_ISO)
        stack_ani.tofile(F_STACK_ANI)
        
        # Abre Visualizador
        print("Abrindo visualizador...")
        app = StackViewer(cmps, stack_iso, stack_ani)