import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

"""
SCRIPT: PIPELINE DE MUTE COM QC (DIAGNÓSTICO VISUAL)
====================================================
1. Calcula o delay baseado no Fcut.
2. Gera gráficos de DIAGNÓSTICO (Traço e Sismograma).
3. Se o QC for aprovado, aplica o mute no arquivo todo.
====================================================
"""

# ================= 1. DADOS DO PROJETO =================
# Caminhos
BASE_DIR = r'C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master'
F_BIN_IN = f'{BASE_DIR}/outputs/Line_CMP_Sorted_AP2.bin'
F_HEAD   = f'{BASE_DIR}/outputs/Trace_Headers_AP2.csv'
F_BIN_OUT= f'{BASE_DIR}/outputs/Line_CMP_Muted_AP2.bin'

# Parâmetros Físicos (CRÍTICO: DEVEM SER IGUAIS AO JSON)
Nt = 1501
dt = 0.001
FCUT = 30.0          # Frequência usada na modelagem (agora 30Hz)
VEL_AGUA = 1500.0    # Velocidade da primeira camada

# Parâmetros de Mute
BUFFER_MS = 60.0     # Margem de segurança (ms) para cortar DEPOIS da onda
TAPER_LEN = 20       # Suavização da borda (amostras)

# ================= 2. LÓGICA DE CÁLCULO =================

def calcular_delay_teorico(f_cut):
    """
    Estima o atraso da fonte Ricker.
    Geralmente o pico está em 1.2 / f_pico, onde f_pico ~ f_cut/3
    """
    f_pico = f_cut / 3.0
    delay_sec = 1.2 / f_pico
    return delay_sec * 1000.0 # Retorna em ms

def get_central_cmp_data():
    """Lê um CMP central para fazer o QC"""
    if not os.path.exists(F_HEAD): raise FileNotFoundError("Headers não encontrados")
    
    df = pd.read_csv(F_HEAD)
    df.columns = df.columns.str.strip()
    
    # Acha o CMP central com maior fold
    counts = df['cmp_x'].value_counts()
    best_cmp = counts.idxmax()
    
    # Filtra dados desse CMP
    df_cmp = df[df['cmp_x'] == best_cmp].sort_values('offset')
    indices = df_cmp['global_trace_index'].values.astype(np.int64)
    offsets = df_cmp['offset'].values
    
    with open(F_BIN_IN, "rb") as f:
        mmap = np.memmap(f, dtype='float32', mode='r', shape=(Nt, len(df)), order='F')
        data = np.array(mmap[:, indices])
        
    return data, offsets, best_cmp

# ================= 3. FUNÇÃO DE QC (O QUE VOCÊ PEDIU) =================

def mostrar_qc_diagnostico(data, offsets, delay_calc_ms, total_mute_ms):
    print("\n--- GERANDO QC VISUAL ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    t_axis = np.arange(Nt) * dt
    
    # --- GRÁFICO 1: QC DO TRAÇO (ZOOM) ---
    # Pega o traço de menor offset (mais próximo da fonte)
    idx_min = np.argmin(np.abs(offsets))
    traco = data[:, idx_min]
    off_traco = offsets[idx_min]
    
    # Calcula onde o pico DEVERIA estar matematicamente
    # T_chegada = Offset/V + Delay
    t_esperado = (np.abs(off_traco) / VEL_AGUA) + (delay_calc_ms / 1000.0)
    
    # Plota o traço (Zoom nos primeiros 500ms ou até onde der)
    zoom_samples = int(0.6 / dt) 
    ax1.plot(t_axis[:zoom_samples], traco[:zoom_samples], 'k-', lw=1.5, label='Traço Real')
    
    # Linha do Delay Calculado
    ax1.axvline(t_esperado, color='r', linestyle='--', lw=2, label=f'Chegada Teórica ({t_esperado*1000:.1f}ms)')
    
    ax1.set_title(f"QC TRAÇO ÚNICO (Offset: {off_traco:.1f}m)")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95, "Se a linha tracejada bater no pico/início,\no cálculo está correto.", 
             transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # --- GRÁFICO 2: QC DO SISMOGRAMA (GATHER) ---
    vm = np.percentile(np.abs(data), 98)
    ax2.imshow(data, aspect='auto', cmap='gray', vmin=-vm, vmax=vm, 
               extent=[offsets[0], offsets[-1], Nt*dt, 0])
    
    # Desenha Linha da Onda Direta (Teórica)
    x_plot = np.linspace(offsets[0], offsets[-1], 100)
    t_onda = (np.abs(x_plot) / VEL_AGUA) + (delay_calc_ms / 1000.0)
    ax2.plot(x_plot, t_onda, 'r--', lw=1, label='Onda Direta (Teórica)')
    
    # Desenha Linha de Mute (Onde vai cortar)
    t_mute = (np.abs(x_plot) / VEL_AGUA) + (total_mute_ms / 1000.0)
    ax2.plot(x_plot, t_mute, 'y-', lw=2, label=f'Linha de Corte (Mute)')
    
    ax2.set_title("QC SISMOGRAMA (Visualização do Corte)")
    ax2.set_xlabel("Offset (m)")
    ax2.legend()
    
    plt.tight_layout()
    print(">> Analise a janela de QC. Feche-a para continuar.")
    plt.show()

# ================= 4. APLICAÇÃO FINAL =================

def aplicar_mute_final(delay_total_ms):
    print("\n--- APLICANDO MUTE NO ARQUIVO ---")
    df = pd.read_csv(F_HEAD)
    n_traces = len(df)
    offsets = df['offset'].values.astype(np.float32)
    
    with open(F_BIN_OUT, "wb") as f_out:
        f_out.seek(n_traces * Nt * 4 - 1); f_out.write(b'\0')
        
    fp_in  = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, n_traces), order='F')
    fp_out = np.memmap(F_BIN_OUT, dtype='float32', mode='r+', shape=(Nt, n_traces), order='F')
    
    block_size = 5000
    num_blocks = (n_traces + block_size - 1) // block_size
    delay_sec = delay_total_ms / 1000.0
    
    for b in tqdm(range(num_blocks)):
        start = b * block_size
        end = min((b+1)*block_size, n_traces)
        
        data = np.array(fp_in[:, start:end])
        offs = offsets[start:end]
        
        # Curva T = |x|/v + delay
        mute_curve = (np.abs(offs) / VEL_AGUA) + delay_sec
        mute_idx = (mute_curve / dt).astype(int)
        
        # Máscara
        rows = np.arange(Nt)[:, None]
        mask = rows > mute_idx[None, :]
        data *= mask
        
        # Taper
        for i in range(data.shape[1]):
            idx = mute_idx[i]
            if idx < Nt:
                lim = min(idx + TAPER_LEN, Nt)
                w = np.linspace(0, 1, lim-idx)
                data[idx:lim, i] *= w
                
        fp_out[:, start:end] = data
        
    print(f"Sucesso! Arquivo salvo: {F_BIN_OUT}")

# ================= MAIN =================

if __name__ == "__main__":
    # 1. Calcula Delay Matemático
    delay_fonte = calcular_delay_teorico(FCUT)
    total_mute = delay_fonte + BUFFER_MS
    
    print(f"Frequência Cut: {FCUT} Hz")
    print(f"Delay Fonte Calculado: {delay_fonte:.2f} ms")
    print(f"Delay Mute Total (com buffer): {total_mute:.2f} ms")
    
    # 2. Gera QC
    try:
        data, offs, cmp_id = get_central_cmp_data()
        mostrar_qc_diagnostico(data, offs, delay_fonte, total_mute)
        
        # 3. Pergunta se aplica
        resp = input("\nO QC visual parece correto? (A linha vermelha bateu na onda?) [s/n]: ")
        if resp.lower() == 's':
            aplicar_mute_final(total_mute)
        else:
            print("Processo cancelado pelo usuário.")
            
    except Exception as e:
        print(f"Erro no QC: {e}")