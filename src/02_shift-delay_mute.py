import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

"""
SCRIPT: PIPELINE DE MUTE AUTOMÁTICO - VERSÃO PRODUÇÃO
======================================================
FUNCIONALIDADE:
- Calcula o delay automaticamente (Fator 4.0 / Fcut).
- Aplica mute cirúrgico baseado na velocidade da água.
- Gera visualização de QC antes de salvar.

COMO USAR EM NOVOS DADOS:
- Apenas altere as variáveis na seção 'CONFIGURAÇÃO DO USUÁRIO'.
======================================================
"""

# ==============================================================================
# 1. CONFIGURAÇÃO DO USUÁRIO (EDITE AQUI PARA NOVOS DADOS)
# ==============================================================================

# --- Caminhos e Arquivos ---
BASE_DIR  = r'C:/Users/AnaCarvs.GISIS/Desktop/Dataset'
ARQUIVO_BIN_ENTRADA = 'Line_CMP_Sorted_AP2.bin'
ARQUIVO_CSV_HEADERS = 'Trace_Headers_AP2.csv'
ARQUIVO_BIN_SAIDA   = 'Line_CMP_Muted_AP2.bin'

# --- Parâmetros da Modelagem (Física) ---
FCUT = 30.0          # Frequência de corte usada na modelagem deste dado
VEL_AGUA = 1500.0    # Velocidade da lâmina d'água

# --- Geometria do Dado (MUITO IMPORTANTE: Ajuste se mudar o dado) ---
Nt = 1501            # Número de amostras por traço
dt = 0.001           # Taxa de amostragem (segundos)

# --- Ajuste Fino do Mute ---
BUFFER_MS = 60.0     # Quantos ms cortar DEPOIS da onda direta (Margem)
TAPER_LEN = 20       # Suavização da borda do corte (amostras)

# ==============================================================================
# 2. LÓGICA CORE (NÃO MEXER)
# ==============================================================================

# Montagem dos caminhos completos
F_BIN_IN  = os.path.join(BASE_DIR, 'Linha', ARQUIVO_BIN_ENTRADA)
F_HEAD    = os.path.join(BASE_DIR, 'Linha', ARQUIVO_CSV_HEADERS)
F_BIN_OUT = os.path.join(BASE_DIR, 'Linha', ARQUIVO_BIN_SAIDA)

def calcular_delay_universal(f_cut):
    """
    Calcula o delay baseado na assinatura descoberta: Fator 4.0
    Fórmula: Delay = 4.0 / Fcut
    """
    if f_cut <= 0: return 0.0
    delay_sec = 4.0 / f_cut
    return delay_sec * 1000.0 # Retorna em ms

def get_central_cmp_data():
    """Lê o CMP central automaticamente para QC"""
    if not os.path.exists(F_HEAD): 
        raise FileNotFoundError(f"Arquivo de headers não encontrado: {F_HEAD}")
    
    print(f"Lendo headers: {ARQUIVO_CSV_HEADERS}...")
    df = pd.read_csv(F_HEAD)
    df.columns = df.columns.str.strip()
    
    # Validação simples de geometria
    if len(df) == 0: raise ValueError("O arquivo de headers está vazio.")
    
    # Acha o CMP central com maior fold
    counts = df['cmp_x'].value_counts()
    best_cmp = counts.idxmax()
    
    print(f"CMP selecionado para QC: {best_cmp} (Fold: {counts[best_cmp]})")
    
    df_cmp = df[df['cmp_x'] == best_cmp].sort_values('offset')
    indices = df_cmp['global_trace_index'].values.astype(np.int64)
    offsets = df_cmp['offset'].values
    
    if not os.path.exists(F_BIN_IN):
        raise FileNotFoundError(f"Arquivo binário não encontrado: {F_BIN_IN}")

    # Leitura do binário
    with open(F_BIN_IN, "rb") as f:
        # Nota: O memmap precisa do Nt correto para não ler lixo
        mmap = np.memmap(f, dtype='float32', mode='r', shape=(Nt, len(df)), order='F')
        data = np.array(mmap[:, indices])
        
    return data, offsets

def mostrar_qc(data, offsets, delay_calc, total_mute):
    print(f"\n--- QC VISUAL (Fcut={FCUT}Hz -> Delay={delay_calc:.1f}ms) ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    t_axis = np.arange(Nt) * dt
    
    # 1. Traço Único
    idx_min = np.argmin(np.abs(offsets))
    traco = data[:, idx_min]
    off_traco = offsets[idx_min]
    
    # T = Offset/V + Delay
    t_pico = (np.abs(off_traco) / VEL_AGUA) + (delay_calc / 1000.0)
    
    zoom = int(min(0.6 / dt, Nt))
    ax1.plot(t_axis[:zoom], traco[:zoom], 'k-', lw=1.5, label='Traço')
    ax1.axvline(t_pico, color='r', linestyle='--', lw=2, label='Pico Previsto')
    ax1.set_title(f"QC TRAÇO (Offset: {off_traco:.1f}m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    msg = f"Delay Previsto: {delay_calc:.1f}ms\n(Fator 4.0 / Fcut)"
    ax1.text(0.02, 0.92, msg, transform=ax1.transAxes, bbox=dict(facecolor='white', edgecolor='red'))

    # 2. Sismograma
    vm = np.percentile(np.abs(data), 98)
    ax2.imshow(data, aspect='auto', cmap='gray', vmin=-vm, vmax=vm, 
               extent=[offsets[0], offsets[-1], Nt*dt, 0])
    
    x_plot = np.linspace(offsets[0], offsets[-1], 100)
    t_pico_curve = (np.abs(x_plot) / VEL_AGUA) + (delay_calc / 1000.0)
    t_mute_curve = (np.abs(x_plot) / VEL_AGUA) + (total_mute / 1000.0)
    
    ax2.plot(x_plot, t_pico_curve, 'r--', lw=1.5, label='Pico')
    ax2.plot(x_plot, t_mute_curve, 'y-', lw=2.5, label='Linha de Corte')
    ax2.set_title("QC SISMOGRAMA")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def aplicar_mute(delay_total_ms):
    print(f"\n--- PROCESSANDO ARQUIVO: {ARQUIVO_BIN_SAIDA} ---")
    df = pd.read_csv(F_HEAD)
    n_traces = len(df)
    offsets = df['offset'].values.astype(np.float32)
    
    # Prepara output
    with open(F_BIN_OUT, "wb") as f_out:
        f_out.seek(n_traces * Nt * 4 - 1); f_out.write(b'\0')
        
    fp_in  = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, n_traces), order='F')
    fp_out = np.memmap(F_BIN_OUT, dtype='float32', mode='r+', shape=(Nt, n_traces), order='F')
    
    block_size = 5000 # Processa em blocos para não estourar memória
    num_blocks = (n_traces + block_size - 1) // block_size
    delay_sec = delay_total_ms / 1000.0
    
    for b in tqdm(range(num_blocks), desc="Aplicando Mute"):
        start = b * block_size
        end = min((b+1)*block_size, n_traces)
        
        data = np.array(fp_in[:, start:end])
        offs = offsets[start:end]
        
        # Cálculo vetorial do mute
        mute_curve = (np.abs(offs) / VEL_AGUA) + delay_sec
        mute_idx = (mute_curve / dt).astype(int)
        
        # Máscara Booleana (Muito mais rápido que for loop)
        rows = np.arange(Nt)[:, None]
        mask = rows > mute_idx[None, :]
        data *= mask
        
        # Aplicação do Taper (Suavização)
        for i in range(data.shape[1]):
            idx = mute_idx[i]
            if idx < Nt:
                lim = min(idx + TAPER_LEN, Nt)
                # Rampa linear de 0 a 1
                data[idx:lim, i] *= np.linspace(0, 1, lim-idx)
                
        fp_out[:, start:end] = data
        
    print(f"Concluído! Arquivo salvo em: {F_BIN_OUT}")

# ================= MAIN =================

if __name__ == "__main__":
    print(">>> INICIANDO PIPELINE DE MUTE AUTOMÁTICO <<<")
    
    # 1. Calcula Delay
    delay_previsto = calcular_delay_universal(FCUT)
    total_mute = delay_previsto + BUFFER_MS
    
    print(f"Parâmetros: FCUT={FCUT}Hz | Nt={Nt} | dt={dt}s")
    print(f"Delay Calculado: {delay_previsto:.2f} ms")
    
    try:
        # 2. Carrega e Mostra QC
        data, offs = get_central_cmp_data()
        mostrar_qc(data, offs, delay_previsto, total_mute)
        
        # 3. Execução
        resp = input("\nO gráfico de QC está correto? [s/n]: ")
        if resp.lower() == 's':
            aplicar_mute(total_mute)
        else:
            print("Operação cancelada. Ajuste as configurações no topo do script.")
            
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print("Dica: Verifique se Nt, dt e os nomes dos arquivos estão corretos na seção de CONFIGURAÇÃO.")