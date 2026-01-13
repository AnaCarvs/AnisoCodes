import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = r'C:/Users/AnaCarvs.GISIS/Desktop/Dataset'

# Entrada (Saída do passo anterior)
FILE_NMO  = f'{BASE_DIR}/Linha/Line_NMO_Corrected_AP2.bin'
FILE_HEAD = f'{BASE_DIR}/Linha/Trace_Headers_AP2.csv'

# Saída
FILE_STACK = f'{BASE_DIR}/Linha/Line_Stack_Final_AP2.bin'

# Geometria
Nt = 1501
dt = 0.001

# Visualização
PCLIP = 99.0 # Percentil para corte de amplitude no plot (melhor contraste)

# ==============================================================================
# 2. ROTINA DE EMPILHAMENTO (STACK)
# ==============================================================================
def run_stacking():
    print(">>> Lendo cabeçalhos...")
    try:
        h = pd.read_csv(FILE_HEAD)
        h.columns = h.columns.str.strip().str.lower()
        if 'cmp_x' in h.columns: h.rename(columns={'cmp_x':'cmp'}, inplace=True)
        if 'global_trace_index' not in h.columns: h['global_trace_index'] = h.index
    except Exception as e:
        print(f"ERRO CRÍTICO ao ler cabeçalhos: {e}")
        return

    # Mapeia o arquivo NMO (Leitura rápida)
    if not os.path.exists(FILE_NMO):
        print(f"ERRO: Arquivo NMO não encontrado em {FILE_NMO}")
        return
        
    print(f">>> Mapeando dados NMO ({Nt} x {len(h)} traces)...")
    nmo_data = np.memmap(FILE_NMO, dtype='float32', mode='r', shape=(Nt, len(h)), order='F')

    # Identifica CMPs únicos ordenados (Eixo X da seção)
    unique_cmps = np.sort(h['cmp'].unique())
    n_cmps = len(unique_cmps)
    
    print(f">>> Empilhando {n_cmps} CMPs...")
    
    # Matriz da Seção Stack (Tempo x CMP)
    stack_section = np.zeros((Nt, n_cmps), dtype=np.float32)
    
    # Agrupamento pandas (Muito mais rápido que filtrar no loop)
    grouped = h.groupby('cmp')
    
    # Loop de Empilhamento
    for i, cmp_val in enumerate(tqdm(unique_cmps, desc="Stacking")):
        if cmp_val not in grouped.groups:
            continue
            
        # Pega índices dos traços que pertencem a este CMP
        trace_indices = grouped.get_group(cmp_val).index.values
        
        # Lê os traços do disco
        gather = np.array(nmo_data[:, trace_indices])
        
        # Fold (Multiplicidade real neste CMP)
        fold = gather.shape[1]
        
        if fold > 0:
            # SOMA E NORMALIZAÇÃO (Média Aritmética - Padrão Indústria)
            # Stack = Sum(Traces) / Fold
            stack_trace = np.sum(gather, axis=1) / fold
            stack_section[:, i] = stack_trace
            
    print(">>> Salvando Seção Stack...")
    # Salva em binário puro (float32)
    stack_section.tofile(FILE_STACK)
    
    # ==============================================================================
    # 3. QC: VISUALIZAÇÃO DA SEÇÃO FINAL
    # ==============================================================================
    print(">>> Gerando QC da Seção Final...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Cálculo de ganho visual (Percentil)
    vm = np.percentile(np.abs(stack_section), PCLIP)
    if vm == 0: vm = 1
    
    # Extent para eixos corretos [CMP_min, CMP_max, Time_max, Time_min]
    extent = [unique_cmps[0], unique_cmps[-1], Nt*dt, 0]
    
    im = ax.imshow(stack_section, aspect='auto', cmap='gray', vmin=-vm, vmax=vm, 
                   extent=extent, origin='upper')
    
    ax.set_title(f"Seção Sísmica Empilhada (Stack) - {n_cmps} CMPs")
    ax.set_xlabel("CMP Number")
    ax.set_ylabel("Time [s]")
    ax.grid(False)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Amplitude")
    
    plt.tight_layout()
    plt.show()
    
    print(f">>> Processo concluído. Arquivo salvo em: {FILE_STACK}")

if __name__ == "__main__":
    import os
    run_stacking()