import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
SCRIPT 03: CMP STACKING (EMPILHAMENTO)
======================================
Objetivo:   
1. Empilhar (Stack) os gathers CMP do dado NMO Corrected.
2. Gerar seção sísmica empilhada.
----------------------------------
Técnica:
- Usa "Smart Stack" (Soma / Contagem de traços não-nulos).
  Isso evita que zonas com Mute fiquem com amplitude artificialmente baixa.
----------------------------------
Parametros:
- Nt: Número de amostras por traço. 
- dt: Taxa de amostragem (s).
- CMP_BIN_SIZE: Tamanho do bin CMP (unidade de distância).
----------------------------------
Outputs:
- Seção sísmica empilhada em arquivo binário.
----------------------------------
Referências:
- Yilmaz, O. (2001). Seismic Data Analysis. SEG.
---------------------------------
Autor: Ana Paula Carvalhos
Data: Dezembro de 2025
Versão: 1.0
---------------------------------

"""

# ================= CONFIGURAÇÃO =================
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'

# Entrada (A saída do script anterior)
F_NMO_IN = f'{BASE_DIR}/outputs/Line_NMO_Corrected.bin'
F_HEAD   = f'{BASE_DIR}/outputs/Trace_Headers.csv'

# Saída
F_SEISMIC_SECTION = f'{BASE_DIR}/outputs/Seismic_Section_Stacked.bin'

Nt = 2501           
dt = 0.001          
CMP_BIN_SIZE = 20.0 

# ================= FUNÇÕES =================

def load_headers():
    print("Carregando headers...")
    df = pd.read_csv(F_HEAD)
    df.columns = df.columns.str.strip()
    return df

def run_stacking():
    if not os.path.exists(F_NMO_IN):
        print(f"ERRO: Arquivo NMO não encontrado: {F_NMO_IN}")
        return None, None

    headers = load_headers()
    
    # Mapeia o arquivo binário gigante (NMO Corrected)
    # Shape = (Nt, N_Traces_Total)
    n_traces_total = len(headers)
    data_nmo = np.memmap(F_NMO_IN, dtype='float32', mode='r', shape=(Nt, n_traces_total), order='F')
    
    # Identifica CMPs únicos
    col_cmp = 'cmp' if 'cmp' in headers.columns else 'cmp_x'
    unique_cmps = np.sort(headers[col_cmp].unique())
    n_cmps = len(unique_cmps)
    
    print(f"Iniciando Empilhamento de {n_cmps} CMPs...")
    
    # Array para a seção final (Time x Space)
    stacked_section = np.zeros((Nt, n_cmps), dtype=np.float32)
    
    # Loop principal (Pode ser otimizado, mas para stack simples o Pandas ajuda)
    # Vamos iterar pelos CMPs únicos
    
    # Dica de performance: Agrupar índices antes do loop
    grouped = headers.groupby(col_cmp)
    
    # Se 'global_trace_index' não existir, criamos (assume ordem sequencial se bater com o binário)
    if 'global_trace_index' not in headers.columns:
        headers['global_trace_index'] = np.arange(len(headers))
        
    idx_col = 'global_trace_index'

    # Barra de progresso
    for i, cmp_val in enumerate(tqdm(unique_cmps)):
        # Pega os índices dos traços que pertencem a este CMP
        indices = grouped.get_group(cmp_val)[idx_col].values.astype(int)
        
        # Carrega apenas os traços desse CMP do disco
        gather = data_nmo[:, indices]
        
        # --- EMPILHAMENTO (Média ou Soma) ---
        # Aqui usamos média (mean) para normalizar pela fold (cobertura),
        # evitando que CMPs com alta cobertura fiquem muito mais fortes que os de baixa.
        # axis=1 soma ao longo dos offsets (colapsa as colunas)
        
        # Filtra zeros se necessário (mute) para não dividir errado na média
        # Mas num stack simples, np.mean resolve bem.
        stack_trace = np.mean(gather, axis=1)
        
        # Salva na seção final
        stacked_section[:, i] = stack_trace

    # Salva resultado em disco
    print("Salvando seção empilhada...")
    stacked_section.tofile(F_SEISMIC_SECTION)
    
    return unique_cmps, stacked_section

def plot_section(cmps, section):
    print("Plotando seção...")
    
    # Ganho para visualização (AGC simples ou Percentile)
    vm = np.percentile(np.abs(section), 98)
    if vm == 0: vm = 1
    
    plt.figure(figsize=(12, 6))
    plt.title("Seção Sísmica Empilhada (Stacked)")
    
    # Extent: [Left, Right, Bottom, Top]
    # Note que Bottom é o tempo máximo e Top é 0
    plt.imshow(section, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
               extent=[cmps[0], cmps[-1], Nt*dt, 0])
    
    plt.xlabel("CMP Number")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

# ================= EXECUÇÃO =================

if __name__ == "__main__":
    cmps, section = run_stacking()
    if section is not None:
        plot_section(cmps, section)