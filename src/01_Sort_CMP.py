import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
SCRIPT 01: SORT CMP + ADVANCED QC (FINAL v10 - WITH SUFFIX + FOLD INFO)
-----------------------------------------------------------------------

Objetivo:
1. Ler sismogramas de múltiplos arquivos de tiros (shots).  
2. Indexar e organizar os traços em função do CMP (Common Mid Point).
3. Gerar arquivo binário ordenado por CMP.
4. Gerar arquivo CSV com cabeçalhos detalhados.
5. Realizar QC avançado com informações de fold mínimo/máximo.
-----------------------------------------------------------------------
Técnica:
- Cada arquivo de tiro é lido e seus traços são indexados.  
- O CMP é calculado e os traços são agrupados por CMP.
- O arquivo binário final é criado usando memória mapeada para eficiência.
- O CSV de cabeçalhos inclui índices globais, CMP, offset, e coordenadas.
- O QC avançado inclui gráficos de fold com min/max destacados.
-----------------------------------------------------------------------
Parâmetros Editáveis:
- BASE_DIR: Diretório base do projeto.  
- ARQUIVO_SRC: Caminho para o arquivo CSV de fontes.
- ARQUIVO_REC: Caminho para o arquivo CSV de receptores.
- PASTA_SISMOGRAMAS: Diretório contendo os arquivos de sismogram
- PREFIXO_ARQUIVO: Prefixo dos arquivos de sismogramas.
- Nt, dt: Parâmetros de amostragem dos dados.
- CMP_BIN_GRID: Tamanho do bin CMP.
- SUFIXO: Sufixo para identificação dos arquivos de saída.
-----------------------------------------------------------------------
Observações:
- Certifique-se de que os arquivos de sismogramas estejam no formato correto.
- Ajuste os parâmetros conforme necessário para o seu conjunto de dados.
- O QC avançado ajuda a validar a geometria e a cobertura dos dados.
-----------------------------------------------------------------------
Autor: Ana Paula Carvalhos
Data: Dezembro de 2025
Versão: 1.0
-----------------------------------------------------------------------
"""

# ================= CONFIGURAÇÃO (EDITÁVEL) =================
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'

# Entradas
ARQUIVO_SRC = f'{BASE_DIR}/inputs/sources_T2.csv' 
ARQUIVO_REC = f'{BASE_DIR}/inputs/receivers_T2.csv' 
PASTA_SISMOGRAMAS = f'{BASE_DIR}/outputs/seismograms/T2/' 

PREFIXO_ARQUIVO = "VTIseismogram_shot_" 

# --- PARÂMETROS DO DADO (MANUAL) ---
Nt = 12001          # Número de amostras (Ex: 12001 para T=6s)
dt = 0.0005         # Taxa de amostragem (Ex: 0.5ms)

# --- GEOMETRIA ---
CMP_BIN_GRID = 10.0 # Tamanho do Bin CMP

# --- CONTROLE DE NOMES (SUFIXO) ---
# Adicione aqui o identificador do teste. Ex: "_T1_8km"
SUFIXO = "_T2_8km" 

# ================= CONFIGURAÇÃO DE SAÍDA =================
# Nomes automáticos baseados no sufixo
NOME_ARQUIVO_BIN = f'Line_CMP_Sorted{SUFIXO}.bin'
NOME_ARQUIVO_CSV = f'Trace_Headers{SUFIXO}.csv'

OUTPUT_BIN = f'{BASE_DIR}/outputs/{NOME_ARQUIVO_BIN}'
OUTPUT_CSV = f'{BASE_DIR}/outputs/{NOME_ARQUIVO_CSV}'
QC_FOLDER  = f'{BASE_DIR}/outputs/QC_Geometry/'
os.makedirs(QC_FOLDER, exist_ok=True)

SALVAR_QC = True
MOSTRAR_QC = True 

# ================= EXECUÇÃO =================

def passo_1_indexar(Nt_user):
    print("--- 1. Indexando Arquivos ---")
    if not os.path.exists(PASTA_SISMOGRAMAS): raise FileNotFoundError("Pasta de dados não encontrada.")
    
    src_df = pd.read_csv(ARQUIVO_SRC); src_df.columns = src_df.columns.str.strip()
    rec_df = pd.read_csv(ARQUIVO_REC); rec_df.columns = rec_df.columns.str.strip()
    
    col_sx = src_df.columns[1]; col_rx = rec_df.columns[1]
    
    files = sorted([f for f in os.listdir(PASTA_SISMOGRAMAS) if f.startswith(PREFIXO_ARQUIVO) and f.endswith('.bin')])
    print(f"-> Arquivos encontrados: {len(files)}")
    print(f"-> Usando Nt manual: {Nt_user}")

    headers = []
    for fname in tqdm(files, desc="Indexando"):
        try:
            parts = fname.replace('.bin', '').split('_')
            if 'shot' in parts:
                idx = parts.index('shot') + 1
                shot_id = int(parts[idx]) - 1 
            else: continue
        except: continue
            
        if shot_id >= len(src_df): continue
        
        sx = src_df.iloc[shot_id][col_sx]
        rxs = rec_df[col_rx].values
        
        df = pd.DataFrame({
            'file_name': fname,
            'shot_id': shot_id,
            'trace_in_shot': np.arange(len(rxs)),
            'src_x': sx,
            'rec_x': rxs
        })
        
        df['offset'] = np.round(np.abs(df['rec_x'] - df['src_x']), 1)
        raw_cmp = (df['rec_x'] + df['src_x']) / 2.0
        df['cmp_x'] = (np.round(raw_cmp / CMP_BIN_GRID) * CMP_BIN_GRID).astype(int)
        
        headers.append(df)
        
    return pd.concat(headers, ignore_index=True) if headers else None

def passo_2_qc_avancado(df, Nt, dt):
    print("\n--- 2. QC Avançado de Dados ---")
    
    # 2.1 QC RAW SHOT
    print("-> Gerando QC: Raw Shot Gather...")
    shot_id_qc = df['shot_id'].unique()[len(df['shot_id'].unique()) // 2]
    shot_df = df[df['shot_id'] == shot_id_qc]
    fname = shot_df.iloc[0]['file_name']
    
    path = os.path.join(PASTA_SISMOGRAMAS, fname)
    raw = np.fromfile(path, dtype=np.float32)
    nrec = len(shot_df)
    
    if raw.size != nrec * Nt:
        print(f"ERRO FATAL: Tamanho incorreto no arquivo {fname}.")
        return 
        
    data_shot = raw.reshape((Nt, nrec), order='C') 
    
    plt.figure(figsize=(10, 6))
    vm = np.percentile(np.abs(data_shot), 99)
    plt.imshow(data_shot, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
               extent=[0, nrec, Nt*dt, 0])
    plt.title(f"QC Raw Shot {shot_id_qc}{SUFIXO}")
    plt.xlabel("Canal"); plt.ylabel("Tempo (s)")
    plt.colorbar(label="Amplitude")
    if SALVAR_QC: plt.savefig(f'{QC_FOLDER}QC_01_Raw_Shot{SUFIXO}.png')
    
    # 2.2 QC GEOMETRY
    print("-> Gerando QC: Geometria CMP...")
    fold = df.groupby('cmp_x').size()
    
    # --- ADIÇÃO: CÁLCULO DE MIN/MAX FOLD ---
    min_fold = fold.min()
    max_fold = fold.max()
    # ---------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(fold.index, fold.values, 'k-')
    ax1.fill_between(fold.index, fold.values, color='gold', alpha=0.5)
    
    # --- ADIÇÃO: LINHAS E TEXTO NO PLOT ---
    ax1.axhline(max_fold, color='red', linestyle='--', alpha=0.6, label=f'Max: {max_fold}')
    ax1.axhline(min_fold, color='blue', linestyle='--', alpha=0.6, label=f'Min: {min_fold}')
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Fold"); ax1.set_title(f"Fold Coverage {SUFIXO} (Min: {min_fold} | Max: {max_fold})")
    # ---------------------------------------
    
    ax1.grid(True, alpha=0.3)
    
    samp = df.sample(min(len(df), 50000))
    ax2.scatter(samp['cmp_x'], samp['offset'], c='blue', s=1, alpha=0.1)
    ax2.set_ylabel("Offset (m)"); ax2.set_xlabel("CMP (m)")
    ax2.set_title("Offset Distribution")
    ax2.grid(True, alpha=0.3)
    if SALVAR_QC: plt.savefig(f'{QC_FOLDER}QC_02_Geometry{SUFIXO}.png')

    # 2.3 QC CMP GATHERS
    print("-> Gerando QC: Exemplos de CMP Gathers...")
    unique_cmps = np.sort(df['cmp_x'].unique())
    targets = [
        unique_cmps[int(len(unique_cmps)*0.2)], 
        unique_cmps[int(len(unique_cmps)*0.5)], 
        unique_cmps[int(len(unique_cmps)*0.8)]  
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)
    fig.suptitle(f"QC CMP Gathers {SUFIXO}")
    
    for ax, tgt in zip(axes, targets):
        traces = df[df['cmp_x'] == tgt].sort_values('offset')
        if traces.empty: continue
        
        gather = np.zeros((Nt, len(traces)), dtype=np.float32)
        offs = traces['offset'].values
        
        for i, (_, row) in enumerate(traces.iterrows()):
            try:
                r = np.fromfile(os.path.join(PASTA_SISMOGRAMAS, row['file_name']), dtype=np.float32)
                d = r.reshape((Nt, -1), order='C')
                gather[:, i] = d[:, int(row['trace_in_shot'])]
            except: pass
            
        ax.imshow(gather, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
                  extent=[offs[0], offs[-1], Nt*dt, 0])
        ax.set_title(f"CMP {tgt} (Fold {len(traces)})")
        ax.set_xlabel("Offset")
        
    axes[0].set_ylabel("Tempo (s)")
    if SALVAR_QC: plt.savefig(f'{QC_FOLDER}QC_03_CMPs{SUFIXO}.png')
    
    if MOSTRAR_QC: plt.show()

def passo_3_gravar(df, Nt):
    print(f"\n--- 3. Gravando Saída: {NOME_ARQUIVO_BIN} ---")
    
    # Ordena
    df_sorted = df.sort_values(by=['cmp_x', 'offset']).reset_index(drop=True)
    df_sorted['global_trace_index'] = df_sorted.index
    
    # Salva CSV
    cols = ['global_trace_index', 'cmp_x', 'offset', 'src_x', 'rec_x']
    df_sorted[cols].to_csv(OUTPUT_CSV, index=False)
    print(f"-> Header salvo.")
    
    # Salva Binário
    n_traces = len(df_sorted)
    with open(OUTPUT_BIN, "wb") as f:
        f.seek(n_traces * Nt * 4 - 1); f.write(b'\0')
        
    fp = np.memmap(OUTPUT_BIN, dtype='float32', mode='r+', shape=(Nt, n_traces), order='F')
    
    grouped = df_sorted.groupby('file_name')
    for fname, group in tqdm(grouped, desc="Consolidando"):
        path = os.path.join(PASTA_SISMOGRAMAS, fname)
        try:
            raw = np.fromfile(path, dtype=np.float32)
            nrec = raw.size // Nt
            if raw.size != nrec * Nt: continue
            
            shot_data = raw.reshape((Nt, nrec), order='C')
            
            src = group['trace_in_shot'].values
            dst = group['global_trace_index'].values
            
            if len(dst) > 0: fp[:, dst] = shot_data[:, src]
        except: pass
            
    del fp
    print("--- Processamento Concluído! ---")

if __name__ == "__main__":
    print(f"CONFIGURAÇÃO MANUAL: Nt={Nt}, dt={dt}s | Sufixo='{SUFIXO}'")
    df = passo_1_indexar(Nt)
    if df is not None:
        passo_2_qc_avancado(df, Nt, dt)
        passo_3_gravar(df, Nt)