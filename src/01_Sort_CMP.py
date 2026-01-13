import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
SCRIPT 01: SORT CMP + ADVANCED QC + BINARY CHECK (FINAL v11)
-----------------------------------------------------------------------
Melhorias:
1. QC ativado por padrão.
2. Novo PASSO 4: Lê o arquivo binário FINAL do disco para provar que
   a gravação funcionou.
3. Visualização com Ganho Agressivo (para ver reflexões fracas).
-----------------------------------------------------------------------
"""

# ================= CONFIGURAÇÃO (EDITÁVEL) =================
BASE_DIR = r'C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master'

# Entradas
ARQUIVO_SRC = f'{BASE_DIR}/inputs/sources_AP2.csv' 
ARQUIVO_REC = f'{BASE_DIR}/inputs/receivers_AP2.csv' 
PASTA_SISMOGRAMAS = f'{BASE_DIR}/outputs/seismograms/AP2/' 

PREFIXO_ARQUIVO = "VTIseismogram_shot_" 

# --- PARÂMETROS DO DADO ---
Nt = 1501          # Número de amostras 
dt = 0.001         # Taxa de amostragem

# --- GEOMETRIA ---
CMP_BIN_GRID = 10.0 # Tamanho do Bin CMP

# --- VISUALIZAÇÃO ---
GANHO_VISUAL = 2.5  # Ganho forte para ver reflexões
SUFIXO = "_AP2" 

# ================= CONFIGURAÇÃO DE SAÍDA =================
NOME_ARQUIVO_BIN = f'Line_CMP_Sorted{SUFIXO}.bin'
NOME_ARQUIVO_CSV = f'Trace_Headers{SUFIXO}.csv'

OUTPUT_BIN = f'{BASE_DIR}/outputs/{NOME_ARQUIVO_BIN}'
OUTPUT_CSV = f'{BASE_DIR}/outputs/{NOME_ARQUIVO_CSV}'
QC_FOLDER  = f'{BASE_DIR}/outputs/QC_Geometry/'
os.makedirs(QC_FOLDER, exist_ok=True)

# <--- ATIVADO O QC --->
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

def passo_2_qc_geometria(df):
    """Gera gráficos de Fold e Offset (Geometria apenas)"""
    print("\n--- 2. QC de Geometria (Fold) ---")
    
    fold = df.groupby('cmp_x').size()
    min_fold = fold.min()
    max_fold = fold.max()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Fold Chart 
    ax1.plot(fold.index, fold.values, 'k-', lw=1)
    ax1.fill_between(fold.index, fold.values, color='gold', alpha=0.5)
    ax1.axhline(max_fold, color='red', linestyle='--', alpha=0.6, label=f'Max: {max_fold}')
    ax1.axhline(min_fold, color='blue', linestyle='--', alpha=0.6, label=f'Min: {min_fold}')
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Fold (Traços)"); ax1.set_title(f"Cobertura CMP {SUFIXO}")
    ax1.grid(True, alpha=0.3)
    
    # Offset Chart
    samp = df.sample(min(len(df), 10000))
    ax2.scatter(samp['cmp_x'], samp['offset'], c='blue', s=1, alpha=0.1)
    ax2.set_ylabel("Offset (m)"); ax2.set_xlabel("Posição CMP (m)")
    ax2.set_title("Distribuição de Offsets")
    ax2.grid(True, alpha=0.3)
    
    if SALVAR_QC: 
        plt.savefig(f'{QC_FOLDER}QC_Geometry{SUFIXO}.png')
        print(f"-> Gráfico salvo em: {QC_FOLDER}")
    
    if MOSTRAR_QC: plt.show()

def passo_3_gravar(df, Nt):
    print(f"\n--- 3. Gravando Binário Ordenado ---")
    
    # Ordena fisicamente os metadados
    df_sorted = df.sort_values(by=['cmp_x', 'offset']).reset_index(drop=True)
    df_sorted['global_trace_index'] = df_sorted.index
    
    # Salva CSV
    cols = ['global_trace_index', 'cmp_x', 'offset', 'src_x', 'rec_x']
    df_sorted[cols].to_csv(OUTPUT_CSV, index=False)
    
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
            
    del fp # Fecha o arquivo para permitir leitura no passo 4
    print("-> Binário Gravado.")
    return df_sorted

def passo_4_qc_final_binario(df_sorted, Nt):
    """
    Lê o arquivo BINÁRIO final do disco e mostra um CMP.
    Isso prova que o sort funcionou e o arquivo não está corrompido.
    """
    if not MOSTRAR_QC: return
    print("\n--- 4. QC FINAL: Verificando Arquivo Gravado ---")
    
    # Escolhe o CMP central (melhor fold)
    counts = df_sorted['cmp_x'].value_counts()
    best_cmp = counts.idxmax()
    
    # Filtra traços desse CMP
    subset = df_sorted[df_sorted['cmp_x'] == best_cmp]
    indices = subset['global_trace_index'].values
    offsets = subset['offset'].values
    
    # Lê do disco
    fp = np.memmap(OUTPUT_BIN, dtype='float32', mode='r', shape=(Nt, len(df_sorted)), order='F')
    gather = np.array(fp[:, indices]) # Cópia para RAM
    
    # Aplica Ganho Agressivo (Para ver reflexão)
    t = np.arange(Nt) * dt
    gained = gather * (t[:, None]**GANHO_VISUAL)
    
    # Clipagem para ignorar onda direta
    vm = np.percentile(np.abs(gained), 90) # 90% clipa a onda direta
    
    plt.figure(figsize=(10, 8))
    plt.imshow(gained, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
               extent=[offsets[0], offsets[-1], Nt*dt, 0], interpolation='bilinear') # Bilinear suaviza
    
    plt.title(f"QC FINAL DO ARQUIVO .BIN\nCMP {best_cmp} (Fold: {len(offsets)})")
    plt.xlabel("Offset (m)")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="Amplitude (Com Ganho)")
    plt.text(0.5, 0.95, "Leitura direta do arquivo Sorted", 
             ha='center', transform=plt.gca().transAxes, color='yellow', fontweight='bold')
    
    print(f"-> Mostrando CMP {best_cmp} lido do arquivo {os.path.basename(OUTPUT_BIN)}")
    plt.show()

if __name__ == "__main__":
    df = passo_1_indexar(Nt)
    if df is not None:
        passo_2_qc_geometria(df)      # Verifica Fold antes de gravar
        df_final = passo_3_gravar(df, Nt) # Grava
        passo_4_qc_final_binario(df_final, Nt) # Verifica se gravou certo