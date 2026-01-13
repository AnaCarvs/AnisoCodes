import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
SCRIPT 04: CMP STACKING (UNIVERSAL)
=====================================================
Instruções de Uso:
1. Altere APENAS a seção 'CONFIGURAÇÃO DO USUÁRIO'.
2. O script detecta automaticamente o número de traços e CMPs.
3. O 'Smart Stack' lida com mutes e geometrias irregulares.
=====================================================
"""

# ==============================================================================
# 1. CONFIGURAÇÃO DO USUÁRIO (EDITE AQUI)
# ==============================================================================

# Caminhos (Use barras normais '/' ou r'C:\...')
BASE_DIR = r'C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master'

# Arquivos de Entrada (Devem existir)
FILE_NMO_INPUT  = f'{BASE_DIR}/outputs/Line_NMO_Corrected_AP2.bin' # Dado NMO corrigido
FILE_HEADERS    = f'{BASE_DIR}/outputs/Trace_Headers_AP2.csv'      # Headers correspondentes

# Arquivo de Saída (Será criado)
FILE_STACK_OUT  = f'{BASE_DIR}/outputs/Seismic_Section_Stacked_AP2.bin'

# Parâmetros do Seu Dado (Verifique no header ou EBCDIC)
Nt = 1501             # Número de amostras por traço
dt = 0.001            # Taxa de amostragem (segundos)
CMP_BIN_SIZE = 10.0   # Tamanho do bin (apenas p/ info, não afeta o cálculo direto)

# ==============================================================================
# 2. LÓGICA DE PROCESSAMENTO (NÃO PRECISA EDITAR)
# ==============================================================================

def load_headers_safe(path):
    print(f"Lendo headers: {os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de headers não encontrado: {path}")
    
    df = pd.read_csv(path)
    # Padroniza nomes de colunas (remove espaços e minúsculas)
    df.columns = df.columns.str.strip().str.lower()
    
    # Cria índice global se não existir (essencial para mapear o binário)
    if 'global_trace_index' not in df.columns:
        print("-> Criando índice de traços sequencial...")
        df['global_trace_index'] = np.arange(len(df))
        
    return df

def run_stacking():
    # 1. Verificações
    if not os.path.exists(FILE_NMO_INPUT):
        print(f"ERRO CRÍTICO: Binário de entrada não existe: {FILE_NMO_INPUT}")
        return None, None

    # 2. Carrega Geometria
    headers = load_headers_safe(FILE_HEADERS)
    n_traces_total = len(headers)
    
    # Valida tamanho do arquivo binário para evitar erros de leitura
    expected_bytes = n_traces_total * Nt * 4 # 4 bytes por float32
    actual_bytes = os.path.getsize(FILE_NMO_INPUT)
    
    if actual_bytes != expected_bytes:
        print(f"\n[ALERTA] Tamanho do arquivo incorreto!")
        print(f"Esperado (Nt={Nt} * Traces={n_traces_total}): {expected_bytes} bytes")
        print(f"Encontrado: {actual_bytes} bytes")
        print("-> Verifique se o parâmetro 'Nt' está correto para este dado.")
        return None, None

    # 3. Mapeia o Dado (Memmap para não estourar a RAM)
    print(f"Mapeando dado NMO ({n_traces_total} traços x {Nt} amostras)...")
    data_nmo = np.memmap(FILE_NMO_INPUT, dtype='float32', mode='r', shape=(Nt, n_traces_total), order='F')
    
    # 4. Prepara Empilhamento
    col_cmp = 'cmp' if 'cmp' in headers.columns else 'cmp_x'
    unique_cmps = np.sort(headers[col_cmp].unique())
    n_cmps = len(unique_cmps)
    
    print(f"Iniciando Empilhamento de {n_cmps} CMPs únicos...")
    
    # Array da Seção Final (Tempo x CMP)
    stacked_section = np.zeros((Nt, n_cmps), dtype=np.float32)
    
    # Agrupa índices para acesso rápido (muito mais rápido que filtrar df no loop)
    grouped_indices = headers.groupby(col_cmp)['global_trace_index'].apply(list).to_dict()
    
    # 5. Loop de Stack
    for i, cmp_val in enumerate(tqdm(unique_cmps, desc="Stacking")):
        # Pega índices dos traços que pertencem a este CMP
        idxs = grouped_indices[cmp_val]
        
        # Lê o gather (apenas os traços necessários)
        gather = np.array(data_nmo[:, idxs])
        
        # --- SMART STACK ---
        # Soma as amplitudes
        sum_trace = np.sum(gather, axis=1)
        
        # Conta quantos traços contribuíram (não são zero)
        # Isso compensa o Mute: se 50% dos traços foram mutados no topo, 
        # dividimos por 50% do fold, mantendo a amplitude correta.
        live_fold = np.sum(gather != 0, axis=1)
        
        # Evita divisão por zero (onde não tem dado nenhum, fica 0)
        live_fold[live_fold == 0] = 1.0
        
        # Média normalizada
        stacked_section[:, i] = sum_trace / live_fold

    # 6. Salva Resultado
    print(f"Salvando seção empilhada em: {os.path.basename(FILE_STACK_OUT)}")
    stacked_section.tofile(FILE_STACK_OUT)
    
    return unique_cmps, stacked_section

def plot_result(cmps, section):
    print("\n--- Visualizando Resultado ---")
    plt.figure(figsize=(14, 8))
    
    # Aplica ganho t^2 apenas para visualização
    t_axis = np.arange(Nt, dtype=np.float32) * dt
    gain = t_axis ** 2.0
    view_data = section * gain[:, None]
    
    # Define contraste robusto
    vm = np.percentile(np.abs(view_data), 99.0)
    if vm <= 0: vm = 1
    
    plt.imshow(view_data, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
               extent=[cmps[0], cmps[-1], Nt*dt, 0], interpolation='bilinear')
    
    plt.title(f"Seção Empilhada (Stacked)\n{len(cmps)} CMPs | Nt={Nt} | dt={dt}s")
    plt.xlabel("CMP Number")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="Amplitude (Ganho t^2)")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    cmps, section = run_stacking()
    if section is not None:
        plot_result(cmps, section)