import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURAÇÃO =================
# Coloque aqui o caminho do seu arquivo binário
ARQUIVO_BIN = r"C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master\outputs\seismograms\AP\VTIseismogram_shot_15_Nt5001_Nrec151.bin" 

# Parâmetros de Tempo (Sempre fixos na modelagem)
Nt = 5001       # Número de amostras de tempo
dt = 0.0005      # Taxa de amostragem (s)

# ================= LEITURA AUTOMÁTICA =================

def visualizar_tiro(path):
    if not os.path.exists(path):
        print(f"ERRO: Arquivo não encontrado: {path}")
        return

    # 1. Descobre o Nrec automaticamente pelo tamanho do arquivo
    # Tamanho = Nt * Nrec * 4 bytes (float32)
    file_size = os.path.getsize(path)
    bytes_per_trace = Nt * 4
    nrec = file_size // bytes_per_trace
    
    print(f"--- Lendo Arquivo ---")
    print(f"Arquivo: {os.path.basename(path)}")
    print(f"Tamanho: {file_size} bytes")
    print(f"Geometria Detectada: {Nt} amostras x {nrec} traços")

    # 2. Leitura dos Dados
    try:
        raw_data = np.fromfile(path, dtype=np.float32)
        
        # Reshape para (Nt, Nrec) -> Time-Sequential
        # Se a imagem ficar "riscada" na horizontal, troque para order='F' ou transponha .T
        data = raw_data.reshape((Nt, nrec), order='C')
        
    except Exception as e:
        print(f"Erro ao ler binário: {e}")
        return

    # 3. Plotagem (QC)
    plt.figure(figsize=(10, 8))
    
    # Calcula ganho para visualização (Clip em 98%)
    vm = np.percentile(np.abs(data), 98)
    
    # Extent ajusta os eixos: [Xmin, Xmax, Ymax, Ymin] (Y invertido)
    plt.imshow(data, aspect='auto', cmap='gray', vmin=-vm, vmax=vm,
               extent=[0, nrec, Nt*dt, 0])
    
    plt.title(f"Visualização Raw: {os.path.basename(path)}\n({nrec} Canais)")
    plt.xlabel("Número do Traço (Canal)")
    plt.ylabel("Tempo (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_tiro(ARQUIVO_BIN)