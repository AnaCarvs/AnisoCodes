import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
SCRIPT 00: Geração de Geometria Marine 2D (Lado Direito / Positive Offsets)
----------------------------------------------------------------------------

Objetivo:
- Gerar as coordenadas absolutas de fontes e receptores para uma aquisição 
  sísmica marine 2D com cabo (streamer) e navio se movendo. 
- Criar um grid de receptores (superset) que cubra toda a área iluminada.
- Exportar arquivos CSV padronizados para importação em softwares de
    modelagem ou processamento sísmico.

----------------------------------------------------------------------------
Técnica:
- Define parâmetros físicos do modelo (limites, profundidades).
- Calcula posições absolutas das fontes com base na navegação do navio.
- Gera um grid de receptores que cobre todos os offsets possíveis.
- Simula a aquisição tiro-a-tiro para calcular o fold (cobertura).
- Gera gráficos de controle de qualidade (QC) e uma tabela resumo.

----------------------------------------------------------------------------
Parâmetros Editáveis:
- Domínio do Modelo (Lx, Zsrc, Zrec)
- Configuração do Navio/Cabo (SPI, GPI, Near, Cable Length)
- Navegação (First Shot, Last Shot)
- Output (Diretórios e Nomes de Arquivos)
----------------------------------------------------------------------------
Observações:
- A geometria gerada considera apenas o lado direito (offsets positivos).
- O script cria automaticamente a pasta de output se não existir.
- Requer bibliotecas: numpy, pandas, matplotlib, os

-------------------------------------------------------------------------------
Autor: Ana Paula Carvalhos
Data: Dezembro de 2025
Versão: 1.0
-------------------------------------------------------------------------------

"""

# ==============================================================================
# 1. PARÂMETROS DO PROJETO (CONFIGURAÇÃO FÍSICA)
# ==============================================================================

# --- Domínio do Modelo (Limites da Área) ---
MODEL_X_MIN = 0.0       # Início do modelo em metros
MODEL_X_MAX = 25000.0   # Fim do modelo em metros
MODEL_Z_SRC = 5.0       # Profundidade da Fonte (m)
MODEL_Z_REC = 10.0      # Profundidade do Receptor (m)

# --- Configuração Navio/Cabo (Geometria Relativa) ---
SHOT_INT  = 200.0      # Intervalo de Tiro (Flip-Flop ou Simples) em metros
GROUP_INT = 25.0       # Distância entre canais (Group Interval) em metros

# Definição do Cabo (Streamer)
NEAR_OFFSET = 100.0    # Distância da fonte ao 1º canal. 
                       # IMPORTANTE: Valor positivo garante geometria "End-On" (Puxando).
CABLE_LEN   = 8000.0   # Comprimento ativo do cabo em metros.

# --- Navegação (Geometria Absoluta) ---
FIRST_SHOT = 8100.0    # Coordenada X do primeiro tiro
LAST_SHOT  = 16900.0   # Coordenada X do último tiro

# --- Output (Diretórios e Arquivos) ---
# Cria a pasta automaticamente se não existir
OUTPUT_DIR = r"Inputs/Geometry"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

FILE_SRC   = "sources_T2.csv"
FILE_REC   = "receivers_T2.csv"
PLOT_NAME  = "QC_Fold_T2.png"
TABLE_NAME = "QC_Tabela_Aquisicao_T2.png"

# ==============================================================================
# 2. MOTOR DE GERAÇÃO (GEOMETRIA)
# ==============================================================================

def generate_marine_geometry():
    """
    Calcula as coordenadas absolutas de todas as fontes e cria um grid de 
    receptores (Superset) que cobre toda a área iluminada pela aquisição.
    
    Returns:
        sx (np.array): Array com coordenadas X das fontes.
        rx (np.array): Array com coordenadas X dos receptores (Superset).
    """
    print(f"\n--- Gerando Geometria Marine 2D (Lado Direito / Positive Offsets) ---")
    
    # 1. Gerar Coordenadas das Fontes (Sources)
    # Calcula quantos tiros cabem entre First e Last shot
    n_shots = int(np.floor((LAST_SHOT - FIRST_SHOT) / SHOT_INT)) + 1
    sx = np.linspace(FIRST_SHOT, FIRST_SHOT + (n_shots-1)*SHOT_INT, n_shots)

    # 2. Gerar Grid de Receptores (Receivers)
    # Lógica: Como é Marine, os receptores se movem. Precisamos criar um "Superset"
    # de estações fixas no fundo (conceitual) que representem todas as posições 
    # onde um hidrofone passará.
    
    max_offset_val = NEAR_OFFSET + CABLE_LEN
    
    # O grid deve começar na frente do 1º tiro (Offset Mínimo)
    min_rx_pos = sx[0] + NEAR_OFFSET 
    # O grid deve terminar na ponta do cabo do último tiro (Offset Máximo)
    max_rx_pos = sx[-1] + max_offset_val 
    
    # Ajusta o grid para ser múltiplo exato do GROUP_INT (Snap to Grid)
    grid_start = np.floor(min_rx_pos / GROUP_INT) * GROUP_INT
    grid_end   = np.ceil(max_rx_pos / GROUP_INT) * GROUP_INT
    
    # Cria o array de receptores
    n_recs = int(np.round((grid_end - grid_start) / GROUP_INT)) + 1
    rx = np.linspace(grid_start, grid_start + (n_recs-1)*GROUP_INT, n_recs)
    
    # Clip: Garante que não tenhamos coordenadas fora dos limites do modelo físico
    rx = rx[(rx >= MODEL_X_MIN) & (rx <= MODEL_X_MAX)]
    
    print(f"-> Intervalo de Receptores Gerado: {np.min(rx):.1f}m a {np.max(rx):.1f}m")
    
    return sx, rx

# ==============================================================================
# 3. QC E RELATÓRIO (VISUALIZAÇÃO)
# ==============================================================================

def qc_industry_standard(sx, rx):
    """
    Simula a aquisição tiro-a-tiro para calcular o Fold (cobertura) e 
    gerar os gráficos de controle de qualidade (QC).
    
    Args:
        sx (array): Coordenadas das fontes.
        rx (array): Coordenadas do grid de receptores.
    """
    print("\n--- Gerando Relatório Gráfico e Calculando Fold... ---")
    
    # --- A. Definição do Cabo Teórico ---
    # Quantos canais físicos existem no cabo?
    num_channels = int(CABLE_LEN / GROUP_INT) + 1 
    
    # Cria os offsets relativos (ex: [100, 125, 150 ... 8100])
    cable_offsets = np.linspace(NEAR_OFFSET, NEAR_OFFSET + CABLE_LEN, num_channels)
    
    # Listas para armazenar os pontos de reflexão (CMPs) de toda a linha
    cmps_list = []
    offsets_list = []
    
    # --- B. Simulação de "Rolling Spread" ---
    # Para cada tiro, calculamos onde caem os receptores e calculamos o CMP
    for shot in sx:
        # Posição Teórica: Onde os hidrofones estariam para este tiro?
        # (Shot + Offset) garante que o receptor está à frente (Lado Direito)
        current_rec_pos = shot + cable_offsets
        
        # Validação: Checa se essas posições teóricas caem dentro do grid 'rx' gerado
        # e dentro dos limites do modelo. (Tolerância de 0.1m para erro de float)
        mask = (current_rec_pos >= np.min(rx) - 0.1) & (current_rec_pos <= np.max(rx) + 0.1)
        
        # Filtra apenas os válidos
        valid_recs = current_rec_pos[mask]
        valid_offsets = cable_offsets[mask] 
        
        # Fórmula do CMP: Ponto médio entre Fonte e Receptor
        curr_cmps = (shot + valid_recs) / 2.0
        
        # Acumula na lista geral
        cmps_list.extend(curr_cmps)
        offsets_list.extend(valid_offsets)

    cmps_arr = np.array(cmps_list)
    offsets_arr = np.array(offsets_list)

    if len(cmps_arr) == 0:
        print("ERRO CRÍTICO: Nenhum CMP gerado. Verifique se Navio+Cabo cabem dentro do Modelo.")
        return

    # --- C. Cálculo do Fold (Histograma) ---
    cmp_bin = GROUP_INT / 2.0  # Regra padrão: Bin size é metade do receiver interval
    
    # Cria os bins (caixinhas) ao longo da linha
    bins = np.arange(np.min(rx), np.max(rx), cmp_bin)
    
    # Conta quantos CMPs caíram em cada bin
    fold_counts, bin_edges = np.histogram(cmps_arr, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Estatísticas de Fold
    max_fold = np.max(fold_counts)
    
    # Identifica onde começa e termina o "Full Fold" (Cobertura completa)
    # Consideramos Full Fold onde a cobertura é >= Max Fold - 0.5
    plateau_idx = np.where(fold_counts >= (max_fold - 0.5))[0]
    
    if len(plateau_idx) > 0:
        start_full_fold = bin_centers[plateau_idx[0]]
        end_full_fold   = bin_centers[plateau_idx[-1]]
        width_full_fold = end_full_fold - start_full_fold
    else:
        start_full_fold, end_full_fold, width_full_fold = 0, 0, 0

    # --- D. Plotagem (Matplotlib) ---
    
    # Configura Figura 1: Gráficos de Linha
    fig_plots, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.2)
    
    # Gráfico 1: Stacking Chart (Spider Plot)
    # Mostra a distribuição de offsets por CMP. Deve formar triângulos/losangos.
    ax1.set_title("Stacking Chart (Apenas Lado Direito / Offsets Positivos)", fontsize=12, fontweight='bold')
    
    # Otimização: Se tiver muitos pontos, plota apenas 1 a cada 'step' para não travar
    step = 10 if len(cmps_arr) > 10000 else 1
    ax1.scatter(cmps_arr[::step], offsets_arr[::step], c='blue', s=1, alpha=0.3, label='Traços')
    
    # Linhas de referência da zona de interesse
    ax1.axvline(start_full_fold, color='green', linestyle='--', linewidth=2, label='Início Full Fold')
    ax1.axvline(end_full_fold, color='green', linestyle='--', linewidth=2, label='Fim Full Fold')
    
    ax1.set_ylabel("Offset (m)")
    ax1.set_ylim(0, CABLE_LEN + 500)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.5)
    
    # Gráfico 2: Diagrama de Fold
    ax2.set_title(f"Diagrama de Fold (Máximo: {max_fold})", fontsize=12, fontweight='bold')
    ax2.fill_between(bin_centers, fold_counts, color='gold', edgecolor='black', alpha=0.8)
    ax2.axvline(start_full_fold, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(end_full_fold, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Fold (Cobertura)")
    ax2.set_xlabel("Posição do CMP (m)")
    ax2.grid(True, which='both', alpha=0.5)
    
    fig_plots.tight_layout()
    
    # Salva Figura 1
    plot_path = os.path.join(OUTPUT_DIR, PLOT_NAME)
    fig_plots.savefig(plot_path, dpi=300)
    print(f"Gráfico de QC salvo em: {plot_path}")

    # --- E. Tabela de Relatório ---
    fig_table = plt.figure(figsize=(6, 5)) 
    ax_table = fig_table.add_subplot(111)
    ax_table.axis('off') # Remove eixos, queremos só a tabela
    ax_table.set_title("Parâmetros de Geometria (Right Leg)", fontsize=14, fontweight='bold', pad=20)
    
    # Dados da tabela
    table_data = [
        ["Total de Tiros (Sources)", f"{len(sx)}"],
        ["Total de Canais (Ativos/Tiro)", f"{num_channels}"],
        ["Intervalo de Tiro (SPI)", f"{SHOT_INT:.1f} m"],
        ["Intervalo de Receptor (GPI)", f"{GROUP_INT:.1f} m"],
        ["Offset Mínimo (Near)", f"+{NEAR_OFFSET:.1f} m"],
        ["Offset Máximo (Far)", f"+{NEAR_OFFSET + CABLE_LEN:.1f} m"],
        ["Bin de CMP", f"{cmp_bin:.1f} m"],
        ["Fold Máximo Alcançado", f"{int(max_fold)}"],
        ["Zona de Full Fold (Largura)", f"{width_full_fold:.1f} m"],
        ["Limites Full Fold (CMP)", f"{start_full_fold:.0f} - {end_full_fold:.0f} m"]
    ]
    
    # Criação da tabela visual
    tbl = ax_table.table(cellText=table_data, 
                         colLabels=["Parâmetro", "Valor"], 
                         loc='center', 
                         cellLoc='center',
                         colWidths=[0.5, 0.4])
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2) # Aumenta altura das linhas
    
    # Estilização do Header da tabela
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')

    fig_table.tight_layout()
    
    # Salva Figura 2
    table_path = os.path.join(OUTPUT_DIR, TABLE_NAME)
    fig_table.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"Tabela de parâmetros salva em: {table_path}")
    
    print("\nExibindo janelas... (Feche a janela do gráfico para encerrar o script)")
    plt.show()

# ==============================================================================
# 4. FUNÇÃO DE EXPORTAÇÃO CSV
# ==============================================================================

def save_csv(sx, rx):
    """
    Salva as coordenadas geradas em arquivos CSV padronizados para importação
    em softwares de modelagem ou processamento.
    """
    # Exporta Sources
    df_src = pd.DataFrame({
        'index': range(1, len(sx)+1), 
        'coordx': sx, 
        'coordz': [MODEL_Z_SRC]*len(sx)
    })
    src_path = os.path.join(OUTPUT_DIR, FILE_SRC)
    df_src.to_csv(src_path, index=False)
        
    # Exporta Receivers (Superset)
    df_rec = pd.DataFrame({
        'index': range(1, len(rx)+1), 
        'coordx': rx, 
        'coordz': [MODEL_Z_REC]*len(rx)
    })
    rec_path = os.path.join(OUTPUT_DIR, FILE_REC)
    df_rec.to_csv(rec_path, index=False)
        
    print(f"Arquivos CSV exportados com sucesso para: {OUTPUT_DIR}")

# ==============================================================================
# EXECUÇÃO DO PROGRAMA
# ==============================================================================
if __name__ == "__main__":
    # 1. Gera a geometria
    s, r = generate_marine_geometry()
    
    # 2. Salva os dados brutos
    save_csv(s, r)
    
    # 3. Gera os gráficos de QC e Tabela
    qc_industry_standard(s, r)