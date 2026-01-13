import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import correlate
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = r'C:/Users/AnaCarvs.GISIS/Desktop/Dataset'

# Arquivos
FILE_HEAD      = f'{BASE_DIR}/Linha/Trace_Headers_AP2.csv'
FILE_GATH_NMO  = f'{BASE_DIR}/Linha/Line_NMO_Corrected_AP2.bin'       
FILE_GATH_ETA  = f'{BASE_DIR}/Linha/Line_Aniso_Corrected_Final_AP2.bin'
FILE_STACK_NMO = f'{BASE_DIR}/Linha/Line_Stack_Final_AP2.bin'          
FILE_STACK_ETA = f'{BASE_DIR}/Linha/Line_Stack_Final_Eta_AP2.bin'
FILE_PICKS_ETA = f'{BASE_DIR}/Picks/Eta_Picks_AP2.csv' # Carregando Picks para Contexto

# Parâmetros
Nt = 1501
dt = 0.001
FREQ_MAX = 80.0
MAX_LAG_MS = 60 

# ==============================================================================
# 2. PROCESSAMENTO
# ==============================================================================
def stack_data_if_needed(input_bin, output_bin, headers, nt):
    if os.path.exists(output_bin):
        print(f"Stack {os.path.basename(output_bin)} já existe.")
        return
    print(f"Gerando Stack: {os.path.basename(output_bin)}...")
    data_in = np.memmap(input_bin, dtype='float32', mode='r', shape=(Nt, len(headers)), order='F')
    unique_cmps = np.sort(headers['cmp'].unique())
    n_cmps = len(unique_cmps)
    stack_data = np.zeros((Nt, n_cmps), dtype=np.float32)
    grouped = headers.groupby('cmp')
    for i, c in enumerate(tqdm(unique_cmps, desc="Stacking")):
        if c not in grouped.groups: continue
        idxs = grouped.get_group(c).index.values
        gather = np.array(data_in[:, idxs])
        if gather.shape[1] > 0:
            stack_data[:, i] = np.sum(gather, axis=1) / gather.shape[1]
    stack_data.tofile(output_bin)

def calc_normalized_correlation(gather, max_lag_samples):
    nt, n_traces = gather.shape
    pilot = np.sum(gather, axis=1) / n_traces 
    pilot = pilot - np.mean(pilot)
    pilot_std = np.std(pilot)
    if pilot_std > 1e-9: pilot /= pilot_std
    
    n_lags = 2 * max_lag_samples + 1
    corr_panel = np.zeros((n_lags, n_traces))
    
    for i in range(n_traces):
        trace = gather[:, i] - np.mean(gather[:, i])
        tr_std = np.std(trace)
        if tr_std > 1e-9:
            trace /= tr_std 
            cc = correlate(trace, pilot, mode='same') / nt
            center = len(cc) // 2
            start = center - max_lag_samples
            end   = center + max_lag_samples + 1
            if start >= 0 and end <= len(cc):
                corr_panel[:, i] = cc[start:end]
    return corr_panel

def calc_spectrum(data, dt):
    traco_medio = np.mean(data, axis=1) 
    n = len(traco_medio)
    yf = fft(traco_medio)
    xf = fftfreq(n, dt)[:n//2]
    amp = 2.0/n * np.abs(yf[0:n//2])
    amp_db = 20 * np.log10(amp + 1e-9)
    return xf, amp_db

# ==============================================================================
# 3. VISUALIZAÇÃO
# ==============================================================================

def qc_stacks_side_by_side(stk_nmo, stk_eta, diff, extent):
    fig, axs = plt.subplots(1, 3, figsize=(18, 8), sharex=True, sharey=True)
    fig.canvas.manager.set_window_title("QC 1: Comparacao de Stacks")
    
    vm = np.max(np.abs(stk_nmo))
    
    im1 = axs[0].imshow(stk_nmo, cmap='gray', vmin=-vm, vmax=vm, aspect='auto', extent=extent)
    axs[0].set_title("Stack NMO (Referência)", fontweight='bold'); axs[0].set_ylabel("Tempo [s]"); axs[0].set_xlabel("CMP")
    
    im2 = axs[1].imshow(stk_eta, cmap='gray', vmin=-vm, vmax=vm, aspect='auto', extent=extent)
    axs[1].set_title("Stack Eta (Corrigido)", fontweight='bold'); axs[1].set_xlabel("CMP")
    
    vm_diff = np.max(np.abs(diff))
    im3 = axs[2].imshow(diff, cmap='seismic', vmin=-vm_diff, vmax=vm_diff, aspect='auto', extent=extent)
    axs[2].set_title("Diferença", fontweight='bold'); axs[2].set_xlabel("CMP")
    
    for ax, im in zip(axs, [im1, im2, im3]):
        div = make_axes_locatable(ax)
        cax = div.append_axes("bottom", size="2%", pad=0.3)
        plt.colorbar(im, cax=cax, orientation='horizontal')
    plt.tight_layout()
    plt.show(block=False)

def qc_traces_detailed(stk_nmo, stk_eta, cmps, t, indices_qc):
    fig, axs = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    fig.canvas.manager.set_window_title("QC 2: Analise de Traco")
    
    for i, idx in enumerate(indices_qc):
        if idx >= stk_nmo.shape[1]: continue
        tr_nmo = stk_nmo[:, idx]; tr_eta = stk_eta[:, idx]; diff = tr_eta - tr_nmo
        cmp_val = cmps[idx]
        max_amp = max(np.max(np.abs(tr_nmo)), np.max(np.abs(tr_eta))) or 1
        
        axs[i].fill_betweenx(t, 0, diff, color='gray', alpha=0.3)
        axs[i].plot(tr_nmo, t, 'b-', lw=0.8, label='NMO')
        axs[i].plot(tr_eta, t, 'r--', lw=1.0, label='Eta')
        axs[i].set_title(f"CMP {int(cmp_val)}")
        axs[i].set_xlim(-max_amp*1.1, max_amp*1.1)
        axs[i].set_ylim(t[-1], 0)
        axs[i].grid(True, alpha=0.3)
        if i==0: axs[i].legend(loc='upper right'); axs[i].set_ylabel("Tempo [s]")
    plt.tight_layout()
    plt.show(block=False)

def qc_spectrum(stk_nmo, stk_eta, dt):
    freq, amp_nmo = calc_spectrum(stk_nmo, dt)
    _, amp_eta = calc_spectrum(stk_eta, dt)
    
    plt.figure(figsize=(8, 5))
    plt.title(f"QC 3: Comparação Espectral (0-{FREQ_MAX}Hz)")
    plt.plot(freq, amp_nmo, 'b-', label='Stack NMO')
    plt.plot(freq, amp_eta, 'r--', label='Stack Eta')
    plt.xlim(0, FREQ_MAX); plt.ylim(-60, np.max(amp_eta)+5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("Frequência [Hz]"); plt.ylabel("Amplitude [dB]")
    plt.show(block=False)

def qc_cmp_context(cmp_val, off, t_axis, g_nmo, g_eta, picks=[]):
    """ QC 4: CMP com Contexto de Picks """
    diff = g_eta - g_nmo
    lag_samples = int(MAX_LAG_MS / 1000 / 0.001)
    
    corr_nmo = calc_normalized_correlation(g_nmo, lag_samples)
    corr_eta = calc_normalized_correlation(g_eta, lag_samples)
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"ANÁLISE DETALHADA: CMP {int(cmp_val)}", fontsize=16, fontweight='bold')
    fig.canvas.manager.set_window_title(f"QC CMP {int(cmp_val)}")
    
    gs = fig.add_gridspec(2, 3)
    
    ext_g = [off.min(), off.max(), t_axis[-1], 0]
    ext_c = [off.min(), off.max(), -MAX_LAG_MS, MAX_LAG_MS]
    vm = np.percentile(np.abs(g_nmo), 98)
    
    # --- Gathers ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(g_nmo, cmap='gray', vmin=-vm, vmax=vm, aspect='auto', extent=ext_g)
    ax1.set_title("Input: NMO Gather", fontweight='bold'); ax1.set_ylabel("Tempo [s]")
    
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax2.imshow(g_eta, cmap='gray', vmin=-vm, vmax=vm, aspect='auto', extent=ext_g)
    ax2.set_title("Output: Eta Gather", fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax3.imshow(diff, cmap='seismic', vmin=-vm/2, vmax=vm/2, aspect='auto', extent=ext_g)
    ax3.set_title("Residual (Diferença)", fontweight='bold')
    
    # --- Overlay dos Picks (O Contexto que faltava) ---
    for ax in [ax1, ax2, ax3]:
        # Desenha linhas onde houve pick
        for t_pick, eta_val in picks:
            ax.axhline(t_pick, color='lime', linestyle='--', linewidth=1.5, alpha=0.8)
            if ax == ax1: # Só coloca texto no primeiro para não poluir
                ax.text(off.max(), t_pick-0.01, f" Pick: {eta_val:.3f}", color='lime', 
                        fontsize=9, fontweight='bold', ha='right')
        ax.grid(True, alpha=0.2, color='cyan')

    # --- Correlação ---
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(corr_nmo, cmap='jet', vmin=0, vmax=1, aspect='auto', extent=ext_c, origin='lower')
    ax4.set_title("Correlação NMO", fontweight='bold'); ax4.set_ylabel("Lag [ms]"); ax4.set_xlabel("Offset [m]")
    ax4.axhline(0, color='white', linestyle='--', alpha=0.7)
    
    ax5 = fig.add_subplot(gs[1, 1], sharey=ax4)
    im5 = ax5.imshow(corr_eta, cmap='jet', vmin=0, vmax=1, aspect='auto', extent=ext_c, origin='lower')
    ax5.set_title("Correlação Eta (Alvo: 0ms)", fontweight='bold'); ax5.set_xlabel("Offset [m]")
    ax5.axhline(0, color='white', linestyle='--', alpha=0.7)
    
    cbar_ax = fig.add_axes([0.65, 0.15, 0.015, 0.3])
    fig.colorbar(im5, cax=cbar_ax, label='Coeficiente de Correlação')
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show(block=False)

# ==============================================================================
# 4. EXECUTOR
# ==============================================================================
def run_final_qc_suite():
    print("--- GERANDO QCS CONTEXTUALIZADOS ---")
    
    try:
        h = pd.read_csv(FILE_HEAD)
        h.columns = h.columns.str.strip().str.lower()
        if 'cmp_x' in h.columns: h.rename(columns={'cmp_x':'cmp'}, inplace=True)
        if 'offset_x' in h.columns: h.rename(columns={'offset_x':'offset'}, inplace=True)
    except Exception as e: print(e); return

    if not os.path.exists(FILE_GATH_ETA) or not os.path.exists(FILE_GATH_NMO):
        print("ERRO: Gathers binários não encontrados."); return

    # Carrega Picks para dar contexto
    picks_dict = {}
    if os.path.exists(FILE_PICKS_ETA):
        print(">> Carregando Picks de Eta...")
        df_picks = pd.read_csv(FILE_PICKS_ETA)
        for c in df_picks['CMP'].unique():
            # Guarda lista de tuplas (tempo, eta)
            picks_dict[int(c)] = list(zip(df_picks[df_picks['CMP']==c]['Time'], df_picks[df_picks['CMP']==c]['Eta']))
    else:
        print("AVISO: Arquivo de picks não encontrado. Gathers sem linhas guia.")

    stack_data_if_needed(FILE_GATH_ETA, FILE_STACK_ETA, h, Nt)
    stack_data_if_needed(FILE_GATH_NMO, FILE_STACK_NMO, h, Nt)
    
    unique_cmps = np.sort(h['cmp'].unique())
    n_cmps = len(unique_cmps)
    stk_nmo = np.array(np.memmap(FILE_STACK_NMO, dtype='float32', mode='r', shape=(Nt, n_cmps), order='C'))
    stk_eta = np.array(np.memmap(FILE_STACK_ETA, dtype='float32', mode='r', shape=(Nt, n_cmps), order='C'))
    diff_stk = stk_eta - stk_nmo
    
    # 1. Stacks
    print(">> QC 1: Stacks...")
    extent_stk = [unique_cmps[0], unique_cmps[-1], Nt*dt, 0]
    qc_stacks_side_by_side(stk_nmo, stk_eta, diff_stk, extent_stk)
    
    # 2. Traços
    print(">> QC 2: Traços...")
    idxs_qc = [int(n_cmps*0.2), int(n_cmps*0.5), int(n_cmps*0.8)]
    qc_traces_detailed(stk_nmo, stk_eta, unique_cmps, np.arange(Nt)*dt, idxs_qc)
    
    # 3. Espectro
    print(">> QC 3: Espectro...")
    qc_spectrum(stk_nmo, stk_eta, dt)
    
    # 4. CMP Gathers
    print(">> QC 4: CMP Gathers Detalhados...")
    m_nmo = np.memmap(FILE_GATH_NMO, dtype='float32', mode='r', shape=(Nt, len(h)), order='F')
    m_eta = np.memmap(FILE_GATH_ETA, dtype='float32', mode='r', shape=(Nt, len(h)), order='F')
    grouped = h.groupby('cmp')
    
    qc_cmps_vals = [unique_cmps[i] for i in idxs_qc]
    
    for c in qc_cmps_vals:
        if c not in grouped.groups: continue
        idxs = grouped.get_group(c).index.values
        offsets = h.loc[idxs, 'offset'].values
        sort = np.argsort(np.abs(offsets))
        off_s = offsets[sort]
        g_nmo = np.array(m_nmo[:, idxs[sort]])
        g_eta = np.array(m_eta[:, idxs[sort]])
        
        # Passa os picks deste CMP para desenhar as linhas
        picks_this_cmp = picks_dict.get(int(c), [])
        
        qc_cmp_context(c, off_s, np.arange(Nt)*dt, g_nmo, g_eta, picks_this_cmp)
        
    print(">> Fim. Feche as janelas para sair.")
    plt.show()

if __name__ == "__main__":
    run_final_qc_suite()