import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import griddata
import os
import gc
from tqdm import tqdm
from numba import jit, prange

"""
SCRIPT 02: VELOCITY ANALYSIS INTERATIVO
===================================================
Objetivo:
1. Permitir picking interativo de curvas de velocidade RMS em CMPs.
2. Gerar modelo de velocidade 2D por interpolação dos picks.
3. Aplicar NMO com o modelo de velocidade gerado.
Técnica:
- Usa semblance para auxiliar no picking.
- Usa Numba para acelerar cálculos pesados.
- Permite navegação por CMPs e ajuste de parâmetros em tempo real.
--------------------------------------------------
Parâmetros Editáveis:
- BASE_DIR: Diretório base dos arquivos de entrada/saída.
- F_BIN_IN: Arquivo binário de entrada (CMP sorted).
- F_HEAD: Arquivo CSV de cabeçalhos de traço.
- F_OUT_PICKS: Arquivo CSV de picks de velocidade.
- F_OUT_VEL: Arquivo binário do modelo de velocidade 2D.
- F_OUT_NMO: Arquivo binário da seção NMO corrigida.
- Nt, dt: Número de amostras e intervalo de tempo.
- CMP_BIN_SIZE, CMP_STEP: Tamanho do bin CMP e passo de navegação.
- INIT_VMIN, INIT_VMAX, INIT_DV: Parâmetros iniciais de velocidade.
- INIT_WIN, INIT_GAIN: Parâmetros iniciais de janela de semblance e ganho.
- VEL_MUTE, BUF_MUTE: Parâmetros de mute baseados em velocidade.
- STRETCH_LIM: Limite de stretch para NMO.
- WIN_SEMBLANCE: Janela de semblance em segundos.
------------------------------------------------
Observações:
- Use o botão "Salvar" para salvar os picks atuais.
- Use o botão "EXPORTAR" para gerar o modelo de velocidade e aplicar NMO.
- Use o botão "RESETAR" para limpar todos os picks.
- Clique com o botão esquerdo para adicionar um pick, clique com o botão direito para remover o pick mais próximo.
---------------------------------------------------
Referências:
- Numba: https://numba.pydata.org/
- Matplotlib Widgets: https://matplotlib.org/stable/users/explain/widgets.html
- Seismic NMO Theory: Yilmaz, O. (2001). Seismic Data Analysis.
- Semblance Analysis: Taner, M. T., & Koehler, F. (1969). Velocity spectra-Digital computer derivation and applications of velocity functions.
---------------------------------------------------

Autor: Ana Paula Carvalhos
Data: Dezembro de 2025
Versão: 1.0
---------------------------------------------------
"""

# ================= CONFIGURAÇÃO =================
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'

F_BIN_IN = f'{BASE_DIR}/outputs/Line_CMP_Sorted.bin'
F_HEAD   = f'{BASE_DIR}/outputs/Trace_Headers.csv'
F_OUT_PICKS = f'{BASE_DIR}/outputs/Velocity_Picks.csv'
F_OUT_VEL   = f'{BASE_DIR}/outputs/Velocity_Model_2D.bin'
F_OUT_NMO   = f'{BASE_DIR}/outputs/Line_NMO_Corrected.bin'

Nt = 12001           
dt = 0.0005          

CMP_BIN_SIZE = 12.5
CMP_STEP = 100.0     

INIT_VMIN = 1300.0
INIT_VMAX = 5500.0
INIT_DV   = 25.0     
INIT_WIN  = 40.0     
INIT_GAIN = 1.0    
VEL_MUTE  = 1500.0     
BUF_MUTE  = 200      
STRETCH_LIM = 60.0    
WIN_SEMBLANCE = 0.4 

V_MIN, V_MAX = INIT_VMIN, INIT_VMAX

# ================= KERNELS (NUMBA) =================

@jit(nopython=True, parallel=True, fastmath=True)
def numba_semblace(data, offsets, times, vels, dt, win_len):
    nt, nrec = data.shape
    n_vel = len(vels)
    semb = np.zeros((nt, n_vel), dtype=np.float32)
    off2 = offsets**2
    half_w = win_len // 2
    
    for iv in prange(n_vel):
        v = vels[iv]
        v2 = v**2 + 1e-9
        num_trace = np.zeros(nt, dtype=np.float32)
        den_trace = np.zeros(nt, dtype=np.float32)
        
        for it in range(nt):
            t0 = times[it]
            sum_amp = 0.0
            sum_sq = 0.0
            for ir in range(nrec):
                t_hyp = np.sqrt(t0**2 + off2[ir]/v2)
                if (t_hyp - t0)/(t0 + 1e-9) > 0.5: continue
                idx = int(round(t_hyp/dt))
                if idx < nt:
                    val = data[idx, ir]
                    sum_amp += val
                    sum_sq += val*val
            num_trace[it] = sum_amp * sum_amp
            den_trace[it] = sum_sq
            
        for it in range(nt):
            start = max(0, it - half_w)
            end = min(nt, it + half_w + 1)
            s_num = 0.0
            s_den = 0.0
            for k in range(start, end):
                s_num += num_trace[k]
                s_den += den_trace[k]
            if s_den > 1e-9:
                semb[it, iv] = s_num / (nrec * s_den)
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_nmo(data, out, offsets, times, v_rms, dt, limit):
    nt, nrec = data.shape
    for i in prange(nrec):
        off2 = offsets[i]**2
        for j in range(nt):
            t0 = times[j]
            v = v_rms[j]
            t_nmo = np.sqrt(t0**2 + off2/(v**2 + 1e-9))
            stretch = (t_nmo - t0) / (t0 + 1e-9)
            if stretch > limit:
                out[j, i] = 0.0
                continue
            idx_float = t_nmo / dt
            idx0 = int(idx_float)
            idx1 = idx0 + 1
            if idx0 >= 0 and idx1 < nt:
                w1 = idx_float - idx0
                w0 = 1.0 - w1
                out[j, i] = w0 * data[idx0, i] + w1 * data[idx1, i]
            else:
                out[j, i] = 0.0

# ================= HELPERS =================

def load_cmp_from_disk(headers, memmap, target, bin_size):
    x_min, x_max = target - bin_size/2, target + bin_size/2
    col = 'cmp' if 'cmp' in headers.columns else 'cmp_x'
    df = headers[(headers[col] >= x_min) & (headers[col] <= x_max)]
    if df.empty: return None, None, None
    
    idx_col = 'global_trace_index' if 'global_trace_index' in df.columns else 'g_idx'
    indices = df[idx_col].values.astype(np.int64)
    offsets = df['offset'].values.astype(np.float32)
    sort = np.argsort(offsets)
    data = np.array(memmap[:, indices[sort]]) 
    return data, offsets[sort], indices[sort]

def process_display(data, offsets, dt, gain_n=2.0):
    nt = data.shape[0]
    buf = int((BUF_MUTE/1000)/dt)
    arr_idx = (np.abs(offsets) / VEL_MUTE / dt).astype(int) + buf
    t_indices = np.arange(nt)[:, None]
    mask = t_indices > arr_idx[None, :]
    out = data * mask
    t = np.arange(nt, dtype=np.float32) * dt
    gained = out * (t[:, None]**gain_n)
    vm = np.percentile(np.abs(gained), 99)
    if vm > 0: gained /= vm
    return gained

def process_export_mute(data, offsets, dt):
    nt = data.shape[0]
    buf = int((BUF_MUTE/1000)/dt)
    arr_idx = (np.abs(offsets) / VEL_MUTE / dt).astype(int) + buf
    t_indices = np.arange(nt)[:, None]
    mask = t_indices > arr_idx[None, :]
    return data * mask

def run_semblance_fast(data, offsets, dt, vels, win_ms):
    times = np.arange(data.shape[0], dtype=np.float32) * dt
    win_samp = int((win_ms/1000)/dt)
    if win_samp < 1: win_samp = 1
    return numba_semblace(data, offsets, times, vels.astype(np.float32), dt, win_samp)

def run_nmo_fast(data, offsets, dt, v_rms, stretch_percent):
    nt = data.shape[0]
    out = np.zeros_like(data)
    times = np.arange(nt, dtype=np.float32) * dt
    limit = stretch_percent / 100.0
    numba_nmo(data, out, offsets, times, v_rms.astype(np.float32), dt, limit)
    return out

# ================= APP =================

class VelocitySuite:
    def __init__(self):
        print("--- INICIANDO SUITE V22 ---")
        if not os.path.exists(F_BIN_IN): return
        
        self.headers = pd.read_csv(F_HEAD)
        self.headers.columns = self.headers.columns.str.strip()
        self.memmap = np.memmap(F_BIN_IN, dtype='float32', mode='r', 
                                shape=(Nt, len(self.headers)), order='F')
        
        col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
        self.all_cmps = np.sort(self.headers[col].unique())
        self.min_cmp, self.max_cmp = self.all_cmps[0], self.all_cmps[-1]
        
        # Navegação
        step = max(1, int(CMP_STEP/(self.all_cmps[1]-self.all_cmps[0])))
        self.nav_cmps = self.all_cmps[::step]
        self.curr_idx = len(self.nav_cmps)//2
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        
        self.picks_db = {}
        self.picks_curr = []
        
        # Parâmetros
        self.p_vmin, self.p_vmax = INIT_VMIN, INIT_VMAX
        self.p_dv, self.p_win, self.p_gain = INIT_DV, INIT_WIN, INIT_GAIN
        
        self.vel_model_2d = None
        self.qc_n_panels = 5
        self.fp_nmo_qc = None
        self.fig2, self.fig3 = None, None
        self.ln_prof, self.ln_mark_curr = None, None
        
        self.t_ax = np.arange(Nt, dtype=np.float32)*dt
        self.update_v_axis()
        self.data_center = None
        
        self.init_windows()
        self.load_cmp(self.curr_cmp)

    def update_v_axis(self):
        self.v_ax = np.arange(self.p_vmin, self.p_vmax + self.p_dv, self.p_dv, dtype=np.float32)

    def init_windows(self):
        # JANELA 1 - Layout corrigido
        self.fig1 = plt.figure(figsize=(14, 9)) # Aumentei um pouco a altura total
        self.fig1.canvas.manager.set_window_title("1. Picking Interativo")
        
        # 1. Gráficos (Base em 0.40 para liberar espaço embaixo)
        base_plot = 0.40
        height_plot = 0.55
        self.ax1_sem = self.fig1.add_axes([0.05, base_plot, 0.28, height_plot])
        self.ax1_gat = self.fig1.add_axes([0.35, base_plot, 0.28, height_plot], sharey=self.ax1_sem)
        self.ax1_nmo = self.fig1.add_axes([0.65, base_plot, 0.28, height_plot], sharey=self.ax1_sem)
        
        # Fundo Painel Inferior (Área cinza de controles)
        self.fig1.patches.append(plt.Rectangle((0, 0), 1, base_plot - 0.02, 
                                               transform=self.fig1.transFigure, color='#f0f0f0', zorder=-1))
        
        # 2. MAPA (Timeline) - Y ~ 0.30
        self.ax_map = self.fig1.add_axes([0.1, 0.30, 0.8, 0.02])
        self.ax_map.set_yticks([])
        self.ax_map.set_xlim(self.min_cmp, self.max_cmp)
        self.ax_map.set_xlabel("Cobertura da Linha", fontsize=8)
        self.ax_map.plot(self.all_cmps, np.zeros_like(self.all_cmps), 'o', color='lightgray', markersize=2)
        self.ln_map_done, = self.ax_map.plot([], [], 'o', color='green', markersize=4)
        self.ln_map_curr, = self.ax_map.plot([self.curr_cmp], [0], 'o', color='red', markersize=8, markeredgecolor='k')
        
        # 3. NAVEGAÇÃO - Y ~ 0.22 (Logo abaixo do mapa)
        y_nav = 0.20
        self.btn_prev = Button(self.fig1.add_axes([0.40, y_nav, 0.04, 0.04]), "<<")
        self.btn_prev.on_clicked(self.on_prev_cmp)
        
        self.txt_goto = TextBox(self.fig1.add_axes([0.43, y_nav, 0.08, 0.04]), "CMP:", initial=str(int(self.curr_cmp)))
        self.txt_goto.on_submit(self.on_text_goto)
        
        self.btn_next = Button(self.fig1.add_axes([0.54, y_nav, 0.04, 0.04]), ">>")
        self.btn_next.on_clicked(self.on_next_cmp)
        
        # 4. LINHA 1 DE CONTROLES: PARÂMETROS - Y ~ 0.13
        # Espaçamos horizontalmente para caber os labels "Min:", "Max:", etc.
        y_p = 0.13
        h_ctl = 0.035
        
        # Labels do TextBox ficam à esquerda do eixo. Deixamos margem.
        self.txt_vmin = TextBox(self.fig1.add_axes([0.08, y_p, 0.06, h_ctl]), "Min:", initial=str(self.p_vmin))
        self.txt_vmax = TextBox(self.fig1.add_axes([0.19, y_p, 0.06, h_ctl]), "Max:", initial=str(self.p_vmax))
        self.txt_dv   = TextBox(self.fig1.add_axes([0.31, y_p, 0.05, h_ctl]), "Step:", initial=str(self.p_dv))
        self.txt_win  = TextBox(self.fig1.add_axes([0.43, y_p, 0.05, h_ctl]), "Win:", initial=str(self.p_win))
        self.txt_gain = TextBox(self.fig1.add_axes([0.55, y_p, 0.05, h_ctl]), "Gain:", initial=str(self.p_gain))
        
        self.btn_upd = Button(self.fig1.add_axes([0.65, y_p, 0.08, h_ctl]), "Update", color='lightblue')
        self.btn_upd.on_clicked(self.on_config_update)
        
        # 5. LINHA 2 DE CONTROLES: AÇÕES - Y ~ 0.04 (Fundo da tela)
        y_a = 0.04
        w_btn = 0.10
        spacing = 0.15 # Espaço entre centros dos botões
        
        self.btn_save = Button(self.fig1.add_axes([0.15, y_a, w_btn, h_ctl]), "Salvar", color='lightgreen')
        self.btn_save.on_clicked(self.save_picks_disk)
        
        self.btn_exp = Button(self.fig1.add_axes([0.35, y_a, w_btn, h_ctl]), "EXPORTAR", color='gold')
        self.btn_exp.on_clicked(self.export_data)
        
        self.btn_rst = Button(self.fig1.add_axes([0.55, y_a, w_btn, h_ctl]), "RESETAR", color='salmon')
        self.btn_rst.on_clicked(self.reset_all)
        
        self.btn_cls = Button(self.fig1.add_axes([0.75, y_a, w_btn, h_ctl]), "FECHAR", color='#ffcccc')
        self.btn_cls.on_clicked(lambda e: plt.close('all'))
        
        # --- Configs Gráficos ---
        self.im_sem = self.ax1_sem.imshow(np.zeros((10,10)), aspect='auto', cmap='jet', vmin=0, vmax=1)
        self.ln_pick, = self.ax1_sem.plot([], [], 'r-o', lw=2.5, markersize=6, zorder=100)
        
        self.im_gat = self.ax1_gat.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.im_nmo = self.ax1_nmo.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.hyp_lines = []
        
        self.ax1_sem.set_title("Semblance"); self.ax1_sem.set_xlabel("Vel (m/s)"); self.ax1_sem.set_ylabel("Tempo (s)")
        self.ax1_gat.set_title("Input"); self.ax1_gat.set_xlabel("Offset (m)")
        self.ax1_nmo.set_title("NMO Preview"); self.ax1_nmo.set_xlabel("Offset (m)")
        
        for ax in [self.ax1_sem, self.ax1_gat, self.ax1_nmo]: ax.set_ylim(self.t_ax[-1], 0)
        self.fig1.canvas.mpl_connect('button_press_event', self.on_pick)

    def load_cmp(self, cmp_val):
        print(f"CMP {int(cmp_val)}...")
        raw, off, _ = load_cmp_from_disk(self.headers, self.memmap, cmp_val, CMP_BIN_SIZE)
        if raw is None: 
            print("CMP Vazio.")
            return
        
        self.offsets = off
        self.data_center = process_display(raw, off, dt, self.p_gain)
        self.semb = run_semblance_fast(self.data_center, off, dt, self.v_ax, self.p_win)
        
        vm_s = np.percentile(self.semb, 99.5)
        self.im_sem.set_data(self.semb)
        self.im_sem.set_extent([self.p_vmin, self.p_vmax, self.t_ax[-1], 0])
        self.im_sem.set_clim(0, vm_s)
        self.ax1_sem.set_xlim(self.p_vmin, self.p_vmax)
        
        self.im_gat.set_data(self.data_center)
        self.im_gat.set_extent([off[0], off[-1], self.t_ax[-1], 0])
        self.im_gat.set_clim(-1, 1)
        self.ax1_gat.set_xlim(off[0], off[-1])
        
        self.im_nmo.set_extent([off[0], off[-1], self.t_ax[-1], 0])
        self.im_nmo.set_clim(-1, 1)
        self.ax1_nmo.set_xlim(off[0], off[-1])
        
        # Carrega picks existentes
        self.picks_curr = list(self.picks_db.get(int(cmp_val), []))
        
        self.update_dynamic()
        self.ln_map_curr.set_xdata([cmp_val])
        
        keys = np.array(list(self.picks_db.keys()))
        valid_keys = [k for k in keys if len(self.picks_db[k]) > 0]
        if valid_keys:
            self.ln_map_done.set_data(valid_keys, np.zeros(len(valid_keys)))
        
        self.fig1.canvas.draw_idle()

    def update_dynamic(self):
        if self.data_center is None: return
        self.picks_curr.sort(key=lambda x: x[0])
        
        if self.picks_curr:
            ts, vs = zip(*self.picks_curr)
            self.ln_pick.set_data(vs, ts)
            v_curve = np.interp(self.t_ax, ts, vs, left=vs[0], right=vs[-1])
        else:
            self.ln_pick.set_data([], [])
            v_curve = np.full(Nt, 2000.0)
        
        d_nmo = run_nmo_fast(self.data_center, self.offsets, dt, v_curve, 1000.0)
        self.im_nmo.set_data(d_nmo)
        
        if hasattr(self, 'ln_prof') and self.ln_prof is not None:
            if self.fig2 and plt.fignum_exists(self.fig2.number):
                self.ln_prof.set_data(v_curve, self.t_ax)
                if hasattr(self, 'ln_mark_curr'): self.ln_mark_curr.set_xdata([self.curr_cmp])
                self.fig2.canvas.draw_idle()
        
        [l.remove() for l in self.hyp_lines]
        self.hyp_lines = []
        for t0, vrms in self.picks_curr:
            tx = np.sqrt(t0**2 + (self.offsets**2)/(vrms**2))
            l, = self.ax1_gat.plot(self.offsets, tx, 'r', lw=1.5, alpha=0.8)
            self.hyp_lines.append(l)
        self.fig1.canvas.draw_idle()

    # --- NAVEGAÇÃO ---

    def change_cmp_index(self, idx):
        # 1. Salva estado ATUAL antes de mudar (Cópia)
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        
        # 2. Muda índice
        idx = max(0, min(len(self.nav_cmps)-1, idx))
        self.curr_idx = idx
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        
        # 3. Atualiza UI
        self.txt_goto.set_val(str(int(self.curr_cmp)))
        
        # 4. Carrega NOVO estado
        self.load_cmp(self.curr_cmp)

    def on_prev_cmp(self, event):
        self.change_cmp_index(self.curr_idx - 1)

    def on_next_cmp(self, event):
        self.change_cmp_index(self.curr_idx + 1)

    def on_text_goto(self, text):
        try:
            target = float(text)
            idx = np.abs(self.nav_cmps - target).argmin()
            self.change_cmp_index(idx)
        except: pass

    def on_config_update(self, event):
        try:
            self.p_vmin = float(self.txt_vmin.text)
            self.p_vmax = float(self.txt_vmax.text)
            self.p_dv   = float(self.txt_dv.text)
            self.p_win  = float(self.txt_win.text)
            self.p_gain = float(self.txt_gain.text)
            self.update_v_axis()
            self.load_cmp(self.curr_cmp)
        except: print("Erro parâmetros")

    def on_pick(self, event):
        if event.inaxes != self.ax1_sem or self.fig1.canvas.toolbar.mode != '': return
        
        if event.button == 1: self.picks_curr.append((event.ydata, event.xdata))
        elif event.button == 3 and self.picks_curr:
             dists = [((t-event.ydata)**2 + ((v-event.xdata)/1000)**2) for t, v in self.picks_curr]
             self.picks_curr.pop(np.argmin(dists))
        
        # Salva instantâneo
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        self.update_dynamic()

    def reset_all(self, event):
        self.picks_db = {}
        self.picks_curr = []
        self.update_dynamic()
        self.ln_map_done.set_data([], [])
        self.fig1.canvas.draw()

    def save_picks_disk(self, event):
        # Commit final
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        rows = []
        for c, p in self.picks_db.items():
            for t, v in p: rows.append({'CMP': c, 'Time': t, 'Velocity': v})
        try:
            pd.DataFrame(rows).to_csv(F_OUT_PICKS, index=False)
            print(f"Picks Salvos.")
        except Exception as e: print(f"Erro CSV: {e}")

    def cleanup_qc(self):
        if self.fig2 and plt.fignum_exists(self.fig2.number): plt.close(self.fig2)
        if self.fig3 and plt.fignum_exists(self.fig3.number): plt.close(self.fig3)
        self.fig2, self.fig3 = None, None
        self.ln_prof = None 
        if self.fp_nmo_qc is not None:
            del self.fp_nmo_qc
            self.fp_nmo_qc = None
        gc.collect()

    def export_data(self, event):
        print("\n--- EXPORTANDO ---")
        self.cleanup_qc()
        self.save_picks_disk(None)
        if not os.path.exists(F_OUT_PICKS): return
        df_p = pd.read_csv(F_OUT_PICKS)
        if df_p.empty: return

        print("1. Interpolando Modelo...")
        picked_cmps = np.sort(df_p['CMP'].unique())
        dense_profiles = {}
        for c in picked_cmps:
            p = df_p[df_p['CMP'] == c].sort_values('Time')
            v_curve = np.interp(self.t_ax, p['Time'], p['Velocity'], left=p['Velocity'].iloc[0], right=p['Velocity'].iloc[-1])
            dense_profiles[c] = v_curve

        self.vel_model_2d = np.zeros((Nt, len(self.all_cmps)), dtype=np.float32)
        
        for i, cmp_curr in enumerate(self.all_cmps):
            if cmp_curr in dense_profiles:
                self.vel_model_2d[:, i] = dense_profiles[cmp_curr]
            else:
                idx_pos = np.searchsorted(picked_cmps, cmp_curr)
                if idx_pos == 0: self.vel_model_2d[:, i] = dense_profiles[picked_cmps[0]]
                elif idx_pos == len(picked_cmps): self.vel_model_2d[:, i] = dense_profiles[picked_cmps[-1]]
                else:
                    c1, c2 = picked_cmps[idx_pos - 1], picked_cmps[idx_pos]
                    w = (cmp_curr - c1) / (c2 - c1)
                    v_left = dense_profiles[c1]
                    v_right = dense_profiles[c2]
                    self.vel_model_2d[:, i] = (1.0 - w) * v_left + w * v_right
        
        self.vel_model_2d.astype(np.float32).tofile(F_OUT_VEL)

        print("2. Gerando NMO...")
        try:
            with open(F_OUT_NMO, 'wb') as f: f.seek(len(self.headers)*Nt*4 - 1); f.write(b'\0')
            fp_nmo = np.memmap(F_OUT_NMO, dtype='float32', mode='r+', shape=(Nt, len(self.headers)), order='F')
            fp_raw = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, len(self.headers)), order='F')
            
            for i, c_val in enumerate(tqdm(self.all_cmps)):
                raw, off, idxs = load_cmp_from_disk(self.headers, fp_raw, c_val, CMP_BIN_SIZE)
                if raw is None: continue
                muted = process_export_mute(raw, off, dt) 
                nmo = run_nmo_fast(muted, off, dt, self.vel_model_2d[:, i], STRETCH_LIM)
                fp_nmo[:, idxs] = nmo
                
            del fp_nmo, fp_raw
            print("Concluído.")
            self.open_qc_windows(self.vel_model_2d, df_p)
        except Exception as e:
            print(f"Erro Export: {e}")

    def open_qc_windows(self, vel_model, df_p):
        # JANELA 2
        self.fig2 = plt.figure(figsize=(14, 6))
        self.fig2.canvas.manager.set_window_title("2. QC Modelo")
        ax_map = self.fig2.add_subplot(121)
        self.ax_prf = self.fig2.add_subplot(122)
        
        im = ax_map.imshow(vel_model, aspect='auto', cmap='jet', vmin=self.p_vmin, vmax=self.p_vmax,
                           extent=[self.all_cmps[0], self.all_cmps[-1], self.t_ax[-1], 0])
        ax_map.scatter(df_p['CMP'], df_p['Time'], c='k', s=10)
        ax_map.set_title("Modelo Interpolado"); ax_map.set_xlabel("CMP"); ax_map.set_ylabel("Tempo")
        plt.colorbar(im, ax=ax_map, label="Vrms")
        
        idx_curr = np.abs(self.all_cmps - self.curr_cmp).argmin()
        self.ln_prof, = self.ax_prf.plot(vel_model[:, idx_curr], self.t_ax, 'k-', lw=2)
        self.ln_mark_curr = ax_map.axvline(self.all_cmps[idx_curr], color='w', ls='--')
        
        self.ax_prf.set_ylim(self.t_ax[-1], 0); self.ax_prf.grid(True)
        self.ax_prf.set_xlim(self.p_vmin, self.p_vmax)
        self.ax_prf.set_title(f"Perfil CMP {self.all_cmps[idx_curr]}")
        
        self.vel_cached = vel_model
        self.fig2.canvas.mpl_connect('button_press_event', self.on_map_click)

        # JANELA 3
        self.fig3 = plt.figure(figsize=(14, 6))
        self.fig3.canvas.manager.set_window_title("3. QC NMO")
        self.ax3_main = self.fig3.add_axes([0.05, 0.25, 0.9, 0.7])
        
        ax_sl = self.fig3.add_axes([0.2, 0.05, 0.5, 0.03])
        self.sl_qc = Slider(ax_sl, "Posição", self.min_cmp, self.max_cmp, valinit=self.curr_cmp, valstep=CMP_BIN_SIZE)
        self.sl_qc.on_changed(self.update_qc_view)
        
        self.txt_qc = TextBox(self.fig3.add_axes([0.8, 0.05, 0.1, 0.05]), "Qtd:", initial=str(self.qc_n_panels))
        self.txt_qc.on_submit(self.on_qc_text)
        
        self.im_multi = self.ax3_main.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.ax3_main.set_title("Supergather NMO")
        self.ax3_main.set_ylabel("Tempo (s)"); self.ax3_main.set_xlabel("Offset (m)")
        
        try:
            self.fp_nmo_qc = np.memmap(F_OUT_NMO, dtype='float32', mode='r', shape=(Nt, len(self.headers)), order='F')
            self.update_qc_view(None)
            plt.show()
        except Exception as e:
            print(f"Erro ao abrir QC: {e}")

    def on_map_click(self, event):
        if event.inaxes != self.fig2.axes[0]: return
        idx = np.searchsorted(self.all_cmps, event.xdata)
        if 0 <= idx < len(self.all_cmps):
            self.ln_prof.set_data(self.vel_cached[:, idx], self.t_ax)
            self.ln_mark_curr.set_xdata([event.xdata])
            self.ax_prf.set_title(f"Perfil CMP {int(event.xdata)}")
            self.fig2.canvas.draw_idle()

    def on_qc_text(self, text):
        try: self.qc_n_panels = int(text)
        except: pass
        self.update_qc_view(None)

    def update_qc_view(self, _):
        if self.fp_nmo_qc is None: return
        target = self.sl_qc.val
        idx = np.searchsorted(self.all_cmps, target)
        n = self.qc_n_panels
        
        start, end = max(0, idx - n//2), min(len(self.all_cmps), idx + n)
        panels, sep = [], np.zeros((Nt, 5))
        for i in range(start, end):
            raw, _, _ = load_cmp_from_disk(self.headers, self.fp_nmo_qc, self.all_cmps[i], CMP_BIN_SIZE)
            if raw is None: continue
            vm = np.percentile(np.abs(raw), 98); 
            if vm > 0: raw /= vm
            panels.append(raw); panels.append(sep)
        
        if panels:
            full = np.hstack(panels[:-1])
            self.im_multi.set_data(full)
            self.im_multi.set_extent([0, full.shape[1], self.t_ax[-1], 0])
            self.im_multi.set_clim(-1, 1)
            self.ax3_main.set_aspect('auto')
            self.ax3_main.set_ylim(self.t_ax[-1], 0)
            self.fig3.canvas.draw_idle()

if __name__ == "__main__":
    app = VelocitySuite()
    plt.show()