import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
import os
import gc
from tqdm import tqdm
from numba import jit, prange

"""
SCRIPT 04: ETA ANALYSIS (RESIDUAL ANISOTROPY)
====================================================
Objetivo: 
1. Analisar o dado NMO Isotrópico para estimar o parâmetro ETA.
2. Aplicar correção Residual de Anisotropia (ETA) no dado NMO Isotrópico.
------------------------------------
Técnica:
- Usa semblance para análise de ETA.
- Usa correção Residual baseada em modelo de velocidade RMS.
------------------------------------
Parâmetros:
- Nt: Número de amostras por traço.
- dt: Taxa de amostragem (s).
- CMP_BIN_SIZE: Tamanho do bin CMP (unidade de distância).
------------------------------------
Outputs:
- Arquivo CSV com picks de ETA.
- Modelo 2D de ETA em arquivo binário.
- Dado NMO Anisotrópico corrigido em arquivo binário.
------------------------------------
Referências:
- Carvajal, X. et al. (2019). Residual Moveout Correction in VTI Media Using an Efficient Approximation of the Nonhyperbolic Moveout Equation. Geophysics.
- Alkhalifa, I. (2000). Anisotropic Normal-Moveout Correction Using a Generalized Dix Equation. Geophysics.
- Alkhalifa, I., Tsvankin, I. (1995). Velocity Analysis for Transversely Isotropic Media. Geophysics.
- Alkhalifa, I., Tsvankin, I. (1996). Nonhyperbolic Moveout of Seismic Reflections in VTI Media. Geophysics.
------------------------------------
Autor: Ana Paula Carvalho
Data: Dezembro de 2024 
Versão: 1.0

"""

# ================= CONFIGURAÇÃO =================
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'

# Entrada: Dado já corrigido de NMO (Isotrópico)
F_BIN_IN    = f'{BASE_DIR}/outputs/Line_NMO_Corrected.bin' 
F_HEAD      = f'{BASE_DIR}/outputs/Trace_Headers.csv'
F_VEL_IN    = f'{BASE_DIR}/outputs/Velocity_Model_2D.bin' 

# Saídas
F_OUT_PICKS = f'{BASE_DIR}/outputs/Eta_Picks.csv'
F_OUT_ETA   = f'{BASE_DIR}/outputs/Eta_Model_2D.bin'
F_OUT_ANISO = f'{BASE_DIR}/outputs/Line_Aniso_Corrected.bin'

Nt = 2501           
dt = 0.001          

CMP_BIN_SIZE = 10.0 
CMP_STEP = 50.0     

# Parâmetros
INIT_EMIN = -0.3
INIT_EMAX = 0.6
INIT_DE   = 0.01     
INIT_WIN  = 60.0     
INIT_GAIN = 5.0 
VEL_MUTE  = 1500.0     
BUF_MUTE  = 100      
STRETCH_LIM = 35.0    

# ================= KERNELS RESIDUAIS =================

@jit(nopython=True, fastmath=True)
def calc_residual_shift(t0, x, v, eta):
    if t0 < 1e-5: return t0
    
    v2 = v**2
    x2 = x**2
    t02 = t0**2
    
    t_iso_sq = t02 + x2/v2
    t_iso = np.sqrt(t_iso_sq)
    
    den_term = (1.0 + 2.0 * eta) * x2
    full_den = v2 * (t02 * v2 + den_term)
    
    if full_den < 1e-5: return -1.0
    
    num = 2.0 * eta * (x2**2)
    t_aniso_sq = t_iso_sq - (num / full_den)
    
    if t_aniso_sq < 0: return -1.0
    t_aniso = np.sqrt(t_aniso_sq)
    
    shift = t_aniso - t_iso
    return t0 + shift

@jit(nopython=True, parallel=True, fastmath=True)
def numba_semblance_residual(data, offsets, times, etas, v_profile, dt, win_len):
    nt, nrec = data.shape
    n_eta = len(etas)
    semb = np.zeros((nt, n_eta), dtype=np.float32)
    half_w = win_len // 2
    
    for ie in prange(n_eta):
        eta = etas[ie]
        num_trace = np.zeros(nt, dtype=np.float32)
        den_trace = np.zeros(nt, dtype=np.float32)
        
        for it in range(nt):
            t0 = times[it]
            v = v_profile[it]
            if v < 100: continue

            sum_amp = 0.0
            sum_sq = 0.0
            count_live = 0
            
            for ir in range(nrec):
                t_fetch = calc_residual_shift(t0, offsets[ir], v, eta)
                
                if t_fetch < 0 or t_fetch >= times[-1]: continue
                
                idx_float = t_fetch / dt
                idx0 = int(idx_float)
                idx1 = idx0 + 1
                
                if idx0 >= 0 and idx1 < nt:
                    w1 = idx_float - idx0
                    w0 = 1.0 - w1
                    val = w0 * data[idx0, ir] + w1 * data[idx1, ir]
                    
                    if abs(val) > 1e-9:
                        sum_amp += val
                        sum_sq += val*val
                        count_live += 1
            
            if count_live > 3:
                num_trace[it] = sum_amp * sum_amp
                den_trace[it] = sum_sq
        
        for it in range(nt):
            start = max(0, it - half_w)
            end = min(nt, it + half_w + 1)
            s_num = 0.0; s_den = 0.0
            for k in range(start, end):
                s_num += num_trace[k]; s_den += den_trace[k]
            
            if s_den > 1e-5:
                val = s_num / (nrec * s_den + 1e-9)
                semb[it, ie] = min(1.0, max(0.0, val))
                
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_apply_residual(data, out, offsets, times, v_rms, eta_profile, dt):
    nt, nrec = data.shape
    for i in prange(nrec):
        x = offsets[i]
        for j in range(nt):
            t0 = times[j]
            v = v_rms[j]; eta = eta_profile[j]
            
            t_fetch = calc_residual_shift(t0, x, v, eta)
            
            if t_fetch < 0:
                out[j, i] = 0.0; continue
                
            idx_float = t_fetch / dt
            idx0 = int(idx_float); idx1 = idx0 + 1
            if idx0 >= 0 and idx1 < nt:
                w1 = idx_float - idx0; w0 = 1.0 - w1
                out[j, i] = w0 * data[idx0, i] + w1 * data[idx1, i]
            else: out[j, i] = 0.0

# ================= APP =================

class EtaSuite:
    def __init__(self):
        print("--- INICIANDO SUITE ETA (FINAL v30) ---")
        if not os.path.exists(F_BIN_IN) or not os.path.exists(F_VEL_IN):
            print("ERRO: Input Data (NMO) ou Modelo de Velocidade faltando.")
            return
        
        self.headers = pd.read_csv(F_HEAD)
        self.headers.columns = self.headers.columns.str.strip()
        
        self.fp_data = np.memmap(F_BIN_IN, dtype='float32', mode='r', 
                                 shape=(Nt, len(self.headers)), order='F')
        
        col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
        self.all_cmps = np.sort(self.headers[col].unique())
        self.min_cmp, self.max_cmp = self.all_cmps[0], self.all_cmps[-1]
        
        try:
            self.fp_vel = np.memmap(F_VEL_IN, dtype='float32', mode='r', 
                                    shape=(Nt, len(self.all_cmps)), order='C')
        except Exception as e:
            print(f"Erro vel: {e}"); return

        step = max(1, int(CMP_STEP/(self.all_cmps[1]-self.all_cmps[0])))
        self.nav_cmps = self.all_cmps[::step]
        self.curr_idx = len(self.nav_cmps)//2
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        
        self.picks_db = {}
        self.picks_curr = []
        
        self.p_emin, self.p_emax = INIT_EMIN, INIT_EMAX
        self.p_de, self.p_win, self.p_gain = INIT_DE, INIT_WIN, INIT_GAIN
        
        self.eta_model_2d = None
        self.qc_n_panels = 5
        self.fp_aniso_qc = None
        
        self.fig2, self.fig3, self.fig4 = None, None, None
        self.ln_prof, self.ln_mark_curr = None, None
        
        self.show_guides = False
        self.guide_lines = []
        
        self.t_ax = np.arange(Nt, dtype=np.float32)*dt
        self.update_eta_axis()
        self.data_center = None
        self.curr_v_curve = None 
        
        self.init_windows()
        self.load_cmp(self.curr_cmp)

    def update_eta_axis(self):
        self.eta_ax = np.arange(self.p_emin, self.p_emax + self.p_de, self.p_de, dtype=np.float32)

    def init_windows(self):
        self.fig1 = plt.figure(figsize=(14, 9))
        self.fig1.canvas.manager.set_window_title("1. Análise Residual (ETA)")
        base_plot = 0.40; height_plot = 0.55
        self.ax1_sem = self.fig1.add_axes([0.05, base_plot, 0.28, height_plot])
        self.ax1_gat = self.fig1.add_axes([0.35, base_plot, 0.28, height_plot], sharey=self.ax1_sem)
        self.ax1_nmo = self.fig1.add_axes([0.65, base_plot, 0.28, height_plot], sharey=self.ax1_sem)
        
        self.fig1.patches.append(plt.Rectangle((0, 0), 1, base_plot - 0.02, 
                                               transform=self.fig1.transFigure, color='#f0f0f0', zorder=-1))
        
        self.ax_map = self.fig1.add_axes([0.1, 0.30, 0.8, 0.02])
        self.ax_map.set_yticks([]); self.ax_map.set_xlim(self.min_cmp, self.max_cmp)
        self.ax_map.set_xlabel("Picking Status", fontsize=8)
        self.ax_map.plot(self.all_cmps, np.zeros_like(self.all_cmps), 'o', color='lightgray', markersize=2)
        self.ln_map_done, = self.ax_map.plot([], [], 'o', color='green', markersize=4)
        self.ln_map_curr, = self.ax_map.plot([self.curr_cmp], [0], 'o', color='red', markersize=8, markeredgecolor='k')
        
        y_nav = 0.22
        self.btn_prev = Button(self.fig1.add_axes([0.40, y_nav, 0.04, 0.04]), "<<")
        self.btn_prev.on_clicked(self.on_prev_cmp)
        self.txt_goto = TextBox(self.fig1.add_axes([0.45, y_nav, 0.08, 0.04]), "CMP:", initial=str(int(self.curr_cmp)))
        self.txt_goto.on_submit(self.on_text_goto)
        self.btn_next = Button(self.fig1.add_axes([0.54, y_nav, 0.04, 0.04]), ">>")
        self.btn_next.on_clicked(self.on_next_cmp)
        
        y_p, h_ctl = 0.13, 0.035
        self.txt_emin = TextBox(self.fig1.add_axes([0.08, y_p, 0.06, h_ctl]), "EtaMin:", initial=str(self.p_emin))
        self.txt_emax = TextBox(self.fig1.add_axes([0.19, y_p, 0.06, h_ctl]), "EtaMax:", initial=str(self.p_emax))
        self.txt_de   = TextBox(self.fig1.add_axes([0.31, y_p, 0.05, h_ctl]), "Step:", initial=str(self.p_de))
        self.txt_win  = TextBox(self.fig1.add_axes([0.43, y_p, 0.05, h_ctl]), "Win:", initial=str(self.p_win))
        self.txt_gain = TextBox(self.fig1.add_axes([0.55, y_p, 0.05, h_ctl]), "Gain:", initial=str(self.p_gain))
        self.btn_upd = Button(self.fig1.add_axes([0.65, y_p, 0.08, h_ctl]), "Update", color='lightblue')
        self.btn_upd.on_clicked(self.on_config_update)
        
        y_a, w_btn = 0.04, 0.10
        self.btn_save = Button(self.fig1.add_axes([0.15, y_a, w_btn, h_ctl]), "Salvar", color='lightgreen')
        self.btn_save.on_clicked(self.save_picks_disk)
        self.btn_exp = Button(self.fig1.add_axes([0.35, y_a, w_btn, h_ctl]), "EXPORTAR", color='gold')
        self.btn_exp.on_clicked(self.export_data)
        self.btn_rst = Button(self.fig1.add_axes([0.55, y_a, w_btn, h_ctl]), "RESETAR", color='salmon')
        self.btn_rst.on_clicked(self.reset_all)
        self.btn_cls = Button(self.fig1.add_axes([0.75, y_a, w_btn, h_ctl]), "FECHAR", color='#ffcccc')
        self.btn_cls.on_clicked(lambda e: plt.close('all'))
        
        self.im_sem = self.ax1_sem.imshow(np.zeros((10,10)), aspect='auto', cmap='jet', vmin=0, vmax=1)
        self.ln_pick, = self.ax1_sem.plot([], [], 'w-o', lw=2.5, markersize=6, zorder=100)
        self.im_gat = self.ax1_gat.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.im_nmo = self.ax1_nmo.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        
        self.ax1_sem.set_title("Eta Semblance (Residual)"); self.ax1_sem.set_xlabel("Eta"); self.ax1_sem.set_ylabel("Tempo (s)")
        self.ax1_gat.set_title("Input (Iso NMO)"); self.ax1_gat.set_xlabel("Offset (m)")
        self.ax1_nmo.set_title("Corrected (Preview)"); self.ax1_nmo.set_xlabel("Offset (m)")
        for ax in [self.ax1_sem, self.ax1_gat, self.ax1_nmo]: ax.set_ylim(self.t_ax[-1], 0)
        self.fig1.canvas.mpl_connect('button_press_event', self.on_pick)

    def load_cmp_data(self, target):
        x_min, x_max = target - CMP_BIN_SIZE/2, target + CMP_BIN_SIZE/2
        col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
        df = self.headers[(self.headers[col] >= x_min) & (self.headers[col] <= x_max)]
        if df.empty: return None, None, None, None
        
        idx_col = 'global_trace_index' if 'global_trace_index' in df.columns else 'g_idx'
        indices = df[idx_col].values.astype(np.int64)
        offsets = df['offset'].values.astype(np.float32)
        sort = np.argsort(offsets)
        data = np.array(self.fp_data[:, indices[sort]])
        
        idx_cmp = np.searchsorted(self.all_cmps, target)
        idx_cmp = max(0, min(len(self.all_cmps)-1, idx_cmp))
        v_curve = np.array(self.fp_vel[:, idx_cmp])
        return data, offsets[sort], indices[sort], v_curve

    def load_cmp(self, cmp_val):
        print(f"Residual Analysis CMP {int(cmp_val)}...")
        raw, off, _, v_curve = self.load_cmp_data(cmp_val)
        if raw is None: return

        self.offsets = off
        self.curr_v_curve = v_curve
        
        nt = raw.shape[0]
        t = np.arange(nt, dtype=np.float32) * dt
        
        if self.p_gain == 0.0:
            gained = raw 
        else:
            gained = raw * (t[:, None]**self.p_gain)
            vm = np.percentile(np.abs(gained), 99)
            if vm > 0: gained /= vm
            
        self.data_center = gained
        
        win_samp = max(1, int((self.p_win/1000)/dt))
        self.semb = numba_semblance_residual(self.data_center, off, self.t_ax, self.eta_ax, 
                                             v_curve.astype(np.float32), dt, win_samp)
        
        vm_s = np.percentile(self.semb, 99.5)
        self.im_sem.set_data(self.semb)
        self.im_sem.set_extent([self.p_emin, self.p_emax, self.t_ax[-1], 0])
        self.im_sem.set_clim(0, vm_s)
        self.ax1_sem.set_xlim(self.p_emin, self.p_emax)
        
        self.im_gat.set_data(self.data_center)
        self.im_gat.set_extent([off[0], off[-1], self.t_ax[-1], 0])
        self.im_gat.set_clim(-1, 1)
        self.ax1_gat.set_xlim(off[0], off[-1])
        
        self.im_nmo.set_extent([off[0], off[-1], self.t_ax[-1], 0])
        self.im_nmo.set_clim(-1, 1)
        self.ax1_nmo.set_xlim(off[0], off[-1])
        
        self.picks_curr = list(self.picks_db.get(int(cmp_val), []))
        self.update_dynamic()
        self.ln_map_curr.set_xdata([cmp_val])
        
        keys = np.array(list(self.picks_db.keys()))
        valid_keys = [k for k in keys if len(self.picks_db[k]) > 0]
        if valid_keys: self.ln_map_done.set_data(valid_keys, np.zeros(len(valid_keys)))
        self.fig1.canvas.draw_idle()

    def update_dynamic(self):
        if self.data_center is None or self.curr_v_curve is None: return
        self.picks_curr.sort(key=lambda x: x[0])
        if self.picks_curr:
            ts, es = zip(*self.picks_curr)
            self.ln_pick.set_data(es, ts)
            eta_curve = np.interp(self.t_ax, ts, es, left=es[0], right=es[-1])
        else:
            self.ln_pick.set_data([], [])
            eta_curve = np.zeros(Nt, dtype=np.float32)
        
        out = np.zeros_like(self.data_center)
        numba_apply_residual(self.data_center, out, self.offsets, self.t_ax, 
                             self.curr_v_curve, eta_curve, dt)
        self.im_nmo.set_data(out)
        
        if hasattr(self, 'ln_prof') and self.ln_prof is not None:
            if self.fig2 and plt.fignum_exists(self.fig2.number):
                self.ln_prof.set_data(eta_curve, self.t_ax)
                if hasattr(self, 'ln_mark_curr'): self.ln_mark_curr.set_xdata([self.curr_cmp])
                self.fig2.canvas.draw_idle()
        self.fig1.canvas.draw_idle()

    def change_cmp_index(self, idx):
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        idx = max(0, min(len(self.nav_cmps)-1, idx))
        self.curr_idx = idx
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        self.txt_goto.set_val(str(int(self.curr_cmp)))
        self.load_cmp(self.curr_cmp)

    def on_prev_cmp(self, event): self.change_cmp_index(self.curr_idx - 1)
    def on_next_cmp(self, event): self.change_cmp_index(self.curr_idx + 1)
    
    def on_text_goto(self, text):
        try:
            target = float(text)
            idx = np.abs(self.nav_cmps - target).argmin()
            self.change_cmp_index(idx)
        except: pass

    def on_config_update(self, event):
        try:
            self.p_emin = float(self.txt_emin.text)
            self.p_emax = float(self.txt_emax.text)
            self.p_de   = float(self.txt_de.text)
            self.p_win  = float(self.txt_win.text)
            self.p_gain = float(self.txt_gain.text)
            self.update_eta_axis()
            if event is not None: # Se chamado por botao, recarrega
                self.load_cmp(self.curr_cmp)
        except: print("Erro parâmetros")

    def on_pick(self, event):
        if event.inaxes != self.ax1_sem or self.fig1.canvas.toolbar.mode != '': return
        if event.button == 1: self.picks_curr.append((event.ydata, event.xdata))
        elif event.button == 3 and self.picks_curr:
             dists = [((t-event.ydata)**2 + ((e-event.xdata)*5)**2) for t, e in self.picks_curr]
             self.picks_curr.pop(np.argmin(dists))
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        self.update_dynamic()

    def reset_all(self, event):
        self.picks_db = {}
        self.picks_curr = []
        self.update_dynamic()
        self.ln_map_done.set_data([], [])
        self.fig1.canvas.draw()

    def save_picks_disk(self, event):
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        rows = []
        for c, p in self.picks_db.items():
            for t, e in p: rows.append({'CMP': c, 'Time': t, 'Eta': e})
        try:
            pd.DataFrame(rows).to_csv(F_OUT_PICKS, index=False)
            print(f"Picks de Eta Salvos.")
        except Exception as e: print(f"Erro CSV: {e}")

    def cleanup_qc(self):
        if self.fig2 and plt.fignum_exists(self.fig2.number): plt.close(self.fig2)
        if self.fig3 and plt.fignum_exists(self.fig3.number): plt.close(self.fig3)
        if self.fig4 and plt.fignum_exists(self.fig4.number): plt.close(self.fig4)
        self.fig2, self.fig3, self.fig4 = None, None, None
        self.ln_prof = None
        if self.fp_aniso_qc is not None: del self.fp_aniso_qc; self.fp_aniso_qc = None
        gc.collect()

    def export_data(self, event):
        print("\n--- EXPORTANDO DADOS ANISOTRÓPICOS (RESIDUAL) ---")
        
        # 1. FORÇA ATUALIZAÇÃO DOS PARÂMETROS DA UI
        self.on_config_update(None)
        
        self.cleanup_qc()
        self.save_picks_disk(None)
        if not os.path.exists(F_OUT_PICKS): return
        df_p = pd.read_csv(F_OUT_PICKS)
        if df_p.empty: return

        print("1. Interpolando Modelo de Eta...")
        picked_cmps = np.sort(df_p['CMP'].unique())
        dense_profiles = {}
        for c in picked_cmps:
            p = df_p[df_p['CMP'] == c].sort_values('Time')
            e_curve = np.interp(self.t_ax, p['Time'], p['Eta'], left=p['Eta'].iloc[0], right=p['Eta'].iloc[-1])
            dense_profiles[c] = e_curve

        self.eta_model_2d = np.zeros((Nt, len(self.all_cmps)), dtype=np.float32)
        
        for i, cmp_curr in enumerate(self.all_cmps):
            if cmp_curr in dense_profiles:
                self.eta_model_2d[:, i] = dense_profiles[cmp_curr]
            else:
                idx_pos = np.searchsorted(picked_cmps, cmp_curr)
                if idx_pos == 0: self.eta_model_2d[:, i] = dense_profiles[picked_cmps[0]]
                elif idx_pos == len(picked_cmps): self.eta_model_2d[:, i] = dense_profiles[picked_cmps[-1]]
                else:
                    c1, c2 = picked_cmps[idx_pos - 1], picked_cmps[idx_pos]
                    w = (cmp_curr - c1) / (c2 - c1)
                    e_left = dense_profiles[c1]
                    e_right = dense_profiles[c2]
                    self.eta_model_2d[:, i] = (1.0 - w) * e_left + w * e_right
        
        self.eta_model_2d.astype(np.float32).tofile(F_OUT_ETA)

        print("2. Aplicando Correção Residual (RAW)...")
        try:
            with open(F_OUT_ANISO, 'wb') as f: f.seek(len(self.headers)*Nt*4 - 1); f.write(b'\0')
            fp_out = np.memmap(F_OUT_ANISO, dtype='float32', mode='r+', shape=(Nt, len(self.headers)), order='F')
            
            for i, c_val in enumerate(tqdm(self.all_cmps)):
                # Carrega NMO Data ORIGINAL (Sem ganho hardcoded)
                raw, off, idxs, v_curve = self.load_cmp_data(c_val)
                if raw is None: continue
                
                eta_curve = self.eta_model_2d[:, i]
                
                corrected = np.zeros_like(raw)
                # Passa RAW diretamente para salvar RAW
                numba_apply_residual(raw, corrected, off, self.t_ax, v_curve, eta_curve, dt)
                
                fp_out[:, idxs] = corrected
            del fp_out
            print("Concluído.")
            self.open_qc_windows(self.eta_model_2d, df_p)
        except Exception as e: print(f"Erro Export: {e}")

    def open_qc_windows(self, eta_model, df_p):
        # JANELA 2: Modelo ETA
        self.fig2 = plt.figure(figsize=(12, 5))
        self.fig2.canvas.manager.set_window_title("2. QC Modelo Eta")
        ax_map = self.fig2.add_subplot(121)
        self.ax_prf = self.fig2.add_subplot(122)
        im = ax_map.imshow(eta_model, aspect='auto', cmap='seismic', vmin=-0.2, vmax=0.2,
                           extent=[self.all_cmps[0], self.all_cmps[-1], self.t_ax[-1], 0])
        ax_map.scatter(df_p['CMP'], df_p['Time'], c='k', s=10)
        ax_map.set_title("Modelo Eta"); plt.colorbar(im, ax=ax_map)
        idx_curr = np.abs(self.all_cmps - self.curr_cmp).argmin()
        self.ln_prof, = self.ax_prf.plot(eta_model[:, idx_curr], self.t_ax, 'k-', lw=2)
        self.ln_mark_curr = ax_map.axvline(self.all_cmps[idx_curr], color='g', ls='--')
        self.ax_prf.set_ylim(self.t_ax[-1], 0); self.ax_prf.grid(True)
        self.ax_prf.set_xlim(self.p_emin, self.p_emax)
        self.eta_cached = eta_model
        self.fig2.canvas.mpl_connect('button_press_event', self.on_map_click)

        # JANELA 3: QC Gather
        self.fig3 = plt.figure(figsize=(10, 6))
        self.fig3.canvas.manager.set_window_title("3. QC Supergather ANISO")
        self.ax3_main = self.fig3.add_axes([0.05, 0.25, 0.9, 0.7])
        ax_sl = self.fig3.add_axes([0.2, 0.05, 0.5, 0.03])
        self.sl_qc = Slider(ax_sl, "Posição", self.min_cmp, self.max_cmp, valinit=self.curr_cmp, valstep=CMP_BIN_SIZE)
        self.sl_qc.on_changed(self.update_qc_view)
        
        self.txt_qc = TextBox(self.fig3.add_axes([0.8, 0.05, 0.1, 0.05]), "Qtd:", initial=str(self.qc_n_panels))
        self.txt_qc.on_submit(self.on_qc_text)
        
        self.im_multi = self.ax3_main.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.ax3_main.set_title("Supergather Aniso NMO")
        
        # JANELA 4: COMPARAÇÃO
        self.fig4 = plt.figure(figsize=(14, 6))
        self.fig4.canvas.manager.set_window_title("4. QC Comparativo (Iso vs Aniso)")
        self.ax4_iso = self.fig4.add_subplot(121)
        self.ax4_ani = self.fig4.add_subplot(122, sharey=self.ax4_iso)
        
        self.im_cmp_iso = self.ax4_iso.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.im_cmp_ani = self.ax4_ani.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        
        self.ax4_iso.set_title("Antes (Isotrópico)")
        self.ax4_ani.set_title("Depois (Anisotrópico)")
        
        ax_sl4 = self.fig4.add_axes([0.2, 0.05, 0.6, 0.03])
        self.sl_qc4 = Slider(ax_sl4, "CMP", self.min_cmp, self.max_cmp, valinit=self.curr_cmp, valstep=CMP_BIN_SIZE)
        self.sl_qc4.on_changed(self.update_cmp_view)
        
        # Checkbox para Linhas Guia
        ax_chk = self.fig4.add_axes([0.85, 0.05, 0.1, 0.05])
        self.chk_grid = CheckButtons(ax_chk, ["Guias"], [False])
        self.chk_grid.on_clicked(self.on_toggle_grid)
        
        try:
            self.fp_aniso_qc = np.memmap(F_OUT_ANISO, dtype='float32', mode='r', shape=(Nt, len(self.headers)), order='F')
            self.update_qc_view(None)
            self.update_cmp_view(None)
            plt.show()
        except Exception as e: print(f"Erro ao abrir QC: {e}")

    def on_map_click(self, event):
        if event.inaxes != self.fig2.axes[0]: return
        idx = np.searchsorted(self.all_cmps, event.xdata)
        if 0 <= idx < len(self.all_cmps):
            self.ln_prof.set_data(self.eta_cached[:, idx], self.t_ax)
            self.ln_mark_curr.set_xdata([event.xdata])
            self.fig2.canvas.draw_idle()

    def on_qc_text(self, text):
        try: self.qc_n_panels = int(text)
        except: pass
        self.update_qc_view(None)

    def on_toggle_grid(self, label):
        self.show_guides = not self.show_guides
        self.update_cmp_view(None)

    def update_qc_view(self, _):
        if self.fp_aniso_qc is None: return
        target = self.sl_qc.val
        idx = np.searchsorted(self.all_cmps, target)
        n = self.qc_n_panels
        start, end = max(0, idx - n//2), min(len(self.all_cmps), idx + n)
        panels, sep = [], np.zeros((Nt, 5))
        for i in range(start, end):
            x_min, x_max = self.all_cmps[i] - CMP_BIN_SIZE/2, self.all_cmps[i] + CMP_BIN_SIZE/2
            col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
            df = self.headers[(self.headers[col] >= x_min) & (self.headers[col] <= x_max)]
            if df.empty: continue
            idx_col = 'global_trace_index' if 'global_trace_index' in df.columns else 'g_idx'
            indices = df[idx_col].values.astype(np.int64)
            offsets = df['offset'].values.astype(np.float32)
            sort = np.argsort(offsets)
            
            trace_qc = np.array(self.fp_aniso_qc[:, indices[sort]])
            vm = np.percentile(np.abs(trace_qc), 98)
            if vm > 0: trace_qc /= vm
            panels.append(trace_qc); panels.append(sep)
            
        if panels:
            full = np.hstack(panels[:-1])
            self.im_multi.set_data(full)
            self.im_multi.set_extent([0, full.shape[1], self.t_ax[-1], 0])
            self.im_multi.set_clim(-1, 1)
            self.ax3_main.set_aspect('auto')
            self.ax3_main.set_ylim(self.t_ax[-1], 0)
            self.fig3.canvas.draw_idle()

    def update_cmp_view(self, _):
        if self.fp_aniso_qc is None: return
        target = self.sl_qc4.val
        
        # 1. Input (Iso NMO)
        raw_iso, _, _, _ = self.load_cmp_data(target)
        if raw_iso is not None:
             # Aplica Gain VISUAL (se configurado)
             nt = raw_iso.shape[0]
             t = np.arange(nt, dtype=np.float32) * dt
             if self.p_gain != 0: raw_iso *= (t[:, None]**self.p_gain)
             
             vm = np.percentile(np.abs(raw_iso), 98)
             if vm > 0: raw_iso /= vm
             self.im_cmp_iso.set_data(raw_iso)
             self.im_cmp_iso.set_extent([0, raw_iso.shape[1], self.t_ax[-1], 0])
             self.im_cmp_iso.set_clim(-1, 1)
             self.ax4_iso.set_aspect('auto')

        # 2. Output (Aniso NMO)
        x_min, x_max = target - CMP_BIN_SIZE/2, target + CMP_BIN_SIZE/2
        col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
        df = self.headers[(self.headers[col] >= x_min) & (self.headers[col] <= x_max)]
        if not df.empty:
            idx_col = 'global_trace_index' if 'global_trace_index' in df.columns else 'g_idx'
            indices = df[idx_col].values.astype(np.int64)
            offsets = df['offset'].values.astype(np.float32)
            sort = np.argsort(offsets)
            raw_ani = np.array(self.fp_aniso_qc[:, indices[sort]])
            
            # Aplica Gain VISUAL (se configurado)
            nt = raw_ani.shape[0]
            t = np.arange(nt, dtype=np.float32) * dt
            if self.p_gain != 0: raw_ani *= (t[:, None]**self.p_gain)
            
            vm = np.percentile(np.abs(raw_ani), 98)
            if vm > 0: raw_ani /= vm
            self.im_cmp_ani.set_data(raw_ani)
            self.im_cmp_ani.set_extent([0, raw_ani.shape[1], self.t_ax[-1], 0])
            self.im_cmp_ani.set_clim(-1, 1)
            self.ax4_ani.set_aspect('auto')
            
        # 3. Desenha Linhas Guia
        [l.remove() for l in self.guide_lines]
        self.guide_lines = []
        
        if self.show_guides:
            t_guides = np.arange(0, self.t_ax[-1], 0.2)
            for tg in t_guides:
                l1 = self.ax4_iso.axhline(tg, color='r', linestyle='--', alpha=0.5, lw=1)
                l2 = self.ax4_ani.axhline(tg, color='r', linestyle='--', alpha=0.5, lw=1)
                self.guide_lines.extend([l1, l2])
            
        self.fig4.canvas.draw_idle()

if __name__ == "__main__":
    app = EtaSuite()
    plt.show()