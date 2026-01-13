import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from numba import jit, prange
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# ==============================================================================
# 1. CONFIGURAÇÃO E INPUT DE FILTRO
# ==============================================================================
BASE_DIR = r'C:/Users/anapa/OneDrive/Área de Trabalho/SeismicModeling2D-master/SeismicModeling2D-master'

F_BIN_IN    = f'{BASE_DIR}/outputs/Line_NMO_Corrected_AP2.bin' 
F_HEAD      = f'{BASE_DIR}/outputs/Trace_Headers_AP2.csv'
F_VEL_IN    = f'{BASE_DIR}/outputs/Velocity_Model_RMS_AP2.bin' 

F_OUT_PICKS = f'{BASE_DIR}/outputs/Eta_Picks_AP2.csv'
F_OUT_ETA   = f'{BASE_DIR}/outputs/Eta_Model_2D_AP2.bin'
F_OUT_ANISO = f'{BASE_DIR}/outputs/Line_Aniso_Corrected_AP2.bin'

Nt, dt = 1501, 0.001
E_MIN, E_MAX, E_STEP = -0.05, 0.45, 0.01
GANHO_VISUAL = 5.0

print("\n--- INICIALIZAÇÃO DO PICKER ---")
OFFSET_MIN_SEMBLANCE = float(input("Digite o OFFSET MÍNIMO para o SEMBLANCE (ex: 900): "))

# ==============================================================================
# 2. KERNELS (FÍSICA DE ALKHALIFAH)
# ==============================================================================

@jit(nopython=True, fastmath=True)
def get_residual_time(t0, x, v, eta):
    if t0 < 0.001: return t0
    x2, v2, t02 = x**2, v**2, t0**2
    t_iso_sq = t02 + x2/v2
    den_ani = v2 * (t02 * v2 + (1.0 + 2.0 * eta) * x2)
    if den_ani < 1e-9: return t0
    t_ani_sq = t_iso_sq - (2.0 * eta * (x2**2)) / den_ani
    if t_ani_sq < 0: return t0
    return t0 + (np.sqrt(t_ani_sq) - np.sqrt(t_iso_sq))

@jit(nopython=True, parallel=True, fastmath=True)
def calc_semblance_far(data, offsets, times, etas, v_profile, dt, win_len, off_min):
    nt, n_eta = data.shape[0], len(etas); nrec = data.shape[1]
    semb = np.zeros((nt, n_eta), dtype=np.float32); half_w = win_len // 2
    for ie in prange(n_eta):
        eta = etas[ie]; num_tr, den_tr, count_tr = np.zeros(nt), np.zeros(nt), np.zeros(nt)
        for it in range(nt):
            t0, v = times[it], v_profile[it]
            if v < 500: continue
            s_amp, s_sq, n_live = 0.0, 0.0, 0
            for ir in range(nrec):
                if abs(offsets[ir]) < off_min: continue 
                t_f = get_residual_time(t0, offsets[ir], v, eta)
                if t_f < 0 or t_f >= (nt-1)*dt: continue
                idx = t_f / dt; i0 = int(idx); f = idx - i0
                val = (1.0 - f)*data[i0, ir] + f*data[i0+1, ir]
                s_amp += val; s_sq += val*val; n_live += 1
            if n_live > 1: num_tr[it], den_tr[it], count_tr[it] = s_amp**2, s_sq, n_live
        for it in range(half_w, nt - half_w):
            sn, sd, tn = np.sum(num_tr[it-half_w:it+half_w]), np.sum(den_tr[it-half_w:it+half_w]), np.sum(count_tr[it-half_w:it+half_w])
            if sd > 1e-10: 
                avg_n = tn / (2.0 * half_w)
                if avg_n >= 1.0: semb[it, ie] = sn / (avg_n * sd)
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_apply_correction(data, out, offsets, times, v_rms, eta_profile, dt):
    nt, nrec = data.shape
    for ir in prange(nrec):
        x = offsets[ir]
        for it in range(nt):
            t_f = get_residual_time(times[it], x, v_rms[it], eta_profile[it])
            if t_f < 0 or t_f >= (nt-1)*dt: out[it, ir] = 0.0
            else:
                idx = t_f / dt; i0 = int(idx); f = idx - i0
                out[it, ir] = (1.0 - f)*data[i0, ir] + f*data[i0+1, ir]

# ==============================================================================
# 3. INTERFACE COMPLETA (PICKING + EXPORTAÇÃO)
# ==============================================================================

class FinalEtaPicker:
    def __init__(self):
        self.h = pd.read_csv(F_HEAD); self.h.columns = self.h.columns.str.strip().str.lower()
        self.all_cmps = np.sort(self.h['cmp'].unique() if 'cmp' in self.h.columns else self.h['cmp_x'].unique())
        self.m = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, len(self.h)), order='F')
        self.v_mod = np.memmap(F_VEL_IN, dtype='float32', mode='r', shape=(Nt, len(self.all_cmps)), order='F')
        self.nav = self.all_cmps[::max(1, len(self.all_cmps)//50)]; self.curr_idx = len(self.nav)//2
        self.t_ax, self.eta_ax = np.arange(Nt)*dt, np.arange(E_MIN, E_MAX + E_STEP, E_STEP)
        self.picks = {}; self.setup_ui(); self.load_cmp()

    def build_profile(self, ts, es):
        p = np.zeros(Nt, dtype=np.float32)
        if not ts: return p
        idx = np.clip(np.round(np.array(ts)/dt).astype(int), 0, Nt-1)
        p[:idx[0]] = es[0]; 
        for i in range(len(idx)-1): p[idx[i]:idx[i+1]] = es[i]
        p[idx[-1]:] = es[-1]
        return gaussian_filter1d(p, sigma=(30.0/dt)/4.0, mode='nearest')

    def setup_ui(self):
        self.fig, (self.ax_s, self.ax_g, self.ax_n) = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
        self.fig.canvas.manager.set_window_title("Eta Picker - [S]: Salvar | [E]: Exportar | [Setas]: Navegar")
        self.im_s = self.ax_s.imshow(np.zeros((Nt, len(self.eta_ax))), aspect='auto', cmap='jet', extent=[E_MIN, E_MAX, Nt*dt, 0])
        self.ln_p, = self.ax_s.plot([], [], 'w-o', markersize=5, markeredgecolor='k')
        self.im_g = self.ax_g.imshow(np.zeros((Nt, 10)), aspect='auto', cmap='gray')
        self.im_n = self.ax_n.imshow(np.zeros((Nt, 10)), aspect='auto', cmap='gray')
        self.ax_s.set_title(f"Semblance (X > {OFFSET_MIN_SEMBLANCE}m)"); self.ax_g.set_title("Input (Todos)"); self.ax_n.set_title("Preview")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key); self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def load_cmp(self):
        c = self.nav[self.curr_idx]; idx_c = np.abs(self.all_cmps - c).argmin()
        df = self.h[(self.h['cmp' if 'cmp' in self.h.columns else 'cmp_x'] == c)]
        idxs = df['global_trace_index'].values; off = df['offset'].values; s = np.argsort(np.abs(off))
        self.curr_raw = np.array(self.m[:, idxs[s]]); self.curr_off = off[s].astype(np.float32); self.curr_v = np.array(self.v_mod[:, idx_c])
        d_vis = self.curr_raw * (self.t_ax[:, None]**GANHO_VISUAL); vm = np.percentile(np.abs(d_vis), 98) or 1.0; d_vis /= vm
        semb = calc_semblance_far(d_vis, self.curr_off, self.t_ax, self.eta_ax, self.curr_v, dt, 60, OFFSET_MIN_SEMBLANCE)
        self.im_s.set_data(semb); self.im_s.set_clim(0, np.percentile(semb, 99.5))
        ext = [off.min(), off.max(), Nt*dt, 0]; self.im_g.set_data(d_vis); self.im_g.set_extent(ext); self.im_g.set_clim(-1, 1)
        self.im_n.set_extent(ext); self.im_n.set_clim(-1, 1); self.curr_p = self.picks.get(int(c), []); self.update_preview()

    def update_preview(self):
        eta_c = np.zeros(Nt, dtype=np.float32)
        if self.curr_p: self.curr_p.sort(); ts, es = zip(*self.curr_p); self.ln_p.set_data(es, ts); eta_c = self.build_profile(ts, es)
        else: self.ln_p.set_data([], [])
        out = np.zeros_like(self.curr_raw); numba_apply_correction(self.curr_raw, out, self.curr_off, self.t_ax, self.curr_v, eta_c, dt)
        d_n = out * (self.t_ax[:, None]**GANHO_VISUAL); vm = np.percentile(np.abs(d_n), 98) or 1.0
        self.im_n.set_data(d_n/vm); self.fig.suptitle(f"CMP {int(self.nav[self.curr_idx])}"); self.fig.canvas.draw_idle()

    def on_key(self, ev):
        if ev.key == 'right': self.curr_idx = min(len(self.nav)-1, self.curr_idx+1); self.load_cmp()
        elif ev.key == 'left': self.curr_idx = max(0, self.curr_idx-1); self.load_cmp()
        elif ev.key == 's': rows = [{'CMP':c,'Time':t,'Eta':e} for c,p in self.picks.items() for t,e in p]; pd.DataFrame(rows).to_csv(F_OUT_PICKS, index=False); print("Picks salvos.")
        elif ev.key == 'e': self.export_all()

    def on_click(self, ev):
        if ev.inaxes != self.ax_s: return
        if ev.button == 1: self.curr_p.append((ev.ydata, ev.xdata))
        elif ev.button == 3 and self.curr_p: d = [((p[0]-ev.ydata)**2 + ((p[1]-ev.xdata)*5)**2) for p in self.curr_p]; self.curr_p.pop(np.argmin(d))
        self.picks[int(self.nav[self.curr_idx])] = self.curr_p; self.update_preview()

    def export_all(self):
        print("\n--- EXPORTANDO TUDO ---"); self.on_key(type('obj', (object,), {'key': 's'})())
        df = pd.read_csv(F_OUT_PICKS); p_cmps = np.sort(df['CMP'].unique()); eta_2d = np.zeros((Nt, len(self.all_cmps)), dtype=np.float32)
        profiles = {c: self.build_profile(df[df['CMP']==c].sort_values('Time')['Time'].values, df[df['CMP']==c].sort_values('Time')['Eta'].values) for c in p_cmps}
        for i, c in enumerate(self.all_cmps):
            idx = np.searchsorted(p_cmps, c)
            c_ref = p_cmps[0] if idx == 0 else p_cmps[-1] if idx == len(p_cmps) else p_cmps[idx]
            eta_2d[:, i] = profiles[c_ref]
        eta_2d.tofile(F_OUT_ETA); 
        with open(F_OUT_ANISO, 'wb') as f: f.seek(len(self.h)*Nt*4 - 1); f.write(b'\0')
        fp_o = np.memmap(F_OUT_ANISO, dtype='float32', mode='r+', shape=(Nt, len(self.h)), order='F')
        for i, c in enumerate(tqdm(self.all_cmps, desc="Corrigindo linha")):
            df_c = self.h[(self.h['cmp' if 'cmp' in self.h.columns else 'cmp_x'] == c)]
            idxs = df_c['global_trace_index'].values; off = df_c['offset'].values.astype(np.float32)
            raw = np.array(self.m[:, idxs]); corr = np.zeros_like(raw); numba_apply_correction(raw, corr, off, self.t_ax, self.v_mod[:, i], eta_2d[:, i], dt)
            fp_o[:, idxs] = corr
        print("CONCLUÍDO!"); plt.show()

if __name__ == "__main__":
    app = FinalEtaPicker(); plt.show()