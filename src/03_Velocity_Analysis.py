import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
import sys
from numba import jit, prange 
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d 
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = r'C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master'

FILE_BIN  = f'{BASE_DIR}/outputs/Line_CMP_Muted_AP2.bin'
FILE_HEAD = f'{BASE_DIR}/outputs/Trace_Headers_AP2.csv'

# Saídas
OUT_PICKS = f'{BASE_DIR}/outputs/Velocity_Picks_AP2.csv'
OUT_RMS   = f'{BASE_DIR}/outputs/Velocity_Model_RMS_AP2.bin'
OUT_INT   = f'{BASE_DIR}/outputs/Velocity_Model_INT_AP2.bin'
OUT_NMO   = f'{BASE_DIR}/outputs/Line_NMO_Corrected_AP2.bin'

IGNORE_OLD_PICKS = False  

Nt = 1501
dt = 0.001
CMP_BIN_SIZE = 10.0
MIN_FOLD = 50

# Visual
GANHO_VISUAL = 2.5
SUAVIZAR = True
V_MIN = 1400.0; V_MAX = 4500.0; V_STEP = 20.0
SEMBLANCE_WIN_MS = 40.0; STRETCH_LIM = 40.0

# Parâmetro de Suavização Vertical
VERTICAL_SMOOTH_MS = 40.0 

# ==============================================================================
# 2. KERNELS (MUTE REMOVIDO)
# ==============================================================================
@jit(nopython=True, parallel=True, fastmath=True)
def numba_semblace(data, offsets, times, vels, dt, win_len):
    nt, nrec = data.shape
    n_vel = len(vels)
    semb = np.zeros((nt, n_vel), dtype=np.float32)
    off2 = offsets**2
    half_w = win_len // 2
    
    for iv in prange(n_vel):
        v2 = vels[iv]**2 + 1e-9
        num = np.zeros(nt, dtype=np.float32); den = np.zeros(nt, dtype=np.float32)
        for it in range(nt):
            t0 = times[it]; s_amp = 0.0; s_sq = 0.0
            for ir in range(nrec):
                t_hyp = np.sqrt(t0**2 + off2[ir]/v2)
                idx = int(round(t_hyp/dt))
                if idx < nt:
                    val = data[idx, ir]
                    s_amp += val; s_sq += val*val
            num[it] = s_amp**2; den[it] = s_sq
        for it in range(nt):
            s = max(0, it - half_w); e = min(nt, it + half_w + 1)
            sn = 0.0; sd = 0.0
            for k in range(s, e): sn += num[k]; sd += den[k]
            if sd > 1e-9: semb[it, iv] = sn / (nrec * sd)
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_nmo(data, out, offsets, times, v_rms, dt, limit):
    nt, nrec = data.shape
    for i in prange(nrec):
        off2 = offsets[i]**2
        for j in range(nt):
            t0 = times[j]; v2 = v_rms[j]**2 + 1e-9
            t_nmo = np.sqrt(t0**2 + off2/v2)
            idx = t_nmo / dt; i0 = int(idx); i1 = i0 + 1
            if i0 >= 0 and i1 < nt:
                w1 = idx - i0
                out[j, i] = (1.0-w1)*data[i0, i] + w1*data[i1, i]
            else: out[j, i] = 0.0

# ==============================================================================
# 3. LÓGICA DE VELOCIDADE
# ==============================================================================
def build_trace_from_picks(t_picks, v_rms_picks, Nt, dt):
    t_p, v_p = np.array(t_picks), np.array(v_rms_picks)
    idx_p = np.clip(np.round(t_p / dt).astype(int), 0, Nt-1)
    f_rms = interp1d(t_p, v_p, kind='linear', bounds_error=False, fill_value=(v_p[0], v_p[-1]))
    tr_rms = f_rms(np.arange(Nt)*dt).astype(np.float32)
    tr_int = np.zeros(Nt, dtype=np.float32)
    tr_int[:idx_p[0]] = v_p[0]
    for i in range(1, len(t_p)):
        num = (v_p[i]**2 * t_p[i]) - (v_p[i-1]**2 * t_p[i-1])
        den = t_p[i] - t_p[i-1]
        v_dix = np.sqrt(num / den) if num > 0 and den > 0 else v_p[i-1]
        tr_int[idx_p[i-1]:idx_p[i]] = v_dix
    tr_int[idx_p[-1]:] = tr_int[idx_p[-1]-1] if idx_p[-1] > 0 else v_p[-1]
    if VERTICAL_SMOOTH_MS > 0:
        sigma = (VERTICAL_SMOOTH_MS / dt) / 4.0
        tr_rms = gaussian_filter1d(tr_rms, sigma=sigma, mode='nearest')
        tr_int = gaussian_filter1d(tr_int, sigma=sigma, mode='nearest')
    return tr_rms, tr_int

# ==============================================================================
# 4. CLASSE PICKER
# ==============================================================================
class FixedPicker:
    def __init__(self):
        self.h = pd.read_csv(FILE_HEAD)
        self.h.columns = self.h.columns.str.strip().str.lower()
        self.h.rename(columns={'cmp_x':'cmp', 'offset_x':'offset'}, inplace=True)
        if 'global_trace_index' not in self.h.columns: self.h['global_trace_index'] = self.h.index
        self.m = np.memmap(FILE_BIN, dtype='float32', mode='r', shape=(Nt, len(self.h)), order='F')
        folds = self.h['cmp'].value_counts().sort_index()
        self.valid = folds[folds >= MIN_FOLD].index.values
        self.nav = self.valid[::max(1, int(len(self.valid)/50))]
        self.curr = (np.abs(self.nav - folds.idxmax())).argmin()
        self.t = np.arange(Nt, dtype=np.float32)*dt
        self.v = np.arange(V_MIN, V_MAX+V_STEP, V_STEP, dtype=np.float32)
        self.picks = {}
        if not IGNORE_OLD_PICKS and os.path.exists(OUT_PICKS):
            df = pd.read_csv(OUT_PICKS)
            for c in df['CMP'].unique():
                self.picks[int(c)] = list(zip(df[df['CMP']==c]['Time'], df[df['CMP']==c]['Velocity']))
        self.init_ui(); self.load_cmp()

    def init_ui(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
        plt.subplots_adjust(bottom=0.2, top=0.92, wspace=0.15)
        ax_sl = self.fig.add_axes([0.2, 0.08, 0.6, 0.03])
        self.sl = Slider(ax_sl, 'CMP', 0, len(self.nav)-1, valinit=self.curr, valstep=1)
        self.sl.on_changed(self.on_slide)
        kw = {'aspect':'auto', 'interpolation':'bilinear' if SUAVIZAR else 'nearest'}
        self.im1 = self.ax1.imshow(np.zeros((Nt, len(self.v))), cmap='jet', **kw)
        self.ln1, = self.ax1.plot([], [], 'w-o', lw=2, markeredgecolor='k')
        self.im2 = self.ax2.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        self.im3 = self.ax3.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        self.ax1.set_title("Semblance", fontweight='bold'); self.ax2.set_title("Original", fontweight='bold'); self.ax3.set_title("NMO Preview", fontweight='bold')
        self.ax1.set_ylabel("Tempo (s)"); self.ax1.set_xlabel("Velocidade (m/s)"); self.ax1.set_ylim(self.t[-1], 0)
        self.ax2.set_xlabel("Offset (m)"); self.ax3.set_xlabel("Offset (m)")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click); self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.text(0.5, 0.02, "[S] SALVAR | [E] EXPORTAR", ha='center', fontweight='bold', bbox=dict(facecolor='#eeeeee'))

    def on_slide(self, val): self.curr = int(val); self.load_cmp()

    def get_vel_trace(self, target_c):
        p_cmps = np.sort([c for c in self.picks.keys() if self.picks[c]])
        if not p_cmps.size: return np.full(Nt, 1500.0)
        idx = np.searchsorted(p_cmps, target_c)
        if len(p_cmps) == 1 or idx == 0: c1 = c2 = p_cmps[0]; w = 0.0
        elif idx == len(p_cmps): c1 = c2 = p_cmps[-1]; w = 0.0
        else: c1, c2 = p_cmps[idx-1], p_cmps[idx]; w = (target_c - c1) / (c2 - c1)
        t1, v1 = zip(*sorted(self.picks[c1])); tr1 = np.interp(self.t, t1, v1, left=v1[0], right=v1[-1])
        t2, v2 = zip(*sorted(self.picks[c2])); tr2 = np.interp(self.t, t2, v2, left=v2[0], right=v2[-1])
        return (1-w)*tr1 + w*tr2

    def load_cmp(self):
        c = self.nav[self.curr]; df = self.h[(self.h['cmp'] >= c-0.1) & (self.h['cmp'] <= c+0.1)]
        idxs = df['global_trace_index'].values; off = df['offset'].values; s = np.argsort(np.abs(off))
        self.off = off[s]; raw_data = np.array(self.m[:, idxs[s]])
        self.d_view = raw_data * (self.t[:, None]**GANHO_VISUAL)
        vm = np.percentile(np.abs(self.d_view), 98) or 1.0; self.d_view /= vm
        semb = numba_semblace(self.d_view, self.off, self.t, self.v, dt, int(SEMBLANCE_WIN_MS/1000/dt))
        self.im1.set_data(semb); self.im1.set_extent([V_MIN, V_MAX, self.t[-1], 0]); self.im1.set_clim(0, np.percentile(semb, 99.5))
        self.im2.set_data(self.d_view); self.im2.set_extent([self.off.min(), self.off.max(), self.t[-1], 0]); self.im2.set_clim(-1, 1)
        self.im3.set_extent([self.off.min(), self.off.max(), self.t[-1], 0]); self.curr_p = self.picks.get(int(c), []); self.update_ov()

    def update_ov(self):
        vc = self.get_vel_trace(self.nav[self.curr]); nmo = np.zeros_like(self.d_view)
        numba_nmo(self.d_view, nmo, self.off, self.t, vc.astype(np.float32), dt, STRETCH_LIM/100)
        self.im3.set_data(nmo); self.im3.set_clim(-1, 1)
        if self.curr_p: self.curr_p.sort(); ts, vs = zip(*self.curr_p); self.ln1.set_data(vs, ts)
        else: self.ln1.set_data([], [])
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes != self.ax1: return
        if ev.button == 1: self.curr_p.append((ev.ydata, ev.xdata))
        elif ev.button == 3 and self.curr_p:
            d = [((t-ev.ydata)**2 + ((v-ev.xdata)/1000)**2) for t,v in self.curr_p]; self.curr_p.pop(np.argmin(d))
        self.picks[int(self.nav[self.curr])] = self.curr_p; self.update_ov()

    def on_key(self, ev):
        if ev.key == 'right': self.sl.set_val(min(len(self.nav)-1, self.curr+1))
        elif ev.key == 'left': self.sl.set_val(max(0, self.curr-1))
        elif ev.key == 's': 
            rows = [{'CMP':c,'Time':t,'Velocity':v} for c,p in self.picks.items() for (t, v) in p]
            if rows: pd.DataFrame(rows).to_csv(OUT_PICKS, index=False); print("Picks Salvos.")
        elif ev.key == 'e': run_full_export(self)

# ==============================================================================
# 5. EXPORTAÇÃO E QCs
# ==============================================================================
def run_full_export(app):
    print("\n[EXPORTANDO MODELO 2D...]")
    # CORREÇÃO: Desempacotamento de tuplas (t, v) da lista p
    rows = [{'CMP':c,'Time':t,'Velocity':v} for c,p in app.picks.items() for (t, v) in p]
    if not rows: print("ERRO: Nenhum pick encontrado."); return
    df_p = pd.DataFrame(rows); df_p.to_csv(OUT_PICKS, index=False)
    p_cmps = np.sort(df_p['CMP'].unique()); all_cmps = np.sort(app.h['cmp'].unique())
    v_rms_2d = np.zeros((Nt, len(all_cmps)), dtype=np.float32); v_int_2d = np.zeros_like(v_rms_2d)
    profiles = {c: build_trace_from_picks(df_p[df_p['CMP']==c]['Time'].values, df_p[df_p['CMP']==c]['Velocity'].values, Nt, dt) for c in p_cmps}
    for i, c in enumerate(all_cmps):
        idx = np.searchsorted(p_cmps, c)
        if idx == 0: v_rms_2d[:,i], v_int_2d[:,i] = profiles[p_cmps[0]]
        elif idx == len(p_cmps): v_rms_2d[:,i], v_int_2d[:,i] = profiles[p_cmps[-1]]
        else:
            c1, c2 = p_cmps[idx-1], p_cmps[idx]; w = (c - c1)/(c2 - c1)
            v_rms_2d[:,i] = (1-w)*profiles[c1][0] + w*profiles[c2][0]
            v_int_2d[:,i] = (1-w)*profiles[c1][1] + w*profiles[c2][1]
    v_rms_2d.tofile(OUT_RMS); v_int_2d.tofile(OUT_INT)
    fp_o = np.memmap(OUT_NMO, dtype='float32', mode='w+', shape=(Nt, len(app.h)), order='F')
    for i, c in enumerate(tqdm(all_cmps, desc="Exportando NMO")):
        idxs = app.h[app.h['cmp'] == c].index.values; off = app.h.iloc[idxs]['offset'].values.astype(np.float32)
        raw = np.array(app.m[:, idxs]); nmo = np.zeros_like(raw)
        numba_nmo(raw, nmo, off, app.t, v_rms_2d[:, i], dt, STRETCH_LIM/100); fp_o[:, idxs] = nmo
    
    # JANELAS DE QC COMPLETAS
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7))
    im1 = axA.imshow(v_rms_2d, aspect='auto', cmap='jet', extent=[all_cmps[0], all_cmps[-1], Nt*dt, 0])
    axA.set_title("V-RMS densificada (m/s)"); axA.set_xlabel("CMP"); axA.set_ylabel("Tempo (s)"); plt.colorbar(im1, ax=axA)
    im2 = axB.imshow(v_int_2d, aspect='auto', cmap='jet', extent=[all_cmps[0], all_cmps[-1], Nt*dt, 0])
    axB.set_title("V-Intervalar densificada (m/s)"); axB.set_xlabel("CMP"); plt.colorbar(im2, ax=axB)
    
    c_qc = all_cmps[len(all_cmps)//2]; idx_v = np.abs(all_cmps - c_qc).argmin()
    df_q = app.h[app.h['cmp'] == c_qc]; raw_q = np.array(app.m[:, df_q.index.values]); corr_q = np.zeros_like(raw_q)
    numba_nmo(raw_q, corr_q, df_q['offset'].values.astype(np.float32), app.t, v_rms_2d[:, idx_v], dt, STRETCH_LIM/100)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    gain = (app.t[:, None]**GANHO_VISUAL); vm = np.percentile(np.abs(raw_q * gain), 98)
    ax1.imshow(raw_q * gain, aspect='auto', cmap='gray', vmin=-vm, vmax=vm, extent=[0, df_q.shape[0], Nt*dt, 0]); ax1.set_title("Antes (CMP)")
    ax2.imshow(corr_q * gain, aspect='auto', cmap='gray', vmin=-vm, vmax=vm, extent=[0, df_q.shape[0], Nt*dt, 0]); ax2.set_title("Depois (NMO)")
    plt.show()

if __name__ == "__main__":
    app = FixedPicker(); plt.show()