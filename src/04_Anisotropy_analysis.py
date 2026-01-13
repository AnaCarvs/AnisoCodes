import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from math import floor, sqrt
from numba import jit, prange
from scipy.interpolate import interp1d
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = r'C:/Users/AnaCarvs.GISIS/Desktop/Dataset'

# Entrada (NMO CORRIGIDO do passo anterior)
FILE_NMO_IN = f'{BASE_DIR}/Linha/Line_NMO_Corrected_AP2.bin'
FILE_HEAD   = f'{BASE_DIR}/Linha/Trace_Headers_AP2.csv'
FILE_RMS    = f'{BASE_DIR}/Propriedades/Velocity_Model_RMS_AP2.bin' 

# Saídas
OUT_PICKS_ETA = f'{BASE_DIR}/Picks/Eta_Picks_AP2.csv'
OUT_ETA_MODEL = f'{BASE_DIR}/Propriedades/Eta_Model_AP2.bin'
OUT_FINAL_VTI = f'{BASE_DIR}/Linha/Line_Aniso_Corrected_Final_AP2.bin'

IGNORE_OLD_PICKS = False  

# Geometria
Nt = 1501
dt = 0.001
MIN_FOLD = 50

# Scan de Eta (Residual)
ETA_MIN = 0.0
ETA_MAX = 0.3 
ETA_STEP = 0.005

# Visualização
GANHO_VISUAL = 2.5
SUAVIZAR = False
SEMBLANCE_WIN_MS = 40.0

# ==============================================================================
# 2. KERNELS NUMBA (CÁLCULO RESIDUAL - MANTIDO)
# ==============================================================================

@jit(nopython=True, fastmath=True)
def calc_residual_shift(t0, offset, v_nmo, eta):
    """ Calcula APENAS o shift residual (Diferença entre VTI e Hiperbólico). """
    if v_nmo == 0 or eta == 0: return 0.0
    v2 = v_nmo**2; x2 = offset**2; t02 = t0**2
    
    t_hyp_sq = t02 + (x2 / v2)
    t_hyp = sqrt(t_hyp_sq)
    
    denom = t02 * v2 + (1.0 + 2.0 * eta) * x2
    if denom == 0: return 0.0
    
    correction = (2.0 * eta * x2 * x2) / (v2 * denom)
    t_vti_sq = t_hyp_sq - correction
    
    if t_vti_sq < 0: return 0.0
    t_vti = sqrt(t_vti_sq)
    return t_vti - t_hyp

@jit(nopython=True, parallel=True, fastmath=True)
def numba_residual_semblance(nmo_data, offsets, times, v_rms, etas, dt, win_len):
    """ Semblance Residual """
    nt, nrec = nmo_data.shape
    n_eta = len(etas)
    semb = np.zeros((nt, n_eta), dtype=np.float32)
    half_w = win_len // 2
    
    for ie in prange(n_eta):
        eta_val = etas[ie]
        num = np.zeros(nt, dtype=np.float32)
        den = np.zeros(nt, dtype=np.float32)
        for it in range(nt):
            t0 = times[it]; v_val = v_rms[it]
            if v_val < 100: continue
            s_amp = 0.0; s_sq = 0.0
            for ir in range(nrec):
                shift = calc_residual_shift(t0, offsets[ir], v_val, eta_val)
                t_read = t0 + shift
                idx = int(t_read/dt + 0.5)
                if 0 <= idx < nt:
                    val = nmo_data[idx, ir]
                    s_amp += val; s_sq += val*val
            num[it] = s_amp**2; den[it] = s_sq
        for it in range(nt):
            s = max(0, it - half_w); e = min(nt, it + half_w + 1)
            sn = 0.0; sd = 0.0
            for k in range(s, e): sn += num[k]; sd += den[k]
            if sd > 1e-9: semb[it, ie] = sn / (nrec * sd)
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_apply_residual(nmo_data, out, offsets, times, v_rms, eta_trace, dt):
    """ Aplica a correção residual final """
    nt, nrec = nmo_data.shape
    for i in prange(nrec):
        off = offsets[i]
        for j in range(nt):
            t0 = times[j]; v_val = v_rms[j]; eta_val = eta_trace[j]
            if v_val < 100: continue
            shift = calc_residual_shift(t0, off, v_val, eta_val)
            t_read = t0 + shift
            it_frac = t_read / dt
            it_floor = floor(it_frac); it_ceil = it_floor + 1; w = it_frac - it_floor
            if 0 <= it_floor and it_ceil < nt:
                out[j, i] = (1.0 - w) * nmo_data[it_floor, i] + w * nmo_data[it_ceil, i]
            else: out[j, i] = 0.0

# ==============================================================================
# 3. INTERPOLAÇÃO E QC (MODIFICADO: JANELAS SEPARADAS)
# ==============================================================================
def create_vertical_eta_profile(times, etas, target_t_axis):
    if len(times) == 0: return np.zeros_like(target_t_axis)
    df = pd.DataFrame({'t': times, 'e': etas}).sort_values('t')
    f = interp1d(df['t'].values, df['e'].values, kind='linear', 
                 bounds_error=False, fill_value=(df['e'].values[0], df['e'].values[-1]))
    return f(target_t_axis).astype(np.float32)

def plot_qc_residual(qc_data, taxis):
    """ QC MODIFICADO: Abre uma janela separada para cada CMP (Near, Mid, Far) """
    
    for label, (iso, aniso, eta, off) in qc_data.items():
        # Cria uma nova figura para CADA CMP, maximizando a altura
        fig, axs = plt.subplots(1, 3, figsize=(18, 9)) 
        fig.canvas.manager.set_window_title(f"QC Anisotropia: {label}")
        
        # Clips
        vm = np.percentile(np.abs(iso), 99) or 1
        opts = dict(cmap="gray_r", aspect="auto", vmin=-vm, vmax=vm, origin='upper',
                    extent=[off.min(), off.max(), taxis[-1], taxis[0]])
        
        # 1. Input (Iso NMO)
        axs[0].imshow(iso, **opts)
        axs[0].set_title(f"Input: Isotropic NMO\n{label}", fontweight='bold')
        axs[0].set_ylabel("Time [s]")
        axs[0].set_xlabel("Offset [m]")
        axs[0].grid(True, which='both', color='cyan', alpha=0.3, linestyle='--') # GRID ADICIONADO

        # 2. Output (Aniso)
        axs[1].imshow(aniso, **opts)
        axs[1].set_title(f"Output: VTI Corrected\n{label}", fontweight='bold')
        axs[1].set_yticklabels([])
        axs[1].set_xlabel("Offset [m]")
        axs[1].grid(True, which='both', color='cyan', alpha=0.3, linestyle='--') # GRID ADICIONADO
        
        # 3. Eta Profile
        axs[2].plot(eta, taxis, 'g-', lw=2, label='Eta Interpolado')
        axs[2].set_title("Eta Profile", fontweight='bold')
        axs[2].set_ylim(taxis[-1], 0)
        axs[2].set_xlim(ETA_MIN, ETA_MAX)
        axs[2].grid(True, which='both', alpha=0.5)
        axs[2].set_xlabel("Eta Parameter")
        axs[2].legend()
        
        plt.tight_layout()
        plt.show(block=False) # Não bloqueia para abrir as 3 ao mesmo tempo

# ==============================================================================
# 4. INTERFACE
# ==============================================================================
class FixedEtaPicker:
    def __init__(self):
        # Carrega Headers e Dados
        self.h = pd.read_csv(FILE_HEAD)
        self.h.columns = self.h.columns.str.strip().str.lower()
        if 'cmp_x' in self.h.columns: self.h.rename(columns={'cmp_x':'cmp'}, inplace=True)
        if 'offset_x' in self.h.columns: self.h.rename(columns={'offset_x':'offset'}, inplace=True)
        if 'global_trace_index' not in self.h.columns: self.h['global_trace_index'] = self.h.index
        
        print(f"Lendo NMO Input: {FILE_NMO_IN}")
        if not os.path.exists(FILE_NMO_IN): raise FileNotFoundError("Arquivo NMO não encontrado.")
        self.m = np.memmap(FILE_NMO_IN, dtype='float32', mode='r', shape=(Nt, len(self.h)), order='F')
        
        print(f"Lendo Velocidades: {FILE_RMS}")
        if not os.path.exists(FILE_RMS): raise FileNotFoundError("Modelo de Velocidade não encontrado.")
        unique_cmps = np.sort(self.h['cmp'].unique())
        self.vrms_model = np.memmap(FILE_RMS, dtype='float32', mode='r', shape=(Nt, len(unique_cmps)), order='C')
        self.cmp_map = {c: i for i, c in enumerate(unique_cmps)}
        
        folds = self.h['cmp'].value_counts().sort_index()
        self.valid = folds[folds >= MIN_FOLD].index.values
        self.nav = self.valid[::max(1, int(len(self.valid)/50))] 
        self.curr = (np.abs(self.nav - folds.idxmax())).argmin()
        
        self.t = np.arange(Nt, dtype=np.float32)*dt
        self.eta_axis = np.arange(ETA_MIN, ETA_MAX+ETA_STEP, ETA_STEP, dtype=np.float32)
        
        self.picks = {}
        if not IGNORE_OLD_PICKS and os.path.exists(OUT_PICKS_ETA):
            print("Carregando picks anteriores...")
            df = pd.read_csv(OUT_PICKS_ETA)
            for c in df['CMP'].unique():
                self.picks[int(c)] = list(zip(df[df['CMP']==c]['Time'], df[df['CMP']==c]['Eta']))
        
        self.init_ui()
        self.load_cmp()

    def init_ui(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
        plt.subplots_adjust(bottom=0.2, top=0.92, wspace=0.15)
        
        ax_sl = self.fig.add_axes([0.2, 0.08, 0.6, 0.03])
        self.sl = Slider(ax_sl, 'CMP', 0, len(self.nav)-1, valinit=self.curr, valstep=1)
        self.sl.on_changed(self.on_slide)
        
        kw = {'aspect':'auto', 'interpolation':'bilinear' if SUAVIZAR else 'nearest', 'origin':'upper'}
        
        self.im1 = self.ax1.imshow(np.zeros((Nt, len(self.eta_axis))), cmap='jet', **kw)
        self.ln1, = self.ax1.plot([], [], 'w-o', lw=2, markeredgecolor='k')
        
        self.im2 = self.ax2.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        self.im3 = self.ax3.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        
        self.ax1.set_title("Residual Eta Semblance"); self.ax1.set_xlabel("Eta")
        self.ax2.set_title("Input (Isotropic NMO)"); 
        self.ax3.set_title("Corrected (VTI Residual)")
        self.ax1.set_ylim(self.t[-1], 0)
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.text(0.5, 0.02, "[S] SALVAR | [E] EXPORTAR | [R] RESET PICKS | Setas: Navegar", ha='center', fontweight='bold')

    def on_slide(self, val): 
        self.curr = int(val); self.load_cmp()

    def load_cmp(self):
        c = self.nav[self.curr]
        df = self.h[(self.h['cmp'] >= c-0.1) & (self.h['cmp'] <= c+0.1)]
        idxs = df['global_trace_index'].values; off = df['offset'].values
        s = np.argsort(np.abs(off))
        self.off = off[s]; self.d_view = np.array(self.m[:, idxs[s]]) 
        
        idx = self.cmp_map.get(int(c)) or self.cmp_map[list(self.cmp_map.keys())[0]]
        self.v_trace = np.array(self.vrms_model[:, idx])
        
        ext_semb = [ETA_MIN, ETA_MAX, self.t[-1], 0]
        ext_gath = [self.off.min(), self.off.max(), self.t[-1], 0]
        
        semb = numba_residual_semblance(self.d_view, self.off, self.t, self.v_trace, self.eta_axis, dt, int(SEMBLANCE_WIN_MS/1000/dt))
        self.im1.set_data(semb); self.im1.set_extent(ext_semb); self.im1.set_clim(0, np.percentile(semb, 99.5))
        
        vm = np.percentile(np.abs(self.d_view), 98) or 1
        self.im2.set_data(self.d_view); self.im2.set_extent(ext_gath); self.im2.set_clim(-vm, vm)
        self.im3.set_extent(ext_gath)
        
        self.curr_p = self.picks.get(int(c), []); self.update_ov()

    def get_interpolated_eta(self, target_cmp):
        picked = np.sort([c for c in self.picks.keys() if self.picks[c]])
        if not picked.size: return np.zeros(Nt, dtype=np.float32)
        
        if target_cmp in picked:
            ts, es = zip(*sorted(self.picks[target_cmp]))
            return create_vertical_eta_profile(ts, es, self.t)
            
        idx = np.searchsorted(picked, target_cmp)
        if idx == 0: ref = picked[0]
        elif idx == len(picked): ref = picked[-1]
        else: 
            c1, c2 = picked[idx-1], picked[idx]
            w = (target_cmp - c1)/(c2 - c1)
            p1 = create_vertical_eta_profile(*zip(*sorted(self.picks[c1])), self.t)
            p2 = create_vertical_eta_profile(*zip(*sorted(self.picks[c2])), self.t)
            return (1-w)*p1 + w*p2

        ts, es = zip(*sorted(self.picks[ref]))
        return create_vertical_eta_profile(ts, es, self.t)

    def update_ov(self):
        eta_prof = self.get_interpolated_eta(self.nav[self.curr])
        out = np.zeros_like(self.d_view)
        numba_apply_residual(self.d_view, out, self.off, self.t, self.v_trace, eta_prof, dt)
        
        self.im3.set_data(out); self.im3.set_clim(self.im2.get_clim())
        if self.curr_p:
            self.curr_p.sort(); ts, es = zip(*self.curr_p); self.ln1.set_data(es, ts)
        else: self.ln1.set_data([], [])
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes != self.ax1: return
        if ev.button == 1: self.curr_p.append((ev.ydata, ev.xdata))
        elif ev.button == 3 and self.curr_p: 
            d = [((t-ev.ydata)**2 + (e-ev.xdata)**2) for t,e in self.curr_p]
            self.curr_p.pop(np.argmin(d))
        self.picks[int(self.nav[self.curr])] = self.curr_p; self.update_ov()

    def on_key(self, ev):
        if ev.key == 'right': self.sl.set_val(min(len(self.nav)-1, self.curr+1))
        elif ev.key == 'left': self.sl.set_val(max(0, self.curr-1))
        elif ev.key == 'r': # TECLA DE ATALHO PARA RESETAR
            self.picks[int(self.nav[self.curr])] = []
            self.curr_p = []
            self.update_ov()
            print(f">> Picks do CMP {int(self.nav[self.curr])} resetados.")
        elif ev.key == 's': 
            rows = [{'CMP':c,'Time':t,'Eta':e} for c,p in self.picks.items() for (t, e) in p]
            if rows: pd.DataFrame(rows).to_csv(OUT_PICKS_ETA, index=False); print(">> Picks Salvos.")
        elif ev.key == 'e': export_eta(self)

def export_eta(app):
    print("\n--- EXPORTAÇÃO RESIDUAL ---")
    rows = [{'CMP':c,'Time':t,'Eta':e} for c,p in app.picks.items() for (t, e) in p]
    if rows: pd.DataFrame(rows).to_csv(OUT_PICKS_ETA, index=False)
    
    all_cmps = np.sort(app.h['cmp'].unique())
    eta_model = np.zeros((Nt, len(all_cmps)), dtype=np.float32)
    picked = np.sort([c for c in app.picks.keys() if app.picks[c]])
    
    prof_cache = {}
    for c in picked:
        prof_cache[c] = create_vertical_eta_profile(*zip(*sorted(app.picks[c])), app.t)
        
    for i, c in enumerate(tqdm(all_cmps, desc="Interpolando Modelo")):
        if c in picked: eta_model[:, i] = prof_cache[c]; continue
        if not picked.size: continue
        idx = np.searchsorted(picked, c)
        if idx==0: eta_model[:, i] = prof_cache[picked[0]]
        elif idx==len(picked): eta_model[:, i] = prof_cache[picked[-1]]
        else:
            c1, c2 = picked[idx-1], picked[idx]
            w = (c - c1)/(c2 - c1)
            eta_model[:, i] = (1-w)*prof_cache[c1] + w*prof_cache[c2]
            
    eta_model.tofile(OUT_ETA_MODEL)
    
    print(">> Aplicando Correção Residual...")
    fp_o = np.memmap(OUT_FINAL_VTI, dtype='float32', mode='w+', shape=(Nt, len(app.h)), order='F')
    grouped = app.h.groupby('cmp')
    qc_data = {}
    qc_cmps = [all_cmps[0], all_cmps[len(all_cmps)//2], all_cmps[-1]]
    qc_labels = ['NEAR CMP', 'MID CMP', 'FAR CMP']
    
    for i, c in enumerate(tqdm(all_cmps, desc="Applying")):
        if c not in grouped.groups: continue
        idxs = grouped.get_group(c).index.values
        off = app.h.loc[idxs, 'offset'].values.astype(np.float32)
        nmo_iso = np.array(app.m[:, idxs])
        
        v_trace = app.vrms_model[:, app.cmp_map.get(c, 0)]
        eta_trace = eta_model[:, i]
        
        out = np.zeros_like(nmo_iso)
        numba_apply_residual(nmo_iso, out, off, app.t, v_trace, eta_trace, dt)
        fp_o[:, idxs] = out
        
        if c in qc_cmps:
            # Determina o Label correto (Near/Mid/Far)
            if c == qc_cmps[0]: lbl = qc_labels[0] + f" ({int(c)})"
            elif c == qc_cmps[1]: lbl = qc_labels[1] + f" ({int(c)})"
            else: lbl = qc_labels[2] + f" ({int(c)})"
            qc_data[lbl] = (nmo_iso.copy(), out.copy(), eta_trace.copy(), off.copy())
            
    print(">> Concluído.")
    if qc_data: plot_qc_residual(qc_data, app.t)

if __name__ == "__main__":
    app = FixedEtaPicker()
    plt.show()