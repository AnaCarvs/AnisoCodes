import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os
from math import floor
from numba import jit, prange
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ==============================================================================
# 1. CONFIGURAÇÃO
# ==============================================================================
BASE_DIR = r'C:/Users/AnaCarvs.GISIS/Desktop/Dataset'

# Arquivos
FILE_BIN  = f'{BASE_DIR}/Linha/Line_CMP_Muted_AP2.bin'
FILE_HEAD = f'{BASE_DIR}/Linha/Trace_Headers_AP2.csv'

# Saídas
OUT_PICKS = f'{BASE_DIR}/Picks/Velocity_Picks_AP2.csv'
OUT_RMS   = f'{BASE_DIR}/Propriedades/Velocity_Model_RMS_AP2.bin'
OUT_INT   = f'{BASE_DIR}/Propriedades/Velocity_Model_INT_AP2.bin'
OUT_NMO   = f'{BASE_DIR}/Linha/Line_NMO_Corrected_AP2.bin'

IGNORE_OLD_PICKS = False  

# Geometria
Nt = 1501
dt = 0.001
MIN_FOLD = 50

# Visualização e Processamento
GANHO_VISUAL = 2.5
SUAVIZAR = False          # False = Pixelado (Melhor para QC técnico)
V_MIN = 1400.0; V_MAX = 4500.0; V_STEP = 20.0
SEMBLANCE_WIN_MS = 40.0
VERTICAL_SMOOTH_MS = 0.0  # 0.0 = Respeita fielmente seus picks (WYSIWYG)

# ==============================================================================
# 2. KERNELS NUMBA (Cálculos Precisos)
# ==============================================================================

@jit(nopython=True, parallel=True, fastmath=True)
def numba_semblace(data, offsets, times, vels, dt, win_len):
    """ Cálculo de Semblance Otimizado """
    nt, nrec = data.shape
    n_vel = len(vels)
    semb = np.zeros((nt, n_vel), dtype=np.float32)
    off2 = offsets**2
    half_w = win_len // 2
    
    for iv in prange(n_vel):
        v2 = vels[iv]**2 + 1e-9
        num = np.zeros(nt, dtype=np.float32)
        den = np.zeros(nt, dtype=np.float32)
        
        # Empilhamento
        for it in range(nt):
            t0 = times[it]
            s_amp = 0.0
            s_sq = 0.0
            for ir in range(nrec):
                t_hyp = np.sqrt(t0**2 + off2[ir]/v2)
                idx = int(t_hyp/dt + 0.5)
                if idx < nt:
                    val = data[idx, ir]
                    s_amp += val
                    s_sq += val*val
            num[it] = s_amp**2
            den[it] = s_sq
            
        # Suavização Boxcar
        for it in range(nt):
            s = max(0, it - half_w)
            e = min(nt, it + half_w + 1)
            sn = 0.0; sd = 0.0
            for k in range(s, e): 
                sn += num[k]; sd += den[k]
            if sd > 1e-9: semb[it, iv] = sn / (nrec * sd)
            
    return semb

@jit(nopython=True, parallel=True, fastmath=True)
def numba_nmo(data, out, offsets, times, v_rms, dt):
    """ NMO Kernel com Interpolação Linear Precisa """
    nt, nrec = data.shape
    ot = times[0]
    
    for i in prange(nrec):
        off2 = offsets[i]**2
        for j in range(nt):
            t0 = times[j]
            v_val = v_rms[j]
            if v_val == 0: continue
            
            t_nmo = np.sqrt(t0**2 + off2 / (v_val**2))
            
            # Interpolação linear exata
            it_frac = (t_nmo - ot) / dt
            it_floor = floor(it_frac)
            it_ceil = it_floor + 1
            w = it_frac - it_floor
            
            if 0 <= it_floor and it_ceil < nt:
                out[j, i] = (1.0 - w) * data[it_floor, i] + w * data[it_ceil, i]
            else:
                out[j, i] = 0.0

# ==============================================================================
# 3. INTERPOLAÇÃO DE VELOCIDADE (INDUSTRY STANDARD: LINEAR)
# ==============================================================================

def create_vertical_profile(times, vels, target_t_axis):
    """ Cria perfil vertical interpolando LINEARMENTE a velocidade (Vrms). """
    if len(times) == 0:
        return np.full_like(target_t_axis, 1500.0)
    
    # Garante ordenação temporal e unicidade
    df_temp = pd.DataFrame({'t': times, 'v': vels}).sort_values('t')
    t_sorted = df_temp['t'].values
    v_sorted = df_temp['v'].values
    
    # Interpolação Linear Padrão
    f = interp1d(t_sorted, v_sorted, kind='linear', 
                 bounds_error=False, fill_value=(v_sorted[0], v_sorted[-1]))
    
    return f(target_t_axis).astype(np.float32)

def dix_conversion_2d(v_rms_2d, dt):
    """ Converte RMS -> Intervalar (DIX) """
    nt, ncmp = v_rms_2d.shape
    times = np.arange(nt) * dt
    v2_t = (v_rms_2d**2) * times[:, None]
    
    num = np.diff(v2_t, axis=0, prepend=v2_t[0:1, :])
    den = dt
    
    arg = np.maximum(0, num / den)
    v_int = np.sqrt(arg)
    v_int[0, :] = v_rms_2d[0, :]
    return v_int

# ==============================================================================
# 4. FUNÇÃO DE QC TRIPLO (NEAR, MID, FAR)
# ==============================================================================
def plot_multi_cmp_qc(qc_data_dict, taxis, extent_base):
    """
    Exibe QC para CMP Near, Mid e Far.
    """
    fig, axs = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("NMO Quality Control: Near, Mid, Far CMPs", fontsize=16)
    
    keys = list(qc_data_dict.keys())
    
    for i, label in enumerate(keys):
        raw, nmo, vel, off = qc_data_dict[label]
        
        # Extent: [min_off, max_off, max_time, min_time] -> Y invertido visualmente
        gather_extent = [off.min(), off.max(), taxis[-1], taxis[0]]
        
        # Amplitude Clip
        dmax = np.percentile(np.abs(raw), 99)
        if dmax == 0: dmax = 1
        opts = dict(cmap="gray_r", aspect="auto", vmin=-dmax, vmax=dmax, origin='upper', extent=gather_extent)
        
        # 1. Original
        axs[i, 0].imshow(raw, **opts)
        axs[i, 0].set_title(f"{label} - Original")
        axs[i, 0].set_ylabel("Time [s]")
        if i == 2: axs[i, 0].set_xlabel("Offset [m]")
        
        # 2. NMO
        axs[i, 1].imshow(nmo, **opts)
        axs[i, 1].set_title(f"{label} - NMO Corrected")
        axs[i, 1].set_yticklabels([]) 
        if i == 2: axs[i, 1].set_xlabel("Offset [m]")

        # 3. Velocity
        axs[i, 2].plot(vel, taxis, 'r-', lw=2)
        axs[i, 2].set_title(f"{label} - Vrms Profile")
        axs[i, 2].set_ylim(taxis[-1], taxis[0]) # Inverte Eixo T
        axs[i, 2].set_xlim(1400, 5000) 
        axs[i, 2].grid(True, alpha=0.5)
        if i == 2: axs[i, 2].set_xlabel("Velocity [m/s]")

    plt.show(block=False)

# ==============================================================================
# 5. INTERFACE DE PICKING
# ==============================================================================
class FixedPicker:
    def __init__(self):
        self.h = pd.read_csv(FILE_HEAD)
        self.h.columns = self.h.columns.str.strip().str.lower()
        if 'cmp_x' in self.h.columns: self.h.rename(columns={'cmp_x':'cmp'}, inplace=True)
        if 'offset_x' in self.h.columns: self.h.rename(columns={'offset_x':'offset'}, inplace=True)
        if 'global_trace_index' not in self.h.columns: self.h['global_trace_index'] = self.h.index
        
        print(f"Lendo binário: {FILE_BIN}")
        self.m = np.memmap(FILE_BIN, dtype='float32', mode='r', shape=(Nt, len(self.h)), order='F')
        
        folds = self.h['cmp'].value_counts().sort_index()
        self.valid = folds[folds >= MIN_FOLD].index.values
        self.nav = self.valid[::max(1, int(len(self.valid)/50))] 
        self.curr = (np.abs(self.nav - folds.idxmax())).argmin()
        
        self.t = np.arange(Nt, dtype=np.float32)*dt
        self.v = np.arange(V_MIN, V_MAX+V_STEP, V_STEP, dtype=np.float32)
        
        self.picks = {}
        if not IGNORE_OLD_PICKS and os.path.exists(OUT_PICKS):
            print(f"Carregando picks de {OUT_PICKS}...")
            df = pd.read_csv(OUT_PICKS)
            for c in df['CMP'].unique():
                self.picks[int(c)] = list(zip(df[df['CMP']==c]['Time'], df[df['CMP']==c]['Velocity']))
        
        self.init_ui()
        self.load_cmp()

    def init_ui(self):
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
        plt.subplots_adjust(bottom=0.2, top=0.92, wspace=0.15)
        
        ax_sl = self.fig.add_axes([0.2, 0.08, 0.6, 0.03])
        self.sl = Slider(ax_sl, 'CMP', 0, len(self.nav)-1, valinit=self.curr, valstep=1)
        self.sl.on_changed(self.on_slide)
        
        # Configuração Visual com Eixos Corretos
        kw = {'aspect':'auto', 'interpolation':'bilinear' if SUAVIZAR else 'nearest', 'origin':'upper'}
        
        self.im1 = self.ax1.imshow(np.zeros((Nt, len(self.v))), cmap='jet', **kw)
        self.ln1, = self.ax1.plot([], [], 'w-o', lw=2, markeredgecolor='k')
        
        self.im2 = self.ax2.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        self.im3 = self.ax3.imshow(np.zeros((Nt, 10)), cmap='gray', **kw)
        
        self.ax1.set_title("Semblance"); self.ax2.set_title("CMP Original"); self.ax3.set_title("NMO Preview")
        self.ax1.set_ylim(self.t[-1], 0)
        
        # --- AQUI ESTÃO OS EVENTOS DE MOUSE E TECLADO ---
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key) 
        # --------------------------------------------------
        
        self.fig.text(0.5, 0.02, "[S] SALVAR | [E] EXPORTAR | Setas: Navegar (Se travar, clique na imagem)", ha='center', fontweight='bold')
        print(">> INTERFACE PRONTA. Teclas ativas: Esquerda/Direita, 'S', 'E'.")

    def on_slide(self, val): 
        self.curr = int(val); self.load_cmp()

    def load_cmp(self):
        c = self.nav[self.curr]
        df = self.h[(self.h['cmp'] >= c-0.1) & (self.h['cmp'] <= c+0.1)]
        idxs = df['global_trace_index'].values; off = df['offset'].values
        s = np.argsort(np.abs(off))
        self.off = off[s]; raw_data = np.array(self.m[:, idxs[s]])
        
        self.d_view = raw_data * (self.t[:, None]**GANHO_VISUAL)
        vm = np.percentile(np.abs(self.d_view), 98) or 1.0; self.d_view /= vm
        
        # Extents corrigidos (Y Cresce para baixo)
        ext_semb = [V_MIN, V_MAX, self.t[-1], 0]
        ext_gath = [self.off.min(), self.off.max(), self.t[-1], 0]
        
        semb = numba_semblace(self.d_view, self.off, self.t, self.v, dt, int(SEMBLANCE_WIN_MS/1000/dt))
        self.im1.set_data(semb); self.im1.set_extent(ext_semb); self.im1.set_clim(0, np.percentile(semb, 99.5))
        
        self.im2.set_data(self.d_view); self.im2.set_extent(ext_gath); self.im2.set_clim(-1, 1)
        self.im3.set_extent(ext_gath)
        self.curr_p = self.picks.get(int(c), []); self.update_ov()

    def get_interpolated_vel_profile(self, target_cmp):
        picked_cmps = np.sort([c for c in self.picks.keys() if self.picks[c]])
        if not picked_cmps.size: return np.full(Nt, 1500.0)
        
        # Se tem pick exato, usa ele
        if target_cmp in picked_cmps:
            ts, vs = zip(*sorted(self.picks[target_cmp]))
            return create_vertical_profile(ts, vs, self.t)
            
        idx = np.searchsorted(picked_cmps, target_cmp)
        
        if idx == 0: 
            ts, vs = zip(*sorted(self.picks[picked_cmps[0]]))
            return create_vertical_profile(ts, vs, self.t)
        elif idx == len(picked_cmps):
            ts, vs = zip(*sorted(self.picks[picked_cmps[-1]]))
            return create_vertical_profile(ts, vs, self.t)
        
        c_left, c_right = picked_cmps[idx-1], picked_cmps[idx]
        w = (target_cmp - c_left) / (c_right - c_left)
        
        ts1, vs1 = zip(*sorted(self.picks[c_left]))
        prof_left = create_vertical_profile(ts1, vs1, self.t)
        ts2, vs2 = zip(*sorted(self.picks[c_right]))
        prof_right = create_vertical_profile(ts2, vs2, self.t)
        
        return (1-w)*prof_left + w*prof_right

    def update_ov(self):
        vc = self.get_interpolated_vel_profile(self.nav[self.curr])
        nmo = np.zeros_like(self.d_view)
        numba_nmo(self.d_view, nmo, self.off, self.t, vc.astype(np.float32), dt)
        self.im3.set_data(nmo); self.im3.set_clim(-1, 1)
        if self.curr_p:
            self.curr_p.sort(); ts, vs = zip(*self.curr_p); self.ln1.set_data(vs, ts)
        else: self.ln1.set_data([], [])
        self.fig.canvas.draw_idle()

    def on_click(self, ev):
        if ev.inaxes != self.ax1: return
        if ev.button == 1: self.curr_p.append((ev.ydata, ev.xdata))
        elif ev.button == 3 and self.curr_p:
            d = [((t-ev.ydata)**2 + ((v-ev.xdata)/1000)**2) for t,v in self.curr_p]
            self.curr_p.pop(np.argmin(d))
        self.picks[int(self.nav[self.curr])] = self.curr_p; self.update_ov()

    # --- FUNÇÃO DE ATALHOS RESTAURADA ---
    def on_key(self, ev):
        if ev.key == 'right': 
            self.sl.set_val(min(len(self.nav)-1, self.curr+1))
        elif ev.key == 'left': 
            self.sl.set_val(max(0, self.curr-1))
        elif ev.key == 's': 
            rows = [{'CMP':c,'Time':t,'Velocity':v} for c,p in self.picks.items() for (t, v) in p]
            if rows: pd.DataFrame(rows).to_csv(OUT_PICKS, index=False); print(">> Picks Salvos.")
        elif ev.key == 'e': 
            run_full_export(self)
    # ------------------------------------

# ==============================================================================
# 6. EXPORTAÇÃO E QC
# ==============================================================================
def run_full_export(app):
    print("\n--- INICIANDO EXPORTAÇÃO ---")
    rows = [{'CMP':c,'Time':t,'Velocity':v} for c,p in app.picks.items() for (t, v) in p]
    if not rows: print("ERRO: Picks vazios."); return
    pd.DataFrame(rows).to_csv(OUT_PICKS, index=False)
    
    all_cmps = np.sort(app.h['cmp'].unique())
    picked_cmps = np.sort([c for c in app.picks.keys() if app.picks[c]])
    
    print(f">> Interpolando Modelo...")
    
    profiles_cache = {}
    for c in picked_cmps:
        ts, vs = zip(*sorted(app.picks[c]))
        profiles_cache[c] = create_vertical_profile(ts, vs, app.t)
        
    v_rms_2d = np.zeros((Nt, len(all_cmps)), dtype=np.float32)
    
    for i, c in enumerate(tqdm(all_cmps, desc="Construindo Modelo")):
        if c in picked_cmps:
             v_rms_2d[:, i] = profiles_cache[c]
             continue
        idx = np.searchsorted(picked_cmps, c)
        if idx == 0: v_rms_2d[:, i] = profiles_cache[picked_cmps[0]]
        elif idx == len(picked_cmps): v_rms_2d[:, i] = profiles_cache[picked_cmps[-1]]
        else:
            c_left, c_right = picked_cmps[idx-1], picked_cmps[idx]
            w = (c - c_left) / (c_right - c_left)
            v_rms_2d[:, i] = (1-w)*profiles_cache[c_left] + w*profiles_cache[c_right]
            
    if VERTICAL_SMOOTH_MS > 0:
        sigma = (VERTICAL_SMOOTH_MS / dt) / 2.0
        v_rms_2d = gaussian_filter1d(v_rms_2d, sigma=sigma, axis=0, mode='nearest')
        
    print(">> Exportando Binários...")
    v_int_2d = dix_conversion_2d(v_rms_2d, dt)
    v_rms_2d.tofile(OUT_RMS); v_int_2d.tofile(OUT_INT)
    
    print(">> Aplicando NMO e Gerando QC Triplo...")
    fp_o = np.memmap(OUT_NMO, dtype='float32', mode='w+', shape=(Nt, len(app.h)), order='F')
    grouped = app.h.groupby('cmp')
    
    qc_indices = [0, len(all_cmps)//2, -1] 
    qc_cmps = [all_cmps[i] for i in qc_indices]
    qc_labels = ['NEAR CMP', 'MID CMP', 'FAR CMP']
    qc_data_storage = {} 
    
    for i, c in enumerate(tqdm(all_cmps, desc="NMO")):
        if c not in grouped.groups: continue
        idxs = grouped.get_group(c).index.values
        off = app.h.loc[idxs, 'offset'].values.astype(np.float32)
        raw = np.array(app.m[:, idxs])
        nmo = np.zeros_like(raw)
        
        numba_nmo(raw, nmo, off, app.t, v_rms_2d[:, i], dt)
        fp_o[:, idxs] = nmo
        
        if c in qc_cmps:
            lbl_idx = qc_cmps.index(c)
            lbl = f"{qc_labels[lbl_idx]} ({int(c)})"
            qc_data_storage[lbl] = (raw.copy(), nmo.copy(), v_rms_2d[:, i].copy(), off.copy())
            
    print(">> Concluído. Abrindo Janelas de QC...")
    
    model_extent = [all_cmps[0], all_cmps[-1], Nt*dt, 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    im1 = ax1.imshow(v_rms_2d, aspect='auto', cmap='jet', extent=model_extent, origin='upper')
    ax1.set_title("Modelo RMS Final"); ax1.set_ylabel("Time [s]"); plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(v_int_2d, aspect='auto', cmap='jet', extent=model_extent, origin='upper', vmin=1500, vmax=6000)
    ax2.set_title("Modelo Intervalar"); plt.colorbar(im2, ax=ax2)
    plt.show(block=False)
    
    if qc_data_storage:
        plot_multi_cmp_qc(qc_data_storage, app.t, None)

if __name__ == "__main__":
    app = FixedPicker()
    plt.show()