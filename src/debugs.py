import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from scipy.ndimage import gaussian_filter  # Adicionado para suavização
from scipy.interpolate import interp1d
import os
import gc
from tqdm import tqdm
from numba import jit, prange

# ================= CONFIGURAÇÃO =================
BASE_DIR = r'C:\Users\anapa\OneDrive\Área de Trabalho\SeismicModeling2D-master\SeismicModeling2D-master'

F_BIN_IN = f'{BASE_DIR}/outputs/Line_CMP_Muted_AP2.bin'
F_HEAD = f'{BASE_DIR}/outputs/Trace_Headers_AP2.csv'

F_OUT_PICKS = f'{BASE_DIR}/Outputs/Velocity_Picks.csv'
F_OUT_VEL   = f'{BASE_DIR}/Outputs/Velocity_Model_2D.bin'
F_OUT_NMO   = f'{BASE_DIR}/Outputs/Line_NMO_Corrected.bin'

Nt = 1501           
dt = 0.001        

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
        print("--- INICIANDO SUITE V22 (SMOOTH INTERPOLATION) ---")
        if not os.path.exists(F_BIN_IN): return
        
        self.headers = pd.read_csv(F_HEAD)
        self.headers.columns = self.headers.columns.str.strip()
        self.memmap = np.memmap(F_BIN_IN, dtype='float32', mode='r', 
                                shape=(Nt, len(self.headers)), order='F')
        
        col = 'cmp' if 'cmp' in self.headers.columns else 'cmp_x'
        self.all_cmps = np.sort(self.headers[col].unique())
        self.min_cmp, self.max_cmp = self.all_cmps[0], self.all_cmps[-1]
        
        step = max(1, int(CMP_STEP/(self.all_cmps[1]-self.all_cmps[0])))
        self.nav_cmps = self.all_cmps[::step]
        self.curr_idx = len(self.nav_cmps)//2
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        
        self.picks_db = {}
        self.picks_curr = []
        
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
        self.fig1 = plt.figure(figsize=(16, 9))
        self.fig1.canvas.manager.set_window_title("1. Picking Interativo")
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.30, wspace=0.2)
        
        self.ax1_sem = self.fig1.add_axes([0.05, 0.40, 0.28, 0.50])
        self.ax1_gat = self.fig1.add_axes([0.36, 0.40, 0.28, 0.50], sharey=self.ax1_sem)
        self.ax1_nmo = self.fig1.add_axes([0.67, 0.40, 0.28, 0.50], sharey=self.ax1_sem)
        
        self.ax_map = self.fig1.add_axes([0.10, 0.15, 0.80, 0.05])
        self.ax_map.set_yticks([])
        self.ax_map.set_xlim(self.min_cmp, self.max_cmp)
        self.ax_map.set_title("Mapa de Cobertura (CMPs)", fontsize=9)
        self.ax_map.tick_params(axis='x', labelsize=8)
        self.ax_map.plot(self.all_cmps, np.zeros_like(self.all_cmps), 'o', color='lightgray', markersize=2)
        self.ln_map_done, = self.ax_map.plot([], [], 'o', color='green', markersize=5, label='Picks Salvos')
        self.ln_map_curr, = self.ax_map.plot([self.curr_cmp], [0], 'o', color='red', markersize=8, markeredgecolor='k', label='Atual')
        self.ax_map.legend(loc='lower right', fontsize=8)
        
        self.fig1.text(0.5, 0.03, 
                       "ATALHOS: [S] Salvar | [E] Exportar | [R] Reset | [Setas] Navegar | [Esq] Picar | [Dir] Apagar", 
                       fontweight='bold', ha='center', fontsize=10, 
                       bbox=dict(facecolor='#f0f0f0', edgecolor='gray', boxstyle='round,pad=0.5'))

        self.im_sem = self.ax1_sem.imshow(np.zeros((10,10)), aspect='auto', cmap='jet', vmin=0, vmax=1)
        self.ln_pick, = self.ax1_sem.plot([], [], 'r-o', lw=2.0, markersize=5, zorder=100)
        
        self.im_gat = self.ax1_gat.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.im_nmo = self.ax1_nmo.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.hyp_lines = []
        
        self.ax1_sem.set_title("Semblance", fontsize=11, fontweight='bold')
        self.ax1_sem.set_xlabel("Velocidade (m/s)", fontsize=9)
        self.ax1_sem.set_ylabel("Tempo (s)", fontsize=9)
        
        cbar = plt.colorbar(self.im_sem, ax=self.ax1_sem, orientation='horizontal', pad=0.12, fraction=0.05)
        cbar.set_label('Coerência', fontsize=8)
        cbar.ax.tick_params(labelsize=8)

        self.ax1_gat.set_title("Gather Original", fontsize=11, fontweight='bold')
        self.ax1_gat.set_xlabel("Offset (m)", fontsize=9)
        
        self.ax1_nmo.set_title("Prévia NMO", fontsize=11, fontweight='bold')
        self.ax1_nmo.set_xlabel("Offset (m)", fontsize=9)
        
        for ax in [self.ax1_sem, self.ax1_gat, self.ax1_nmo]: 
            ax.set_ylim(self.t_ax[-1], 0)
            ax.grid(True, linestyle=':', alpha=0.5, color='white')
            ax.tick_params(axis='both', labelsize=8)
        
        self.fig1.canvas.mpl_connect('button_press_event', self.on_pick)
        self.fig1.canvas.mpl_connect('key_press_event', self.on_key)

    def load_cmp(self, cmp_val):
        print(f"Carregando CMP {int(cmp_val)}...")
        raw, off, _ = load_cmp_from_disk(self.headers, self.memmap, cmp_val, CMP_BIN_SIZE)
        if raw is None: 
            print("CMP Vazio/Não encontrado.")
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

    def change_cmp_index(self, idx):
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        idx = max(0, min(len(self.nav_cmps)-1, idx))
        self.curr_idx = idx
        self.curr_cmp = float(self.nav_cmps[self.curr_idx])
        self.load_cmp(self.curr_cmp)

    def on_key(self, event):
        if event.key == 'right': self.change_cmp_index(self.curr_idx + 1)
        elif event.key == 'left': self.change_cmp_index(self.curr_idx - 1)
        elif event.key == 's': self.save_picks_disk(None)
        elif event.key == 'e': self.export_data(None)
        elif event.key == 'r':
            self.picks_curr = []
            self.picks_db[int(self.curr_cmp)] = []
            self.update_dynamic()

    def on_pick(self, event):
        if event.inaxes != self.ax1_sem or self.fig1.canvas.toolbar.mode != '': return
        if event.button == 1: self.picks_curr.append((event.ydata, event.xdata))
        elif event.button == 3 and self.picks_curr:
             dists = [((t-event.ydata)**2 + ((v-event.xdata)/1000)**2) for t, v in self.picks_curr]
             self.picks_curr.pop(np.argmin(dists))
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        self.update_dynamic()

    def save_picks_disk(self, event):
        self.picks_db[int(self.curr_cmp)] = list(self.picks_curr)
        rows = []
        for c, p in self.picks_db.items():
            for t, v in p: rows.append({'CMP': c, 'Time': t, 'Velocity': v})
        try:
            pd.DataFrame(rows).to_csv(F_OUT_PICKS, index=False)
            print(f"Picks Salvos em {F_OUT_PICKS}")
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
        print("\n--- EXPORTANDO DADOS (COM SUAVIZAÇÃO) ---")
        self.cleanup_qc()
        self.save_picks_disk(None)
        if not os.path.exists(F_OUT_PICKS): return
        df_p = pd.read_csv(F_OUT_PICKS)
        if df_p.empty: return

        print("1. Criando Malha de Velocidade...")
        
        # 1. Picks únicos
        picked_cmps = np.sort(df_p['CMP'].unique().astype(float))
        
        # 2. Criar perfis verticais nos locais picados
        sparse_vels = np.zeros((Nt, len(picked_cmps)), dtype=np.float32)
        for i, c in enumerate(picked_cmps):
            p = df_p[df_p['CMP'] == c].sort_values('Time')
            sparse_vels[:, i] = np.interp(self.t_ax, p['Time'], p['Velocity'], 
                                          left=p['Velocity'].iloc[0], right=p['Velocity'].iloc[-1])

        # 3. Interpolação Lateral (Scipy)
        if len(picked_cmps) == 1:
            self.vel_model_2d = np.tile(sparse_vels[:, 0:1], (1, len(self.all_cmps)))
        else:
            # Linear para preencher
            f_interp = interp1d(picked_cmps, sparse_vels, kind='linear', axis=1, 
                                bounds_error=False, fill_value=(sparse_vels[:, 0], sparse_vels[:, -1]))
            self.vel_model_2d = f_interp(self.all_cmps).astype(np.float32)
            
            # SUAVIZAÇÃO: Aplica filtro gaussiano leve para remover "quinas"
            # sigma=(vertical_smooth, lateral_smooth)
            # Lateral sigma=20 traces ajuda a mesclar os blocos
            self.vel_model_2d = gaussian_filter(self.vel_model_2d, sigma=(2, 20))

        self.vel_model_2d.tofile(F_OUT_VEL)

        print("2. Aplicando NMO (Exportação)...")
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
            print("Exportação Concluída.")
            self.open_qc_windows(self.vel_model_2d, df_p)
        except Exception as e: print(f"Erro Export: {e}")

    def open_qc_windows(self, vel_model, df_p):
        # JANELA QC 1: MODELO
        self.fig2 = plt.figure(figsize=(12, 6))
        self.fig2.canvas.manager.set_window_title("2. QC Modelo de Velocidade")
        
        plt.subplots_adjust(top=0.88, bottom=0.15)
        
        ax_map = self.fig2.add_subplot(121)
        self.ax_prf = self.fig2.add_subplot(122)
        
        # Escala dinâmica
        dyn_vmin = np.min(vel_model)
        dyn_vmax = np.max(vel_model)
        
        im = ax_map.imshow(vel_model, aspect='auto', cmap='jet', vmin=dyn_vmin, vmax=dyn_vmax,
                           extent=[self.all_cmps[0], self.all_cmps[-1], self.t_ax[-1], 0])
        ax_map.scatter(df_p['CMP'], df_p['Time'], c='k', s=15, label='Picks')
        ax_map.set_title("Modelo 2D Final (Suavizado)", fontweight='bold')
        ax_map.set_xlabel("CMP Index"); ax_map.set_ylabel("Tempo (s)")
        ax_map.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax_map, label="Vrms (m/s)")
        
        idx_curr = np.abs(self.all_cmps - self.curr_cmp).argmin()
        self.ln_prof, = self.ax_prf.plot(vel_model[:, idx_curr], self.t_ax, 'k-', lw=2)
        self.ln_mark_curr = ax_map.axvline(self.all_cmps[idx_curr], color='white', ls='--')
        
        self.ax_prf.set_ylim(self.t_ax[-1], 0); self.ax_prf.grid(True)
        self.ax_prf.set_xlim(dyn_vmin - 50, dyn_vmax + 50)
        self.ax_prf.set_title(f"Perfil CMP {int(self.all_cmps[idx_curr])}", fontweight='bold')
        self.ax_prf.set_xlabel("Velocidade (m/s)"); self.ax_prf.set_ylabel("Tempo (s)")
        
        self.vel_cached = vel_model
        self.fig2.canvas.mpl_connect('button_press_event', self.on_map_click)

        # JANELA QC 2: NMO
        self.fig3 = plt.figure(figsize=(10, 6))
        self.fig3.canvas.manager.set_window_title("3. QC NMO (Supergather)")
        plt.subplots_adjust(bottom=0.20)
        self.ax3_main = self.fig3.add_axes([0.1, 0.25, 0.8, 0.65])
        
        ax_sl = self.fig3.add_axes([0.15, 0.1, 0.5, 0.03])
        self.sl_qc = Slider(ax_sl, "CMP", self.min_cmp, self.max_cmp, valinit=self.curr_cmp, valstep=CMP_BIN_SIZE)
        self.sl_qc.on_changed(self.update_qc_view)
        
        self.txt_qc = TextBox(self.fig3.add_axes([0.75, 0.1, 0.1, 0.05]), "Paineis:", initial=str(self.qc_n_panels))
        self.txt_qc.on_submit(self.on_qc_text)
        
        self.im_multi = self.ax3_main.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.ax3_main.set_title("NMO Control (Vizinhos)", fontweight='bold')
        self.ax3_main.set_xlabel("Traços (Gathers Vizinhos)"); self.ax3_main.set_ylabel("Tempo (s)")
        
        try:
            self.fp_nmo_qc = np.memmap(F_OUT_NMO, dtype='float32', mode='r', shape=(Nt, len(self.headers)), order='F')
            self.update_qc_view(None)
            plt.show()
        except Exception as e: print(f"Erro ao abrir QC NMO: {e}")

    def on_map_click(self, event):
        if event.inaxes != self.fig2.axes[0]: return
        idx = np.searchsorted(self.all_cmps, event.xdata)
        if 0 <= idx < len(self.all_cmps):
            self.ln_prof.set_data(self.vel_cached[:, idx], self.t_ax)
            self.ln_mark_curr.set_xdata([event.xdata])
            self.ax_prf.set_title(f"Perfil CMP {int(event.xdata)}", fontweight='bold')
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
            vm = np.percentile(np.abs(raw), 98)
            if vm > 0: raw /= vm
            panels.append(raw); panels.append(sep)
        if panels:
            full = np.hstack(panels[:-1])
            self.im_multi.set_data(full)
            self.im_multi.set_extent([0, full.shape[1], self.t_ax[-1], 0])
            self.im_multi.set_clim(-1, 1)
            self.ax3_main.set_ylim(self.t_ax[-1], 0)
            self.fig3.canvas.draw_idle()

if __name__ == "__main__":
    app = VelocitySuite()
    plt.show()