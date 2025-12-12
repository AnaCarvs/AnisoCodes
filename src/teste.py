"""
Velocity Suite (v15.1) - Versão completa sem slider (Caixa CMP)
- Caixa CMP + botões prev/next
- Barra com range de CMPs e marcadores (verde = picado, red = atual)
- Semblance (numba), NMO (numba)
- Picks por CMP isolados (sem vazamento)
- Supergather final (NMO + média)
- Exporta modelo Vrms e NMO memmap
- Salva/Carrega picks CSV
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import TextBox, Button
from tqdm import tqdm
from numba import njit, prange

# ---------------- CONFIGURAÇÃO (edite conforme necessidade) ----------------
BASE_DIR = r'C:/Users/Anacarvs/Desktop/SeismicModeling2D-master/SeismicModeling2D-master'
F_BIN_IN = f'{BASE_DIR}/outputs/Line_CMP_Sorted.bin'
F_HEAD   = f'{BASE_DIR}/outputs/Trace_Headers.csv'
F_OUT_PICKS = f'{BASE_DIR}/outputs/Velocity_Picks.csv'
F_OUT_VEL   = f'{BASE_DIR}/outputs/Velocity_Model_2D.bin'
F_OUT_NMO   = f'{BASE_DIR}/outputs/Line_NMO_Corrected.bin'

Nt = 2501
dt = 0.001

CMP_BIN_SIZE = 10.0
CMP_STEP = 50.0

INIT_VMIN = 1500.0
INIT_VMAX = 5500.0
INIT_DV   = 25.0
INIT_WIN  = 40.0    # ms
INIT_GAIN = 2.0
VEL_MUTE  = 1500.0
BUF_MUTE  = 200
STRETCH_LIM = 30.0  # percent acceptable stretch for NMO

# ---------------- NUMBA KERNELS ----------------

@njit(parallel=True, fastmath=True)
def numba_semblance(data, offsets, times, vels, dt, win_len_samples):
    nt, nrec = data.shape
    n_vel = vels.shape[0]
    semb = np.zeros((nt, n_vel), dtype=np.float32)
    off2 = offsets * offsets
    half_w = win_len_samples // 2

    for iv in prange(n_vel):
        v = vels[iv]
        v2 = v * v + 1e-9
        num_trace = np.zeros(nt, dtype=np.float32)
        den_trace = np.zeros(nt, dtype=np.float32)

        for it in range(nt):
            t0 = times[it]
            sum_amp = 0.0
            sum_sq = 0.0
            for ir in range(nrec):
                th = np.sqrt(t0 * t0 + off2[ir] / v2)
                if (th - t0) / (t0 + 1e-9) > 0.5:
                    continue
                idx = int(round(th / dt))
                if 0 <= idx < nt:
                    val = data[idx, ir]
                    sum_amp += val
                    sum_sq += val * val
            num_trace[it] = sum_amp * sum_amp
            den_trace[it] = sum_sq

        for it in range(nt):
            start = it - half_w
            if start < 0:
                start = 0
            end = it + half_w + 1
            if end > nt:
                end = nt
            s_num = 0.0
            s_den = 0.0
            for k in range(start, end):
                s_num += num_trace[k]
                s_den += den_trace[k]
            if s_den > 1e-12:
                semb[it, iv] = s_num / (nrec * s_den)
            else:
                semb[it, iv] = 0.0
    return semb

@njit(parallel=True, fastmath=True)
def numba_nmo(data, out, offsets, times, v_rms, dt, stretch_limit):
    nt, nrec = data.shape
    for ir in prange(nrec):
        off2 = offsets[ir] * offsets[ir]
        for it in range(nt):
            t0 = times[it]
            v = v_rms[it]
            denom = v * v + 1e-9
            t_nmo = np.sqrt(t0 * t0 + off2 / denom)
            stretch = (t_nmo - t0) / (t0 + 1e-9)
            if stretch > stretch_limit:
                out[it, ir] = 0.0
                continue
            idxf = t_nmo / dt
            i0 = int(idxf)
            i1 = i0 + 1
            if i0 >= 0 and i1 < nt:
                w1 = idxf - i0
                w0 = 1.0 - w1
                out[it, ir] = w0 * data[i0, ir] + w1 * data[i1, ir]
            else:
                out[it, ir] = 0.0

# ---------------- HELPERS ----------------

def build_cmp_index_corrected(headers, bin_size):
    col = 'cmp' if 'cmp' in headers.columns else ('cmp_x' if 'cmp_x' in headers.columns else None)
    if col is None:
        raise RuntimeError("Header CSV must contain 'cmp' or 'cmp_x' column.")
    idx_col = 'global_trace_index' if 'global_trace_index' in headers.columns else ('g_idx' if 'g_idx' in headers.columns else None)
    if idx_col is None:
        raise RuntimeError("Header CSV must contain 'global_trace_index' or 'g_idx' column.")

    headers = headers.copy()
    keys = np.round(headers[col].values.astype(np.float64) / bin_size) * bin_size
    headers['key'] = keys.astype(np.int64)
    cmp_map = {}
    grouped = headers.groupby('key')
    for key, group in grouped:
        indices = group[idx_col].values.astype(np.int64)
        offsets = group['offset'].values.astype(np.float32)
        sort_idx = np.argsort(offsets)
        cmp_map[int(key)] = (indices[sort_idx], offsets[sort_idx])
    return cmp_map

def load_cmp_fast(cmp_map, memmap, target_cmp):
    key = int(target_cmp)
    if key not in cmp_map:
        return None, None, None
    indices, offsets = cmp_map[key]
    data = np.array(memmap[:, indices], dtype=np.float32)
    return data, offsets, indices

def load_cmp_from_disk(headers, memmap, target, bin_size):
    col = 'cmp' if 'cmp' in headers.columns else ('cmp_x' if 'cmp_x' in headers.columns else None)
    if col is None:
        return None, None, None
    x_min, x_max = target - bin_size/2, target + bin_size/2
    df = headers[(headers[col] >= x_min) & (headers[col] <= x_max)]
    if df.empty:
        return None, None, None
    idx_col = 'global_trace_index' if 'global_trace_index' in df.columns else ('g_idx' if 'g_idx' in df.columns else None)
    if idx_col is None:
        return None, None, None
    indices = df[idx_col].values.astype(np.int64)
    offsets = df['offset'].values.astype(np.float32)
    sort_idx = np.argsort(offsets)
    data = np.array(memmap[:, indices[sort_idx]], dtype=np.float32)
    return data, offsets[sort_idx], indices[sort_idx]

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
    if vm > 0:
        gained /= vm
    return gained

def run_semblance_fast(data, offsets, dt, vels, win_ms):
    times = np.arange(data.shape[0], dtype=np.float32) * dt
    win_samp = int((win_ms/1000)/dt)
    if win_samp < 1:
        win_samp = 1
    return numba_semblance(data.astype(np.float32), offsets.astype(np.float32), times, vels.astype(np.float32), dt, win_samp)

def run_nmo_fast(data, offsets, dt, v_rms, stretch_percent):
    nt = data.shape[0]
    out = np.zeros_like(data)
    times = np.arange(nt, dtype=np.float32) * dt
    limit = stretch_percent / 100.0
    numba_nmo(data.astype(np.float32), out, offsets.astype(np.float32), times, v_rms.astype(np.float32), dt, limit)
    return out

# ---------------- SUPERGATHER (NMO average across selected CMPS) ----------------

def build_supergather_from_picks(cmp_lookup, memmap, picked_cmps, t_ax):
    """
    picked_cmps: list of cmp keys (int)
    This function will:
      - For each CMP, load gather via cmp_lookup & memmap
      - Try to estimate a representative VRMS from picking stored (we expect picks to be (time, vel) or (time, offset))
      - Apply NMO using a single mean velocity (safe simple approach) and stack
    """
    if not picked_cmps:
        return None, None

    nt = len(t_ax)
    accum = None
    count = 0

    # simple strategy: attempt to find velocity picks from headers? fallback to median 2000
    for key in picked_cmps:
        raw, off, idxs = load_cmp_fast(cmp_lookup, memmap, key)
        if raw is None:
            continue
        # compute a simple Vrms estimate per CMP using energy-weighted method
        # fallback to median amplitude -> velocity ~ 2000 if unknown
        v_est = 2000.0
        # apply muted display and nmo with v_est
        data_center = process_display(raw, off, dt, INIT_GAIN)
        v_rms = np.full(nt, v_est, dtype=np.float32)
        nmo = run_nmo_fast(data_center, off, dt, v_rms, STRETCH_LIM)
        if accum is None:
            accum = np.array(nmo, dtype=np.float32)
        else:
            accum += nmo
        count += 1

    if count == 0:
        return None, None
    stacked = accum / count
    return stacked, off

# ---------------- APP (Caixa CMP + prev/next + barra) ----------------

class VelocitySuiteNoSlider:
    def __init__(self):
        print("--- INICIANDO SUITE V15.1 (NO SLIDER, CAIXA CMP) ---")
        if not os.path.exists(F_BIN_IN) or not os.path.exists(F_HEAD):
            print("Arquivos de entrada (BIN/HEAD) não encontrados. Verifique caminhos.")
            return

        # load headers
        self.headers = pd.read_csv(F_HEAD)
        self.headers.columns = self.headers.columns.str.strip()
        self.cmp_lookup = build_cmp_index_corrected(self.headers, CMP_BIN_SIZE)

        ntraces = len(self.headers)
        self.memmap = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, ntraces), order='F')

        self.all_cmps = np.sort(np.array(list(self.cmp_lookup.keys())))
        self.min_cmp, self.max_cmp = int(self.all_cmps[0]), int(self.all_cmps[-1])
        self.cmp_step = int(self.all_cmps[1]-self.all_cmps[0]) if len(self.all_cmps)>1 else int(CMP_BIN_SIZE)

        # picks_db: dict key->list of picks. Picks can be (time, vel) from semblance or (time, offset) from gather
        self.picks_db = {int(c): [] for c in self.all_cmps}
        # attempt to load saved picks if exists
        if os.path.exists(F_OUT_PICKS):
            try:
                dfp = pd.read_csv(F_OUT_PICKS)
                for _, r in dfp.iterrows():
                    c = int(r['CMP'])
                    f1 = r.get('Field1', None)
                    f2 = r.get('Field2', None)
                    if f1 is not None and f2 is not None and c in self.picks_db:
                        self.picks_db[c].append((float(f1), float(f2)))
            except Exception:
                pass

        self.curr_cmp = float(self.all_cmps[len(self.all_cmps)//2])
        self.t_ax = np.arange(Nt, dtype=np.float32) * dt
        self.p_vmin, self.p_vmax = INIT_VMIN, INIT_VMAX
        self.p_dv, self.p_win, self.p_gain = INIT_DV, INIT_WIN, INIT_GAIN
        self.update_v_axis()

        # GUI state
        self.fig = None
        self.ax_sem = None
        self.ax_gat = None
        self.ax_nmo = None
        self.ax_bar = None
        self.semb = None
        self.data_center = None
        self.offsets = None
        self.hyp_lines = []
        self.pick_markers = []

        # Build UI and load initial CMP
        self.build_ui()
        self.load_cmp(self.curr_cmp)

    def update_v_axis(self):
        self.v_ax = np.arange(self.p_vmin, self.p_vmax + self.p_dv, self.p_dv, dtype=np.float32)

    def build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(3, 10, height_ratios=[8, 0.5, 0.8], hspace=0.3, wspace=0.3)

        self.ax_sem = self.fig.add_subplot(gs[0, 0:3])
        self.ax_gat = self.fig.add_subplot(gs[0, 3:7], sharey=self.ax_sem)
        self.ax_nmo = self.fig.add_subplot(gs[0, 7:10], sharey=self.ax_sem)

        # Controls row (middle): CMP input + prev/next + update + clear + save + export
        ctrl_ax = self.fig.add_subplot(gs[1, 0:10])
        ctrl_ax.axis('off')

        # CMP input
        ax_txt = self.fig.add_axes([0.05, 0.165, 0.08, 0.04])
        self.txt_cmp = TextBox(ax_txt, "CMP:", initial=str(int(self.curr_cmp)))
        self.txt_cmp.on_submit(self.on_text_goto)

        # prev/next
        ax_prev = self.fig.add_axes([0.14, 0.165, 0.04, 0.04])
        self.btn_prev = Button(ax_prev, '<<')
        self.btn_prev.on_clicked(lambda ev: self.on_step(-1))
        ax_next = self.fig.add_axes([0.19, 0.165, 0.04, 0.04])
        self.btn_next = Button(ax_next, '>>')
        self.btn_next.on_clicked(lambda ev: self.on_step(+1))

        # update, clear, save, export, close
        ax_upd = self.fig.add_axes([0.25, 0.165, 0.08, 0.04])
        self.btn_upd = Button(ax_upd, 'ATUALIZAR', color='lightblue')
        self.btn_upd.on_clicked(lambda ev: self.load_cmp(self.curr_cmp))

        ax_clr = self.fig.add_axes([0.35, 0.165, 0.06, 0.04])
        self.btn_clr = Button(ax_clr, 'LIMPAR', color='salmon')
        self.btn_clr.on_clicked(self.clear_picks)

        ax_save = self.fig.add_axes([0.43, 0.165, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Salvar Picks', color='lightgreen')
        self.btn_save.on_clicked(self.save_picks_disk)

        ax_exp = self.fig.add_axes([0.53, 0.165, 0.08, 0.04])
        self.btn_exp = Button(ax_exp, 'EXPORTAR', color='gold')
        self.btn_exp.on_clicked(self.export_data)

        ax_close = self.fig.add_axes([0.63, 0.165, 0.06, 0.04])
        self.btn_close = Button(ax_close, 'FECHAR', color='#ffcccc')
        self.btn_close.on_clicked(lambda ev: plt.close('all'))

        # CMP bar (bottom row: show range and markers)
        self.ax_bar = self.fig.add_subplot(gs[2, 0:10])
        self.ax_bar.set_yticks([])
        self.ax_bar.set_xlim(self.min_cmp - self.cmp_step, self.max_cmp + self.cmp_step)
        self.ax_bar.set_title(f"CMP range: {self.min_cmp} — {self.max_cmp} (click marker to jump)")

        # create markers list (scatter artists)
        self.cmp_markers = []
        xs = self.all_cmps
        ys = np.zeros_like(xs, dtype=float)
        for i, x in enumerate(xs):
            mk, = self.ax_bar.plot(x, 0.0, 'o', color='lightgray', picker=5)
            self.cmp_markers.append(mk)

        # interactive clicks: picking on semblance/gather and clicking markers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick_marker)

        # initial axis labels & placeholders
        self.im_sem = self.ax_sem.imshow(np.zeros((10,10)), aspect='auto', cmap='jet')
        self.im_gat = self.ax_gat.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')
        self.im_nmo = self.ax_nmo.imshow(np.zeros((10,10)), aspect='auto', cmap='gray')

        self.ax_sem.set_title('Semblance'); self.ax_sem.set_xlabel('Vel (m/s)'); self.ax_sem.set_ylabel('Tempo (s)')
        self.ax_gat.set_title('Input (Muted)'); self.ax_gat.set_xlabel('Offset (m)')
        self.ax_nmo.set_title('Preview (No Stretch)'); self.ax_nmo.set_xlabel('Offset (m)')

    # ---------------- UI interactions ----------------

    def on_text_goto(self, text):
        try:
            val = int(float(text))
            if val < self.min_cmp or val > self.max_cmp:
                print(f"CMP out of range: {val}")
                return
            self.load_cmp(val)
        except Exception:
            print("Valor inválido para CMP.")

    def on_step(self, step):
        # find current index in all_cmps
        idx = np.searchsorted(self.all_cmps, self.curr_cmp)
        idx = int(idx)
        new_idx = max(0, min(len(self.all_cmps)-1, idx + step))
        self.load_cmp(self.all_cmps[new_idx])
        self.txt_cmp.set_val(str(int(self.curr_cmp)))

    def on_pick_marker(self, event):
        # clicking a marker jumps to that CMP
        artist = event.artist
        # find index
        for i, mk in enumerate(self.cmp_markers):
            if mk == artist:
                self.load_cmp(self.all_cmps[i])
                self.txt_cmp.set_val(str(int(self.curr_cmp)))
                break

    def on_click(self, event):
        # clicking on semblance -> pick (time, vel)
        if event.inaxes == self.ax_sem:
            if event.xdata is None or event.ydata is None:
                return
            t = float(event.ydata)
            v = float(event.xdata)
            self.picks_db[int(self.curr_cmp)].append((t, v))
            self.load_cmp(self.curr_cmp)
            return

        # clicking on gather -> pick (time, offset)
        if event.inaxes == self.ax_gat:
            if event.xdata is None or event.ydata is None:
                return
            off = float(event.xdata)
            t = float(event.ydata)
            self.picks_db[int(self.curr_cmp)].append((t, off))
            self.load_cmp(self.curr_cmp)
            return

    # ---------------- Core load & update ----------------

    def load_cmp(self, cmp_val):
        cmp_key = int(cmp_val)
        raw, off, idxs = load_cmp_from_disk(self.headers, self.memmap, cmp_key, CMP_BIN_SIZE)
        if raw is None:
            print(f"CMP {cmp_key} not found in headers.")
            return

        self.curr_cmp = cmp_key
        self.offsets = off
        self.data_center = process_display(raw, off, dt, self.p_gain)
        # semblance calc (may be heavy) — only compute for displayed v axis
        try:
            self.semb = run_semblance_fast(self.data_center, off, dt, self.v_ax, self.p_win)
        except Exception as e:
            print("Erro no cálculo de semblance:", e)
            self.semb = np.zeros((self.data_center.shape[0], len(self.v_ax)), dtype=np.float32)

        vm_s = np.percentile(self.semb, 99.5) if np.any(self.semb) else 1.0
        self.im_sem.set_data(self.semb)
        self.im_sem.set_extent([self.p_vmin, self.p_vmax, self.t_ax[-1], 0])
        self.im_sem.set_clim(0, vm_s)
        self.ax_sem.set_xlim(self.p_vmin, self.p_vmax)

        self.im_gat.set_data(self.data_center)
        self.im_gat.set_extent([off[0], off[-1], self.t_ax[-1], 0])
        self.im_gat.set_clim(-1, 1)
        self.ax_gat.set_xlim(off[0], off[-1])

        # NMO preview
        picks_here = list(self.picks_db.get(cmp_key, []))  # copy to avoid aliasing
        # separate semblance-type picks (time, vel) and gather-type picks (time, offset)
        sembl_picks = [p for p in picks_here if p[1] > 1000]  # heuristic: velocity picks > 1000
        if sembl_picks:
            ts, vs = zip(*semblance_picks)
            v_curve = np.interp(self.t_ax, ts, vs, left=vs[0], right=vs[-1]).astype(np.float32)
        else:
            v_curve = np.full(Nt, 2000.0, dtype=np.float32)

        d_nmo = run_nmo_fast(self.data_center, self.offsets, dt, v_curve, STRETCH_LIM)
        self.im_nmo.set_data(d_nmo)

        # update picks display: remove previous hyp lines & markers
        for l in self.hyp_lines:
            try: l.remove()
            except: pass
        self.hyp_lines = []
        for t0, vrms in sembl_picks:
            tx = np.sqrt(t0*t0 + (self.offsets**2)/(vrms**2))
            try:
                l, = self.ax_gat.plot(self.offsets, tx, 'r', lw=1.2, alpha=0.8)
                self.hyp_lines.append(l)
            except Exception:
                pass

        # gather-type picks markers (time, offset)
        for m in getattr(self, 'pick_markers', []):
            try: m.remove()
            except: pass
        self.pick_markers = []
        for p in picks_here:
            # if second value is within offsets range => likely a gather pick
            if p[1] >= np.min(self.offsets)-1e-6 and p[1] <= np.max(self.offsets)+1e-6:
                t_pick, off_pick = p
                try:
                    mk, = self.ax_gat.plot(off_pick, t_pick, 'mo', markersize=6)
                    self.pick_markers.append(mk)
                except Exception:
                    pass

        # update cmp bar markers colors
        for i, key in enumerate(self.all_cmps):
            mk = self.cmp_markers[i]
            if len(self.picks_db[int(key)]) > 0:
                mk.set_color('green')
            else:
                mk.set_color('lightgray')
        # highlight current
        idxc = np.searchsorted(self.all_cmps, self.curr_cmp)
        try:
            self.cmp_markers[int(idxc)].set_color('red')
        except Exception:
            pass

        # update title / text box
        self.ax_sem.set_title("Semblance")
        self.ax_gat.set_title(f"Input (Muted) — CMP {self.curr_cmp}")
        self.txt_cmp.set_val(str(int(self.curr_cmp)))
        self.fig.canvas.draw_idle()

    # ---------------- picks management ----------------

    def clear_picks(self, event):
        self.picks_db[int(self.curr_cmp)] = []
        self.load_cmp(self.curr_cmp)

    def save_picks_disk(self, event):
        rows = []
        for c, picks in self.picks_db.items():
            for p in picks:
                rows.append({'CMP': int(c), 'Field1': p[0], 'Field2': p[1]})
        try:
            pd.DataFrame(rows).to_csv(F_OUT_PICKS, index=False)
            print(f"Picks salvos em {F_OUT_PICKS}")
        except Exception as e:
            print("Erro salvando picks:", e)

    # ---------------- export model & NMO ----------------

    def export_data(self, event):
        print("\n--- EXPORTANDO ---")
        # save picks first
        self.save_picks_disk(None)
        if not os.path.exists(F_OUT_PICKS):
            print("Nenhum pick para exportar.")
            return
        dfp = pd.read_csv(F_OUT_PICKS)
        if dfp.empty:
            print("Arquivo de picks vazio.")
            return

        # build dense profiles per CMP (expect Field1=time, Field2=vel)
        picked_cmps = np.sort(dfp['CMP'].unique())
        dense_profiles = {}
        for c in picked_cmps:
            p = dfp[dfp['CMP']==c].sort_values('Field1')
            times = p['Field1'].values
            vels  = p['Field2'].values
            if len(times) < 2:
                v_curve = np.full(Nt, 2000.0, dtype=np.float32)
            else:
                v_curve = np.interp(self.t_ax, times, vels, left=vels[0], right=vels[-1]).astype(np.float32)
            dense_profiles[int(c)] = v_curve

        # interpolate across all_cmps
        self.vel_model_2d = np.zeros((Nt, len(self.all_cmps)), dtype=np.float32)
        for i, cmp_curr in enumerate(self.all_cmps):
            if int(cmp_curr) in dense_profiles:
                self.vel_model_2d[:, i] = dense_profiles[int(cmp_curr)]
            else:
                idx_pos = np.searchsorted(picked_cmps, cmp_curr)
                if idx_pos == 0:
                    self.vel_model_2d[:, i] = dense_profiles[int(picked_cmps[0])]
                elif idx_pos == len(picked_cmps):
                    self.vel_model_2d[:, i] = dense_profiles[int(picked_cmps[-1])]
                else:
                    c1 = int(picked_cmps[idx_pos-1]); c2 = int(picked_cmps[idx_pos])
                    w = (cmp_curr - c1) / (c2 - c1) if (c2-c1)!=0 else 0.0
                    self.vel_model_2d[:, i] = (1.0-w)*dense_profiles[c1] + w*dense_profiles[c2]
        try:
            self.vel_model_2d.astype(np.float32).tofile(F_OUT_VEL)
            print("Modelo Vrms salvo:", F_OUT_VEL)
        except Exception as e:
            print("Erro ao salvar Vrms:", e)
            return

        # Write NMO corrected traces into memmap for QC
        try:
            with open(F_OUT_NMO, 'wb') as f:
                f.seek(len(self.headers) * Nt * 4 - 1)
                f.write(b'\0')
            fp_nmo = np.memmap(F_OUT_NMO, dtype='float32', mode='r+', shape=(Nt, len(self.headers)), order='F')
            fp_raw = np.memmap(F_BIN_IN, dtype='float32', mode='r', shape=(Nt, len(self.headers)), order='F')

            for i, cmp_val in enumerate(tqdm(self.all_cmps, desc="Export NMO")):
                raw, off, idxs = load_cmp_from_disk(self.headers, fp_raw, int(cmp_val), CMP_BIN_SIZE)
                if raw is None: continue
                muted = process_display(raw, off, dt, self.p_gain)
                nmo = run_nmo_fast(muted, off, dt, self.vel_model_2d[:, i], STRETCH_LIM)
                fp_nmo[:, idxs] = nmo

            del fp_nmo, fp_raw
            print("NMO export concluído:", F_OUT_NMO)
        except Exception as e:
            print("Erro export NMO:", e)

        # Optionally show supergather from picked cmps
        picked_keys = [int(x) for x in picked_cmps]
        superg, off = build_supergather_from_picks(self.cmp_lookup, self.memmap, picked_keys, self.t_ax)
        if superg is not None:
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.imshow(superg, aspect='auto', extent=[off[0], off[-1], self.t_ax[-1], 0], cmap='gray')
            ax2.set_title('Supergather Final (NMO mean)')
            ax2.set_xlabel('Offset (m)'); ax2.set_ylabel('Tempo (s)')
            plt.show()

# ---------------- RUN ----------------

if __name__ == "__main__":
    app = VelocitySuiteNoSlider()
    plt.show()
